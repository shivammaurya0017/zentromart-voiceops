"""Honest heuristic practice evaluation; gold labels never enter agent context."""
import argparse
import getpass
import json
import os
import re
import time
from pathlib import Path
from src.api import Gemini, APIError
from src.retrieval import Retriever, read_csv
from src.agent import Agent, ToolLayer
from run import DEFAULT_DATASET, ROOT


def canonical(text):
    text = ''.join(str(int(c)) if c.isdecimal() else c for c in text.casefold())
    text = re.sub(r'(?<=\d),(?=\d)','',text)
    values = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(),range(20)))
    values.update(dict(zip(('twenty thirty forty fifty sixty seventy eighty ninety').split(),range(20,100,10))))
    scales = {'hundred':100,'thousand':1000,'lakh':100000,'million':1000000}
    pattern = r'\b(?:'+'|'.join(values.keys() | scales.keys())+r')(?:[ -]+(?:'+'|'.join(values.keys() | scales.keys())+r'))*\b'
    def replace(match):
        total,group = 0,0
        for word in re.split('[ -]+',match.group()):
            if word in values:
                group += values[word]
            elif word == 'hundred':
                group = (group or 1)*100
            else:
                total += (group or 1)*scales[word]
                group = 0
        return str(total+group)
    text = re.sub(pattern,replace,text)
    return re.sub(r'\s+', ' ', text).strip()


def facts_score(row, result):
    reply = canonical(result['reply'])
    def contains(alternative):
        alternative = canonical(alternative)
        if alternative.isdigit():
            return bool(re.search(r'(?<!\d)' + re.escape(alternative) + r'(?!\d)',reply))
        return alternative in reply
    facts = [f.strip() for f in row.get('must_include','').split(' ; ') if f.strip()]
    passed = sum(any(contains(alt) for alt in fact.split(' / ')) for fact in facts)
    forbidden = [s.strip() for s in row.get('must_not_include','').split(' ; ') if s.strip()]
    gold = {s.strip() for s in row.get('gold_evidence','').split(' ; ') if s.strip()}
    retrieved = {c['id'] for c in result['retrieved_chunks']}
    acceptable = {s.strip() for s in row.get('acceptable_decisions','').split(' ; ') if s.strip()}
    acceptable.add(row['expected_decision'])
    expected_tool_text = row.get('expected_tool_calls')
    actual_tools = [call['name'] for call in result.get('tool_calls',[])]
    expected_tools = re.findall(r'\b(lookup_order|create_ticket)\s*\(',expected_tool_text or '')
    return {'decision_match':result['decision'] in acceptable,
            'required_fact_hits':passed, 'required_fact_count':len(facts),
            'forbidden_phrase_hit':any(canonical(s) in reply for s in forbidden),
            'gold_retrieved':len(gold & retrieved), 'gold_count':len(gold),
            'tool_names_match':(sorted(expected_tools)==sorted(actual_tools)) if expected_tool_text is not None else None,
            'note':'Literal heuristic only; equivalent dates, translations and negation need manual review.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=DEFAULT_DATASET)
    p.add_argument('--mode', choices=['baseline','rag','conversation'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default=os.getenv('GEMINI_MODEL','gemini-3.5-flash-lite'))
    p.add_argument('--limit', type=int, default=0, help='Smoke-test subset; never report as full evaluation')
    p.add_argument('--delay', type=float, default=5, help='Seconds between turns for free-tier limits')
    args = p.parse_args()
    client = Gemini(os.getenv('GEMINI_API_KEY') or getpass.getpass('Gemini key (hidden): '), args.model)
    retriever = Retriever(args.dataset, ROOT / '.cache/index.json')
    if args.mode != 'baseline' and not retriever.vectors:
        retriever.build(client)
    tools = ToolLayer(args.dataset, ROOT / '.runtime/evaluation')
    conversations = args.mode == 'conversation'
    rows = read_csv(Path(args.dataset) / ('practice/conversations.csv' if conversations else 'practice/questions.csv'))
    if conversations:
        rows.sort(key=lambda r:(r['conversation_id'],int(r['turn'])))
    if args.limit:
        rows = rows[:args.limit]
    records, agents = [], {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        identity = row['conversation_id'] if conversations else row['id']
        agent = agents.setdefault(identity, Agent(client,retriever,tools,baseline=args.mode == 'baseline'))
        message = row['customer_message'] if conversations else row['question']
        record = {'id':identity,'turn':row.get('turn'), 'mode':args.mode}
        try:
            result = agent.respond(message)
            record.update(result=result, checks=facts_score(row,result))
        except (APIError, ValueError) as e:
            record['error'] = str(e)
        records.append(record)
        checks = [r['checks'] for r in records if 'checks' in r]
        report = {'mode':args.mode, 'model':client.model, 'subset_only':bool(args.limit),
            'attempted':len(records), 'successful':len(checks), 'errors':len(records)-len(checks),
            'decision_accuracy':sum(c['decision_match'] for c in checks)/len(checks) if checks else None,
            'literal_fact_recall':sum(c['required_fact_hits'] for c in checks)/max(1,sum(c['required_fact_count'] for c in checks)),
            'evidence_recall':sum(c['gold_retrieved'] for c in checks)/max(1,sum(c['gold_count'] for c in checks)),
            'forbidden_phrase_hits':sum(c['forbidden_phrase_hit'] for c in checks),
            'limitation':'Heuristic scores, not official grading. No manual answer edits. Review equivalent wording, multilingual/date facts and tool correctness.',
            'records':records}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{len(records)}/{len(rows)} {identity}: ' + ('ERROR' if 'error' in record else 'saved'))
        time.sleep(args.delay)


if __name__ == '__main__':
    main()
