"""Section-aware hybrid retrieval: dense cosine search plus BM25, fused by rank."""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    return re.findall(r'\w+', text.lower(), re.UNICODE)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class Retriever:
    def catalog_overview(self):
        """Public catalog-wide evidence; individual search hits cannot describe the whole range."""
        groups = {}
        for row in read_csv(self.dataset / 'data/products.csv'):
            category = row['return_category']
            examples = groups.setdefault(category, [])
            name = row['title'].split(',')[0].strip()
            if name not in examples and len(examples) < 2:
                examples.append(name)
        return {'id':'products.csv:overview', 'file':'products.csv', 'doc_type':'catalog_summary',
                'status':'current', 'text':json.dumps({
                    'basis':'All rows in the supplied product catalog, grouped by return_category.',
                    'categories_and_example_products':groups,
                    'scope':'Catalog snapshot only; not live stock or an exhaustive list of every product the store may offer.'
                }, ensure_ascii=False)}

    def __init__(self, dataset, cache):
        self.dataset, self.cache = Path(dataset), Path(cache)
        self.chunks = []
        for file in sorted((self.dataset / 'kb').glob('*.md')):
            text = file.read_text(encoding='utf-8-sig')
            meta = {}
            if text.startswith('---'):
                for line in text.split('---', 2)[1].splitlines():
                    if ':' in line:
                        k, v = line.split(':', 1)
                        meta[k.strip()] = v.strip()
            pieces = re.split(r'^##\s+\[([^\]]+)\]\s*([^\n]*)\n', text, flags=re.M)
            for i in range(1, len(pieces), 3):
                sid, title, body = pieces[i:i + 3]
                self.chunks.append({'id': sid, 'file': file.name, **meta,
                                    'text': title.strip() + '\n' + body.strip()})
        for product in read_csv(self.dataset / 'data/products.csv'):
            self.chunks.append({'id': 'products.csv:' + product['sku'], 'file': 'products.csv',
                                'doc_type': 'catalog', 'status': 'current',
                                'text': json.dumps(product, ensure_ascii=False)})
        self.documents = [Counter(tokens(c['text'])) for c in self.chunks]
        self.df = Counter(t for doc in self.documents for t in doc)
        self.avg_len = sum(sum(d.values()) for d in self.documents) / len(self.documents)
        self.fingerprint = hashlib.sha256(json.dumps(self.chunks, sort_keys=True).encode()).hexdigest()
        self.vectors = None
        self.embedding_model = 'gemini-embedding-2'
        if self.cache.exists():
            saved = json.loads(self.cache.read_text(encoding='utf-8'))
            if saved.get('fingerprint') == self.fingerprint:
                self.vectors = saved['vectors']
                self.embedding_model = saved.get('embedding_model','gemini-embedding-001')

    def build(self, client):
        self.embedding_model = client.embedding_model
        vectors = client.embed([f"{c['id']} {c.get('title', '')} {c['text']}" for c in self.chunks])
        if len(vectors) != len(self.chunks):
            raise ValueError('Incomplete embedding index; old index was preserved.')
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps({'fingerprint': self.fingerprint, 'vectors': vectors,
                                         'embedding_model':self.embedding_model}), encoding='utf-8')
        self.vectors = vectors

    def search(self, query, client=None, k=12):
        lexical = []
        n = len(self.chunks)
        for index, doc in enumerate(self.documents):
            score = 0
            for t in set(tokens(query)):
                f = doc[t]
                if f:
                    idf = math.log(1 + (n - self.df[t] + .5) / (self.df[t] + .5))
                    score += idf * f * 2.2 / (f + 1.2 * (.25 + .75 * sum(doc.values()) / self.avg_len))
            lexical.append((score, index))
        ranks = [sorted(lexical, reverse=True)]
        if self.vectors is not None and client:
            client.embedding_model = self.embedding_model
            vector = client.embed([query], 'RETRIEVAL_QUERY')[0]
            qnorm = math.sqrt(sum(x*x for x in vector)) or 1
            dense = []
            for i, v in enumerate(self.vectors):
                similarity = sum(a*b for a, b in zip(v, vector)) / (qnorm * (math.sqrt(sum(x*x for x in v)) or 1))
                dense.append((similarity, i))
            ranks.append(sorted(dense, reverse=True))
        fused = Counter()
        for ranking in ranks:
            for rank, (score, i) in enumerate(ranking):
                if score > 0:
                    fused[i] += 1 / (60 + rank)
        # Product rows are a separate evidence type: many near-identical catalog rows
        # must not crowd out the policies explaining their warranty/return conditions.
        skus = {s.upper() for s in re.findall(r'\b[A-Za-z]+-\d{4}\b',query)}
        exact = [i for i,c in enumerate(self.chunks) if c['id'].removeprefix('products.csv:') in skus
                 and c.get('doc_type') == 'catalog']
        ids = list(exact)
        catalog_count = len(exact)
        for i, score in fused.most_common():
            if i in ids:
                continue
            if self.chunks[i].get('doc_type') == 'catalog':
                if skus or catalog_count >= 2:
                    continue
                catalog_count += 1
            ids.append(i)
            if len(ids) >= k:
                break
        # Always expose current successor alongside a retrieved historical/low-authority source.
        # Entire short current policy docs provide the surrounding exceptions as context.
        files = {self.chunks[i]['file'] for i in ids if self.chunks[i].get('doc_type') == 'policy'}
        if any(self.chunks[i]['file'].startswith('pol_returns') for i in ids):
            files.add('pol_returns_v4.md')
        result = [self.chunks[i] for i in ids]
        seen = {c['id'] for c in result}
        for c in self.chunks:
            if c['file'] in files and c.get('status') == 'current' and c['id'] not in seen:
                result.append(c)
                seen.add(c['id'])
        # Public serviceability data is a structured lookup, not semantic similarity.
        # Keep it separate from customer records and do not require identity checks.
        pincodes = set(re.findall(r'(?<!\d)\d{6}(?!\d)',query))
        if pincodes:
            for row in read_csv(self.dataset / 'data/pincodes.csv'):
                if row['pincode'] in pincodes:
                    result.append({'id':'pincodes.csv:' + row['pincode'], 'file':'pincodes.csv',
                                   'doc_type':'reference', 'status':'current',
                                   'text':json.dumps(row,ensure_ascii=False)})
        # A small catalog-wide summary makes broad store questions answerable from
        # data, without changing the existing embedding index or inventing stock.
        result.append(self.catalog_overview())
        return result
