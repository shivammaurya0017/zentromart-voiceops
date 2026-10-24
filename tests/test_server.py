import http.client
import threading
import unittest
from http.server import ThreadingHTTPServer
from server import Handler


class LocalPageTests(unittest.TestCase):
    def test_fresh_page_disables_caching_and_contains_recording_binding(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        connection=http.client.HTTPConnection(*server.server_address,timeout=5)
        try:
            connection.request('GET','/')
            response=connection.getresponse()
            html=response.read().decode('utf-8')
            self.assertEqual(response.status,200)
            self.assertIn('no-store',response.getheader('Cache-Control'))
            self.assertIn('no-cache',response.getheader('Cache-Control'))
            self.assertIn("addEventListener('click',record)",html)
            self.assertNotIn('onclick="record()"',html)
            self.assertNotIn('__TOKEN__',html)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
