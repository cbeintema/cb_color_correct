import base64
from io import BytesIO
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets

from cb_color_correct.description import (
    DEFAULT_MODEL, DescriptionCancelled, DescriptionError, DescriptionSettings,
    DescriptionWorker, ImageSnapshot, LocalModelClient, local_url,
)
from cb_color_correct.censor import CensorCircle, apply_censor_blur
from cb_color_correct.image_ops import FilterParams, process_rgb8_stack


MODEL = {"key": DEFAULT_MODEL, "type": "llm", "loaded_instances": [],
         "capabilities": {"vision": True, "reasoning": {"allowed_options": ["off", "on"]}}}


class DescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_loopback_only_no_paths_or_credentials(self):
        self.assertEqual(local_url("http://localhost:1234/"), "http://127.0.0.1:1234")
        for bad in ("https://example.com", "http://127.0.0.1/remote", "http://user@localhost:1234",
                    "http://127.0.0.1:99999", "http://127.0.0.1?x=1", "file:///tmp/a"):
            with self.subTest(bad=bad), self.assertRaises(DescriptionError):
                local_url(bad)

    def test_image_matches_edits_and_optional_censor(self):
        rgb = np.zeros((40, 60, 3), dtype=np.uint8)
        rgb[10:30, 20:40] = 255
        base, adjust = FilterParams(), FilterParams(exposure=0.5)
        circles = (CensorCircle(0.5, 0.5, 0.3),)
        for selected in ((), circles):
            snapshot = ImageSnapshot(rgb, base, adjust, 0.8, selected, 8)
            encoded = snapshot.encode()
            with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as image:
                expected = process_rgb8_stack(rgb, [base, adjust], 0.8)
                if selected:
                    expected = apply_censor_blur(expected, selected, 8)
                np.testing.assert_array_equal(np.asarray(image), expected)
                self.assertEqual(image.info, {})

    def test_image_dimensions_are_bounded(self):
        snap = ImageSnapshot(np.zeros((900, 1800, 3), dtype=np.uint8), FilterParams(), FilterParams(), 1)
        with Image.open(BytesIO(base64.b64decode(snap.encode().split(",", 1)[1]))) as image:
            self.assertEqual(image.size, (1280, 640))

    def test_request_has_image_and_only_final_message_is_returned(self):
        client = LocalModelClient(DescriptionSettings(), lambda: False)
        response = {"output": [{"type": "reasoning", "content": "private reasoning"},
                               {"type": "message", "content": " A blue landscape. "}]}
        with patch.object(client, "request", return_value=response) as call:
            self.assertEqual(client.describe(MODEL, "data:image/png;base64,abc"), "A blue landscape.")
            payload = call.call_args.args[1]
            self.assertEqual(payload["input"][1]["type"], "image")
            self.assertFalse(payload["store"])
            self.assertEqual(payload["reasoning"], "off")
            self.assertNotIn("previous_response_id", payload)
        with patch.object(client, "request", return_value={"output": []}), self.assertRaises(DescriptionError):
            client.describe(MODEL, "image")

    def test_load_on_demand_and_reuse(self):
        for loaded in ([], [{"id": "existing"}]):
            worker = DescriptionWorker(DescriptionSettings(), "load")
            model = dict(MODEL, loaded_instances=loaded)
            instances = []
            worker.loaded.connect(instances.append)
            with patch.object(LocalModelClient, "models", return_value=[model]), patch.object(
                LocalModelClient, "request", return_value={"instance_id": "owned"}) as request:
                worker.run()
                self.assertEqual(request.call_count, 0 if loaded else 1)
                self.assertEqual(instances, [] if loaded else ["owned"])

    def test_missing_vision_never_sends_image(self):
        worker = DescriptionWorker(DescriptionSettings())
        failures = []
        worker.failed.connect(failures.append)
        with patch.object(LocalModelClient, "models", return_value=[]), patch.object(LocalModelClient, "describe") as describe:
            worker.run()
            describe.assert_not_called()
        self.assertIn("vision model", failures[0])


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/slow":
                    time.sleep(0.4)
                status = {"/auth": 401, "/old": 404, "/oom": 500, "/redirect": 302}.get(self.path, 200)
                self.send_response(status)
                if status == 302:
                    self.send_header("Location", "http://example.com")
                self.end_headers()
                try:
                    self.wfile.write(b"not json" if self.path == "/bad" else b'{"models": []}')
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            def log_message(self, *_args):
                pass
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self, cancelled=lambda: False):
        return LocalModelClient(DescriptionSettings(url=f"http://127.0.0.1:{self.server.server_port}"), cancelled)

    def test_json_errors_and_redirect_rejection(self):
        client = self.client()
        self.assertEqual(client.request("/ok"), {"models": []})
        for path, message in (("/auth", "token"), ("/old", "0.4.0"), ("/oom", "GPU"),
                              ("/redirect", "302"), ("/bad", "JSON")):
            with self.subTest(path=path), self.assertRaisesRegex(DescriptionError, message):
                client.request(path)

    def test_cancel_and_total_timeout_abort_network(self):
        start = time.monotonic()
        client = self.client(lambda: time.monotonic() - start > 0.08)
        with self.assertRaises(DescriptionCancelled):
            client.request("/slow")
        self.assertLess(time.monotonic() - start, 0.35)
        with self.assertRaisesRegex(DescriptionError, "timed out"):
            self.client().request("/slow", timeout=0.05)


if __name__ == "__main__":
    unittest.main()
