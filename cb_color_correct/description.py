"""Local LM Studio image descriptions; network objects live in the worker thread."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtNetwork

from .censor import CensorCircle, apply_censor_blur
from .image_ops import FilterParams, process_rgb8_stack

DEFAULT_MODEL = "qwen3.5-9b-the-defiant-fable-uncensored-heretic-neo-imatrix-max-mtp"
DEFAULT_PROMPT = (
    "Write an 80–150 word artwork description suitable for a DeviantArt post. "
    "Describe the visible subject, composition, colors, lighting, and mood in natural prose. "
    "Stay grounded in the image; do not invent backstory, names, or how it was created. "
    "Return only the description, without a heading, hashtags, or introductory commentary."
)


class DescriptionError(RuntimeError):
    pass


class DescriptionCancelled(DescriptionError):
    pass


def local_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port if parsed.port is not None else 1234
        if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or not 1 <= port <= 65535):
            raise ValueError()
    except ValueError:
        raise DescriptionError("Use a local server address such as http://127.0.0.1:1234.") from None
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{port}"


@dataclass(frozen=True)
class DescriptionSettings:
    url: str = "http://127.0.0.1:1234"
    model: str = DEFAULT_MODEL
    token: str = ""
    prompt: str = DEFAULT_PROMPT
    context_length: int = 8192


@dataclass(frozen=True)
class ImageSnapshot:
    rgb8: np.ndarray
    base: FilterParams
    adjustments: FilterParams
    strength: float
    circles: tuple[CensorCircle, ...] = ()
    blur_radius: float = 24

    def encode(self) -> str:
        rendered = process_rgb8_stack(self.rgb8, [self.base, self.adjustments], self.strength)
        if self.circles:
            rendered = apply_censor_blur(rendered, self.circles, self.blur_radius)
        with Image.fromarray(rendered) as image:
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class LocalModelClient:
    def __init__(self, settings: DescriptionSettings, cancelled: Callable[[], bool]) -> None:
        self.settings = settings
        self.url = local_url(settings.url)
        self.cancelled = cancelled
        self.manager = QtNetwork.QNetworkAccessManager()
        self.manager.setProxy(QtNetwork.QNetworkProxy(QtNetwork.QNetworkProxy.ProxyType.NoProxy))

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise DescriptionCancelled("Cancelled. LM Studio may still be finishing the server operation.")

    def request(self, path: str, payload: dict | None = None, timeout: float = 15) -> dict:
        self.check_cancelled()
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(self.url + path))
        request.setAttribute(QtNetwork.QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QtNetwork.QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        if self.settings.token:
            request.setRawHeader(b"Authorization", ("Bearer " + self.settings.token).encode())
        reply = (self.manager.get(request) if payload is None else
                 self.manager.post(request, json.dumps(payload).encode()))
        loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        deadline = time.monotonic() + timeout
        timed_out = False

        def poll() -> None:
            nonlocal timed_out
            timed_out = time.monotonic() >= deadline
            if self.cancelled() or timed_out:
                reply.abort()
                loop.quit()

        timer.timeout.connect(poll)
        reply.finished.connect(loop.quit)
        timer.start(50)
        try:
            if not reply.isFinished():
                loop.exec()
            self.check_cancelled()
            if timed_out:
                raise DescriptionError("LM Studio timed out. Check its activity and GPU memory before retrying.")
            status = reply.attribute(QtNetwork.QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if status in (401, 403):
                raise DescriptionError("LM Studio requires a valid API token. Enter it in Description settings.")
            if status == 404:
                raise DescriptionError("Model or API not found. Refresh models and use LM Studio 0.4.0 or later.")
            if status is not None and not 200 <= int(status) < 300:
                # Do not echo server bodies: they can include image payloads or credentials.
                raise DescriptionError(f"LM Studio returned HTTP {status}. Check its server log, vision projector, and free GPU memory.")
            if reply.error() != QtNetwork.QNetworkReply.NetworkError.NoError:
                raise DescriptionError("Cannot reach LM Studio. Start its local server and check the address and port.")
            try:
                result = json.loads(bytes(reply.readAll()))
            except (ValueError, UnicodeError):
                raise DescriptionError("LM Studio returned invalid JSON.") from None
            if not isinstance(result, dict):
                raise DescriptionError("LM Studio returned an unexpected response.")
            return result
        finally:
            timer.stop()
            reply.deleteLater()

    def models(self) -> list[dict]:
        models = self.request("/api/v1/models").get("models")
        if not isinstance(models, list):
            raise DescriptionError("LM Studio did not return a model list.")
        return [m for m in models if isinstance(m, dict) and m.get("type") == "llm"
                and m.get("capabilities", {}).get("vision") is True and isinstance(m.get("key"), str)]

    def describe(self, model: dict, image: str) -> str:
        payload = {
            "model": model["key"], "input": [
                {"type": "text", "content": self.settings.prompt},
                {"type": "image", "data_url": image}],
            "store": False, "stream": False, "integrations": [],
            "temperature": 0.7, "max_output_tokens": 512,
        }
        reasoning = model.get("capabilities", {}).get("reasoning") or {}
        if "off" in reasoning.get("allowed_options", []):
            payload["reasoning"] = "off"
        output = self.request("/api/v1/chat", payload, timeout=300).get("output")
        if not isinstance(output, list):
            raise DescriptionError("LM Studio returned no description.")
        text = "\n\n".join(item["content"].strip() for item in output
                           if isinstance(item, dict) and item.get("type") == "message"
                           and isinstance(item.get("content"), str)).strip()
        if not text:
            raise DescriptionError("The model returned no description. Check its reasoning settings or try a shorter prompt.")
        return text


class DescriptionWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    result = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    loaded = QtCore.Signal(str)

    def __init__(self, settings: DescriptionSettings, action: str = "describe",
                 snapshot: ImageSnapshot | None = None, instance_id: str = "") -> None:
        super().__init__()
        self.settings, self.action, self.snapshot = settings, action, snapshot
        self.instance_id = instance_id

    def run(self) -> None:
        client = None
        try:
            client = LocalModelClient(self.settings, self.isInterruptionRequested)
            if self.action == "unload":
                if not self.instance_id:
                    raise DescriptionError("No model loaded by this app to unload.")
                client.request("/api/v1/models/unload", {"instance_id": self.instance_id}, timeout=60)
                self.result.emit(None)
                return
            self.status.emit("Connecting to LM Studio…")
            models = client.models()
            if self.action == "refresh":
                self.result.emit(models)
                return
            model = next((m for m in models if m["key"] == self.settings.model), None)
            if model is None:
                raise DescriptionError("Selected vision model was not found. Refresh models and check that its mmproj file is installed.")
            if not model.get("loaded_instances"):
                self.status.emit("Loading model…")
                loaded = client.request("/api/v1/models/load", {
                    "model": model["key"], "context_length": self.settings.context_length}, timeout=180)
                instance = loaded.get("instance_id")
                if not isinstance(instance, str) or not instance:
                    raise DescriptionError("LM Studio did not return a loaded model instance.")
                self.loaded.emit(instance)
            client.check_cancelled()
            if self.action == "load":
                self.result.emit(None)
                return
            if self.snapshot is None:
                raise DescriptionError("Load an image first.")
            self.status.emit("Preparing edited image…")
            image = self.snapshot.encode()
            client.check_cancelled()
            self.status.emit("Writing description…")
            text = client.describe(model, image)
            client.check_cancelled()
            self.result.emit(text)
        except DescriptionError as exc:
            self.failed.emit(str(exc))
        except Exception:
            self.failed.emit("Description failed unexpectedly. Check the image, LM Studio model, and available memory.")
        finally:
            # Destroy network objects in their owning thread.
            if client is not None:
                del client
