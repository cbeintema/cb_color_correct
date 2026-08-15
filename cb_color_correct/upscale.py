from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import uuid
import zipfile
from typing import Any

from PySide6 import QtCore

from cb_color_correct.image_metadata import save_metadata_free_bytes


@dataclass(frozen=True)
class UpscaleSettings:
    source_path: str
    output_path: str
    comfy_url: str
    engine: str
    resize_type: str
    scale: float
    width: int
    height: int
    quality: str
    seedvr_resolution: int
    seedvr_max_resolution: int
    seedvr_seed: int
    comfyui_python: str = ""
    comfyui_main: str = ""
    comfyui_extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpscalePackagePaths:
    directory: Path
    censored_path: Path
    archive_path: Path
    upscaled_path: Path


def build_upscale_package_paths(
    source_path: str | os.PathLike[str], output_name: str | None = None
) -> UpscalePackagePaths:
    source = Path(source_path)
    package_name = output_name.strip() if output_name else ""
    if package_name and ("/" in package_name or "\\" in package_name):
        raise ValueError("Output name must be a file name without a path")
    package_name = package_name or source.stem
    directory = source.parent / package_name
    return UpscalePackagePaths(
        directory=directory,
        censored_path=directory / f"{package_name}_censored{source.suffix}",
        archive_path=directory / f"{package_name}.zip",
        upscaled_path=directory / f"{package_name}_upscaled.png",
    )


def create_upscale_zip(upscaled_path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        entry = zipfile.ZipInfo(upscaled_path.name, date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.create_system = 0
        entry.external_attr = 0
        entry.extra = b""
        entry.comment = b""
        archive.writestr(entry, upscaled_path.read_bytes())
        entry.external_attr = 0


def build_seedvr_workflow(uploaded_name: str, resolution: int, max_resolution: int, seed: int) -> dict[str, dict[str, Any]]:
    return {
        "10": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "seed": int(seed),
                "resolution": int(resolution),
                "max_resolution": int(max_resolution),
                "batch_size": 1,
                "uniform_batch_size": False,
                "color_correction": "lab",
                "temporal_overlap": 0,
                "prepend_frames": 0,
                "input_noise_scale": 0,
                "latent_noise_scale": 0,
                "offload_device": "cpu",
                "enable_debug": False,
                "image": ["17", 0],
                "dit": ["14", 0],
                "vae": ["13", 0],
            },
        },
        "13": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": "ema_vae_fp16.safetensors",
                "device": "cuda:0",
                "encode_tiled": True,
                "encode_tile_size": 1024,
                "encode_tile_overlap": 128,
                "decode_tiled": True,
                "decode_tile_size": 1024,
                "decode_tile_overlap": 128,
                "tile_debug": "false",
                "offload_device": "cpu",
                "cache_model": False,
            },
        },
        "14": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": "seedvr2_ema_7b_sharp_fp16.safetensors",
                "device": "cuda:0",
                "blocks_to_swap": 36,
                "swap_io_components": False,
                "offload_device": "cpu",
                "cache_model": False,
                "attention_mode": "sdpa",
            },
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "SeedVR", "images": ["10", 0]},
        },
        "16": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_name, "upload": "image"},
        },
        "17": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["16", 0], "alpha": ["16", 1]},
        },
    }


def build_rtx_workflow(
    uploaded_name: str,
    resize_type: str,
    scale: float,
    width: int,
    height: int,
    quality: str,
) -> dict[str, dict[str, Any]]:
    rtx_inputs: dict[str, Any] = {
        "images": ["2", 0],
        "resize_type": resize_type,
        "quality": quality,
    }
    if resize_type == "resize to width and height":
        rtx_inputs["resize_type.width"] = int(width)
        rtx_inputs["resize_type.height"] = int(height)
    else:
        rtx_inputs["resize_type.scale"] = float(scale)

    return {
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_name, "upload": "image"},
        },
        "1": {
            "class_type": "RTXVideoSuperResolution",
            "inputs": rtx_inputs,
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], "filename_prefix": "RTX"},
        },
    }


def comfy_ws_url(comfy_url: str) -> str:
    parsed = urllib.parse.urlparse(comfy_url if "://" in comfy_url else f"http://{comfy_url}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urllib.parse.urlunparse((scheme, parsed.netloc, "", "", "", ""))


class ComfyUpscaleWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    status = QtCore.Signal(str)
    log = QtCore.Signal(str)
    completed = QtCore.Signal(bool, str)

    def __init__(self, settings: UpscaleSettings) -> None:
        super().__init__()
        self.settings = settings
        self.client_id = str(uuid.uuid4())
        self._launched_process: subprocess.Popen[Any] | None = None
        self._launch_log_path: Path | None = None
        self._launch_log_file: Any | None = None

    def _close_launch_log(self) -> None:
        log_file = self._launch_log_file
        self._launch_log_file = None
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass

    def _launch_log_tail(self) -> str:
        self._close_launch_log()
        if self._launch_log_path is None:
            return ""
        try:
            return self._launch_log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        except OSError:
            return ""

    def _check_interrupted(self) -> None:
        if self.isInterruptionRequested():
            raise RuntimeError("Upscale cancelled")

    def _comfy_is_running(self, requests_module: Any) -> bool:
        try:
            requests_module.get(f"{self.settings.comfy_url}/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def _ensure_comfyui(self, requests_module: Any) -> subprocess.Popen[Any] | None:
        if self._comfy_is_running(requests_module):
            return None

        python = self.settings.comfyui_python
        main_py = self.settings.comfyui_main
        if not python or not os.path.isfile(python) or not main_py or not os.path.isfile(main_py):
            python_status = "set" if python else "empty"
            main_status = "set" if main_py else "empty"
            raise RuntimeError(
                f"Cannot reach ComfyUI at {self.settings.comfy_url}; it is not running and auto-launch paths are not valid "
                f"(Python Exe: {python_status}, main.py: {main_status}). "
                "Start it or configure valid Python Exe and ComfyUI main.py paths."
            )

        parsed = urllib.parse.urlparse(self.settings.comfy_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8000)
        self.status.emit("Launching ComfyUI headless...")
        self._launch_log_path = Path(main_py).resolve().parent / "cb_color_correct_comfyui.log"
        self.log.emit(f"ComfyUI is not reachable; launching a temporary headless server (log: {self._launch_log_path})...")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        try:
            self._launch_log_file = self._launch_log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [python, main_py, "--listen", host, "--port", port, *self.settings.comfyui_extra_args],
                cwd=os.path.dirname(os.path.abspath(main_py)),
                creationflags=creation_flags,
                stdout=self._launch_log_file,
                stderr=self._launch_log_file,
            )
        except Exception:
            self._close_launch_log()
            raise
        self._launched_process = process

        deadline = time.monotonic() + 300
        elapsed = 0
        while time.monotonic() < deadline:
            self._check_interrupted()
            time.sleep(1)
            elapsed += 1
            if process.poll() is not None:
                details = self._launch_log_tail()
                message = f"ComfyUI exited unexpectedly with code {process.returncode}"
                if details:
                    message += f". Last output from {self._launch_log_path}:\n{details}"
                raise RuntimeError(message)
            if self._comfy_is_running(requests_module):
                self.log.emit("Temporary ComfyUI server is ready")
                return process
            if elapsed % 30 == 0:
                self.log.emit(f"Still waiting for ComfyUI... ({elapsed}s)")

        raise RuntimeError("ComfyUI did not become ready within 300 seconds")

    def _build_workflow(self, uploaded_name: str) -> dict[str, dict[str, Any]]:
        if self.settings.engine == "SeedVR":
            return build_seedvr_workflow(
                uploaded_name,
                self.settings.seedvr_resolution,
                self.settings.seedvr_max_resolution,
                self.settings.seedvr_seed,
            )
        return build_rtx_workflow(
            uploaded_name,
            self.settings.resize_type,
            self.settings.scale,
            self.settings.width,
            self.settings.height,
            self.settings.quality,
        )

    def _find_output_bytes(self, requests_module: Any, prompt_id: str) -> bytes:
        response = requests_module.get(f"{self.settings.comfy_url}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        history = response.json().get(prompt_id, {})
        for node_output in history.get("outputs", {}).values():
            images = node_output.get("images", []) if isinstance(node_output, dict) else []
            if not images:
                continue
            image_meta = images[0]
            query = urllib.parse.urlencode(
                {
                    "filename": image_meta["filename"],
                    "subfolder": image_meta.get("subfolder", ""),
                    "type": image_meta.get("type", "output"),
                }
            )
            image_response = requests_module.get(f"{self.settings.comfy_url}/view?{query}", timeout=60)
            image_response.raise_for_status()
            return bytes(image_response.content)
        raise RuntimeError("ComfyUI completed without returning an output image")

    def run(self) -> None:
        websocket_connection: Any = None
        success = False
        try:
            try:
                import requests
                import websocket
            except ImportError as exc:
                raise RuntimeError(
                    "Upscaling requires requests and websocket-client. Install requirements.txt and restart."
                ) from exc

            self._check_interrupted()
            self._ensure_comfyui(requests)
            self.status.emit("Uploading image...")
            content_type = mimetypes.guess_type(self.settings.source_path)[0] or "application/octet-stream"
            with open(self.settings.source_path, "rb") as source_file:
                response = requests.post(
                    f"{self.settings.comfy_url}/upload/image",
                    files={
                        "image": (
                            os.path.basename(self.settings.source_path),
                            source_file,
                            content_type,
                        )
                    },
                    timeout=60,
                )
            response.raise_for_status()
            uploaded_name = response.json()["name"]

            self._check_interrupted()
            workflow = self._build_workflow(uploaded_name)
            response = requests.post(
                f"{self.settings.comfy_url}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=60,
            )
            if not response.ok:
                try:
                    detail: object = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(f"{response.status_code} from /prompt: {detail}")
            prompt_id = response.json()["prompt_id"]

            self.status.emit("Waiting for ComfyUI...")
            websocket_connection = websocket.WebSocket()
            websocket_connection.connect(
                f"{comfy_ws_url(self.settings.comfy_url)}/ws?clientId={self.client_id}",
                timeout=600,
            )
            websocket_connection.settimeout(1.0)
            while True:
                self._check_interrupted()
                try:
                    raw_message = websocket_connection.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(raw_message, bytes):
                    continue
                message = json.loads(raw_message)
                if message.get("type") != "executing":
                    continue
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
            websocket_connection.close()
            websocket_connection = None

            self._check_interrupted()
            self.status.emit("Downloading result...")
            output_bytes = self._find_output_bytes(requests, prompt_id)
            save_metadata_free_bytes(output_bytes, Path(self.settings.output_path))
            self.progress.emit(1)
            self.status.emit("Upscale complete")
            self.log.emit(f"Upscaled image saved to {self.settings.output_path}")
            success = True
        except Exception as exc:
            self.status.emit("Upscale failed")
            self.log.emit(str(exc))
        finally:
            if websocket_connection is not None:
                try:
                    websocket_connection.close()
                except Exception:
                    pass
            launched_process = self._launched_process
            self._launched_process = None
            if launched_process is not None:
                self.log.emit("Stopping temporary ComfyUI server...")
                try:
                    if launched_process.poll() is None:
                        launched_process.terminate()
                        launched_process.wait(timeout=10)
                except Exception:
                    try:
                        launched_process.kill()
                    except Exception:
                        pass
            self._close_launch_log()
            self.completed.emit(success, self.settings.output_path if success else "")


class ComfyUILaunchWatcher(QtCore.QThread):
    log = QtCore.Signal(str)
    completed = QtCore.Signal(bool)

    def __init__(self, python: str, main_py: str, comfy_url: str, extra_args: tuple[str, ...]) -> None:
        super().__init__()
        self.python = python
        self.main_py = main_py
        self.comfy_url = comfy_url.rstrip("/")
        self.extra_args = extra_args
        self.process: subprocess.Popen[Any] | None = None
        self._stop_requested = False
        self.ready = False
        self._launch_log_path: Path | None = None
        self._launch_log_file: Any | None = None

    def _close_launch_log(self) -> None:
        log_file = self._launch_log_file
        self._launch_log_file = None
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass

    def _launch_log_tail(self) -> str:
        self._close_launch_log()
        if self._launch_log_path is None:
            return ""
        try:
            return self._launch_log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        except OSError:
            return ""

    def shutdown(self) -> None:
        self._stop_requested = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if not self.isRunning():
            self._close_launch_log()

    def run(self) -> None:
        try:
            import requests
        except ImportError:
            self.log.emit("Launching ComfyUI requires requests. Install requirements.txt and restart.")
            self.completed.emit(False)
            return

        parsed = urllib.parse.urlparse(self.comfy_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8000)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        self._launch_log_path = Path(self.main_py).resolve().parent / "cb_color_correct_comfyui.log"
        try:
            self._launch_log_file = self._launch_log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                [self.python, self.main_py, "--listen", host, "--port", port, *self.extra_args],
                cwd=os.path.dirname(os.path.abspath(self.main_py)),
                creationflags=creation_flags,
                stdout=self._launch_log_file,
                stderr=self._launch_log_file,
            )
        except Exception as exc:
            self._close_launch_log()
            self.log.emit(f"Failed to launch ComfyUI: {exc}")
            self.completed.emit(False)
            return

        self.log.emit(f"ComfyUI headless server launching... (log: {self._launch_log_path})")
        deadline = time.monotonic() + 300
        elapsed = 0
        ready = False
        while time.monotonic() < deadline and not self._stop_requested:
            time.sleep(1)
            elapsed += 1
            if self.process.poll() is not None:
                details = self._launch_log_tail()
                message = f"ComfyUI exited unexpectedly with code {self.process.returncode}"
                if details:
                    message += f". Last output from {self._launch_log_path}:\n{details}"
                self.log.emit(message)
                if not ready:
                    self.completed.emit(False)
                return
            if not ready:
                try:
                    requests.get(f"{self.comfy_url}/system_stats", timeout=3)
                    self.log.emit("ComfyUI headless server is ready")
                    self.ready = True
                    self.completed.emit(True)
                    return
                except Exception:
                    if elapsed % 30 == 0:
                        self.log.emit(f"Still waiting for ComfyUI... ({elapsed}s)")

        if self._stop_requested:
            self.log.emit("ComfyUI launch stopped")
        elif not ready:
            self.log.emit("ComfyUI did not become ready within 300 seconds")
            self.completed.emit(False)
        if self._stop_requested and self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
        self._close_launch_log()
