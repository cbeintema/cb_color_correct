"""Read-only checks for metadata exposed by Pillow and FFprobe."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import warnings

from PIL import ExifTags, Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg", ".mts", ".m2ts", ".3gp")
SCOPE = ("Checks embedded metadata exposed by Pillow (images) or FFprobe (videos). "
         "Basic dimensions, compression, playback and display properties are excluded. "
         "Does not inspect filesystem timestamps, sidecar files, or every video frame.")
# TIFF fields required to decode/layout pixels; all other fields are reported.
TIFF_STRUCTURE = {254, 255, 256, 257, 258, 259, 262, 266, 273, 277, 278,
                  279, 282, 283, 284, 292, 293, 296, 317, 320, 322, 323,
                  324, 325, 338, 339, 347, 513, 514, 529, 530, 531, 532}
IMAGE_STRUCTURE = {"jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi",
                   "aspect", "compression", "transparency", "interlace", "duration",
                   "loop", "background", "bbox", "blend", "disposal", "default_image", "resolution"}


@dataclass
class CheckResult:
    entries: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.entries:
            return "Metadata found" + (" — check incomplete" if self.issues else "")
        return "Couldn't complete check" if self.issues else "Clean — no metadata detected"


def describe(value) -> str:
    if isinstance(value, bytes):
        return f"{len(value):,} bytes"
    return str(value)[:600]


def check_image(path: Path) -> CheckResult:
    result = CheckResult()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        with Image.open(path) as image:
            for frame in range(256):
                if frame:
                    try:
                        image.seek(frame)
                    except EOFError:
                        break
                exif = dict(image.getexif())  # TIFF loading can consume orientation.
                image.load()  # Includes PNG text chunks after the pixel data.
                exif.update(image.getexif())
                prefix = f"Frame {frame + 1}: "
                for tag, value in exif.items():
                    if image.format == "TIFF" and tag in TIFF_STRUCTURE:
                        continue
                    name = ExifTags.TAGS.get(tag, str(tag))
                    result.entries.append(f"{prefix}EXIF {name}: {describe(value)}")
                for key, value in image.info.items():
                    structural = key in IMAGE_STRUCTURE or (image.format == "WEBP" and key == "timestamp")
                    if (not structural or key in getattr(image, "text", {})) and value not in (None, b"", ""):
                        result.entries.append(f"{prefix}{key}: {describe(value)}")
                # JPEG APP markers can carry IPTC, XMP or vendor-specific data.
                for marker, value in getattr(image, "applist", []):
                    if marker == "APP0" and value.startswith(b"JFIF\x00"):
                        continue
                    if marker == "APP14" and value.startswith(b"Adobe"):
                        continue  # Decoder color transform.
                    result.entries.append(f"{prefix}{marker}: {describe(value)}")
                if not getattr(image, "is_animated", False) and image.format != "TIFF":
                    break
            else:
                result.issues.append("Frame limit reached; remaining frames were not checked.")
        result.issues.extend(str(item.message) for item in notices)
    return result


def parse_video(data: dict) -> CheckResult:
    if not data.get("format") or not data.get("streams"):
        raise ValueError("FFprobe returned no readable media streams.")
    result = CheckResult()
    is_mp4 = bool({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"} &
                  set(data["format"].get("format_name", "").split(",")))
    sections = [("File", data["format"])]
    sections += [(f"Stream {s.get('index', i)}", s) for i, s in enumerate(data["streams"])]
    sections += [(f"Chapter {i + 1}", c) for i, c in enumerate(data.get("chapters", []))]
    for name, section in sections:
        for key, value in section.get("tags", {}).items():
            if name == "File" and key in {"major_brand", "minor_version", "compatible_brands"}:
                continue
            if is_mp4 and name.startswith("Stream "):
                if (key, value) in {("language", "und"), ("vendor_id", "[0][0][0][0]")}:
                    continue
                default_handler = {"video": "VideoHandler", "audio": "SoundHandler"}.get(section.get("codec_type"))
                if key == "handler_name" and value == default_handler:
                    continue
            result.entries.append(f"{name} / {key}: {describe(value)}")
        for side_data in section.get("side_data_list", []):
            result.entries.append(f"{name} / side data: {describe(side_data)}")
        if section.get("codec_type") in {"attachment", "data"} or section.get("disposition", {}).get("attached_pic"):
            result.entries.append(f"{name}: embedded attachment, cover image, or data track")
    return result


def check_file(path: Path) -> CheckResult:
    try:
        path = path.resolve(strict=True)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return check_image(path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("This file type is not supported by the quick check.")
        probe = shutil.which("ffprobe")
        if not probe:
            raise RuntimeError("FFprobe was not found. Install FFmpeg and add its bin folder to PATH.")
        completed = subprocess.run(
            [probe, "-v", "error", "-show_format", "-show_streams", "-show_chapters",
             "-of", "json", str(path)], capture_output=True, encoding="utf-8", errors="replace",
            timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode or completed.stderr.strip():
            raise RuntimeError(completed.stderr.strip()[:1500] or "FFprobe could not read this file.")
        return parse_video(json.loads(completed.stdout))
    except Exception as exc:
        return CheckResult(issues=[str(exc)])
