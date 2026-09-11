"""Opt-in local model smoke test: python tests/manual_description_smoke.py.

Uses a synthetic image, writes QA artifacts under .description-qa, and unloads
only the model instance it loads. Requires LM Studio's server on localhost:1234.
"""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from PySide6 import QtCore, QtGui, QtWidgets
from unittest.mock import patch

import main
from cb_color_correct.description import DescriptionSettings, LocalModelClient
from cb_color_correct.theme import apply_ableton_theme


class MemorySettings:
    def __init__(self):
        self.values = {}
    def value(self, key, default=None):
        return self.values.get(key, default)
    def setValue(self, key, value):
        self.values[key] = value
    def sync(self):
        pass


app = QtWidgets.QApplication([])
# Qt's Windows offscreen platform may not discover system fonts automatically.
font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
if font_path.is_file():
    font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
    families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
    if families:
        app.setFont(QtGui.QFont(families[0], 10))
apply_ableton_theme(app)
with patch.object(main.QtCore, "QSettings", return_value=MemorySettings()):
    window = main.MainWindow()
out = Path(".description-qa")
out.mkdir(exist_ok=True)
image = Image.new("RGB", (960, 640), "#f4ecd8")
draw = ImageDraw.Draw(image)
draw.ellipse((100, 190, 330, 420), fill="#d93030")
draw.rectangle((380, 190, 610, 420), fill="#237bc5")
draw.polygon([(760, 170), (645, 420), (885, 420)], fill="#339b4c")
draw.text((320, 490), "COLOR AND SHAPE", fill="#222222", font_size=32)
image.save(out / "source.png")
rgb = np.asarray(image)
window._loaded = main.LoadedImage(out / "source.png", rgb, rgb)
window._original_preview_pixmap = main.QtGui.QPixmap.fromImage(main.rgb8_to_qimage(rgb))
window._apply_current()
window.resize(1200, 800)
window.show()
panel = window.description_panel
window.sidebar_toggle.setChecked(True)
window.sidebar_tabs.setCurrentWidget(window.description_scroll)
if "--render-only" in sys.argv:
    panel.output.setPlainText(json.loads((out / "result.json").read_text(encoding="utf-8"))["description"])
    QtCore.QTimer.singleShot(150, app.quit)
    app.exec()
    window.grab().save(str(out / "panel.png"))
    panel.options.show()
    app.processEvents()
    window._refit_pixmap()
    window.grab().save(str(out / "settings.png"))
    window.close()
    sys.exit(0)
measurements = []
timer = QtCore.QTimer()
def sample():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode == 0:
        measurements.append(int(proc.stdout.strip().splitlines()[0]))
timer.timeout.connect(sample)
timer.start(1000)
sample()
start = time.monotonic()
panel.start("describe")
loop = QtCore.QEventLoop()
panel.idle.connect(loop.quit)
loop.exec()
timer.stop()
app.processEvents()
text = panel.output.toPlainText()
window.grab().save(str(out / "panel.png"))
panel.options.show()
app.processEvents()
window._refit_pixmap()
window.grab().save(str(out / "settings.png"))
report = {"elapsed_seconds": round(time.monotonic() - start, 2), "status": panel.status.text(),
          "description": text, "gpu_baseline_mib": measurements[0], "gpu_peak_sampled_mib": max(measurements)}
print(json.dumps(report, indent=2), flush=True)
(out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
client = LocalModelClient(DescriptionSettings(), lambda: False)
for (_url, _model), instance in panel.owned_instances.items():
    client.request("/api/v1/models/unload", {"instance_id": instance}, timeout=60)
window.close()
app.processEvents()
sys.exit(0 if text else 1)
