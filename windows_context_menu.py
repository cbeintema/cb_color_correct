"""Install/remove the current user's Explorer image command."""
import argparse
import ctypes
from pathlib import Path
import winreg


EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    python = root / ".venv" / "Scripts" / "pythonw.exe"
    if not args.remove and not python.is_file():
        parser.error("Run run.bat first to create the app's virtual environment.")
    command = f'"{python}" "{root / "main.py"}" "%1"'
    for extension in EXTENSIONS:
        key = rf"Software\Classes\SystemFileAssociations\{extension}\shell\CBColorCorrect"
        if args.remove:
            for target in (key + r"\command", key):
                try:
                    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, target)
                except FileNotFoundError:
                    pass
        else:
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as handle:
                winreg.SetValueEx(handle, "", 0, winreg.REG_SZ, "Open in CB Color Correct")
                winreg.SetValueEx(handle, "MultiSelectModel", 0, winreg.REG_SZ, "Single")
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key + r"\command") as handle:
                winreg.SetValueEx(handle, "", 0, winreg.REG_SZ, command)
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)
    print("Context menu removed." if args.remove else "Context menu installed for this user.")


if __name__ == "__main__":
    main()
