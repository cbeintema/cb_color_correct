"""
Clear Meta Data - Windows Context Menu Utility
Strips metadata (EXIF, IPTC, XMP, comments, video tags, zip timestamps, NTFS Zone.Identifier)
from images, videos, and zip files.
"""

import os
import sys
import shutil
import tempfile
import zipfile
import subprocess
import traceback

# Supported file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif',
    '.bmp', '.tiff', '.tif', '.ico', '.jfif',
    '.avif', '.heic'
}

VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.mov', '.avi', '.webm',
    '.wmv', '.flv', '.m4v', '.3gp', '.ts',
    '.mts', '.m2ts', '.vob', '.ogv', '.mpg', '.mpeg'
}

ZIP_EXTENSIONS = {
    '.zip'
}

def find_ffmpeg():
    """Find ffmpeg binary path."""
    which_ffmpeg = shutil.which('ffmpeg')
    if which_ffmpeg:
        return which_ffmpeg
    # Common local paths
    fallback_paths = [
        r'C:\ffmpeg\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe',
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
    ]
    for p in fallback_paths:
        if os.path.isfile(p):
            return p
    return 'ffmpeg'

def remove_alternate_data_streams(file_path):
    """Remove NTFS Zone.Identifier (download source metadata) and other streams."""
    ads_list = [':Zone.Identifier', ':favicon', ':$DATA']
    for ads in ads_list:
        if ads == ':$DATA':
            continue
        ads_path = file_path + ads
        if os.path.exists(ads_path):
            try:
                os.remove(ads_path)
            except Exception:
                pass

def clean_image(file_path):
    """Strip all EXIF, XMP, IPTC, and comments from images."""
    from PIL import Image, ImageOps, ImageSequence

    ext = os.path.splitext(file_path)[1].lower()
    work_dir = os.path.dirname(os.path.abspath(file_path))
    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=work_dir)
    os.close(fd)

    try:
        with Image.open(file_path) as img:
            icc = img.info.get('icc_profile')
            is_anim = getattr(img, 'is_animated', False) and getattr(img, 'n_frames', 1) > 1

            if is_anim:
                frames = []
                durations = []
                for frame in ImageSequence.Iterator(img):
                    f = frame.copy()
                    f.info.pop('comment', None)
                    f.info.pop('exif', None)
                    f.info.pop('xmp', None)
                    frames.append(f)
                    durations.append(frame.info.get('duration', 100))

                save_kwargs = {
                    'save_all': True,
                    'append_images': frames[1:],
                    'duration': durations,
                    'loop': img.info.get('loop', 0),
                    'format': img.format
                }
                if icc:
                    save_kwargs['icc_profile'] = icc
                frames[0].save(temp_path, **save_kwargs)
            else:
                # Apply orientation so pixels match visual rotation before EXIF is dropped
                img_t = ImageOps.exif_transpose(img)
                # Create clean copy with no metadata dictionaries
                clean_img = Image.new(img_t.mode, img_t.size)
                clean_img.paste(img_t)

                fmt = img.format or ('JPEG' if ext in ('.jpg', '.jpeg') else ext.lstrip('.').upper())
                save_kwargs = {'format': fmt}
                if ext in ('.jpg', '.jpeg'):
                    save_kwargs['quality'] = 95
                elif ext == '.webp':
                    save_kwargs['quality'] = 95

                if icc:
                    save_kwargs['icc_profile'] = icc

                clean_img.save(temp_path, **save_kwargs)

        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    remove_alternate_data_streams(file_path)

def clean_video(file_path):
    """Strip metadata tags from video container and streams using FFmpeg (lossless stream copy)."""
    ffmpeg_bin = find_ffmpeg()
    ext = os.path.splitext(file_path)[1].lower()
    work_dir = os.path.dirname(os.path.abspath(file_path))
    fd, temp_path = tempfile.mkstemp(suffix=ext, dir=work_dir)
    os.close(fd)

    # CREATE_NO_WINDOW flag on Windows
    create_no_window = 0x08000000

    try:
        # Data tracks (including Resolve/QuickTime tmcd) are metadata, and may
        # have no codec FFmpeg can mux back into MP4. Keep media streams intact.
        cmd = [
            ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
            '-i', file_path,
            '-map', '0',
            '-map', '-0:d?',
            '-map_metadata', '-1',
            '-map_metadata:s', '-1',
            '-map_chapters', '-1',
            '-c', 'copy',
            '-fflags', '+bitexact',
            '-flags:v', '+bitexact',
            '-flags:a', '+bitexact',
        ]
        if ext in {'.mp4', '.mov', '.m4v', '.3gp'}:
            cmd += ['-write_tmcd', '0']
        cmd.append(temp_path)
        res = subprocess.run(cmd, creationflags=create_no_window, capture_output=True,
                             encoding='utf-8', errors='replace')
        if res.returncode != 0 or not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            details = res.stderr.strip() or 'FFmpeg produced no output.'
            raise RuntimeError(f"FFmpeg failed (code {res.returncode}):\n{details[-4000:]}")

        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    remove_alternate_data_streams(file_path)

def clean_zip(file_path):
    """Strip archive comments, entry comments, and extended timestamps from ZIP files."""
    work_dir = os.path.dirname(os.path.abspath(file_path))
    fd, temp_path = tempfile.mkstemp(suffix='.zip', dir=work_dir)
    os.close(fd)

    try:
        with zipfile.ZipFile(file_path, 'r') as zin:
            compression = zin.compression if zin.filelist else zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(temp_path, 'w', compression=compression) as zout:
                zout.comment = b''
                for item in zin.infolist():
                    new_info = zipfile.ZipInfo(filename=item.filename)
                    # Normalize timestamp to year 2000
                    new_info.date_time = (2000, 1, 1, 0, 0, 0)
                    new_info.comment = b''
                    new_info.extra = b''  # Strips NTFS timestamps, Unix UID/GID
                    new_info.compress_type = item.compress_type
                    new_info.external_attr = item.external_attr  # Preserves permissions and dir flag

                    if item.is_dir():
                        zout.writestr(new_info, b'')
                    else:
                        zout.writestr(new_info, zin.read(item.filename))

        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    remove_alternate_data_streams(file_path)

def clean_file(file_path):
    """Dispatch file to corresponding metadata stripper."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        clean_image(file_path)
    elif ext in VIDEO_EXTENSIONS:
        clean_video(file_path)
    elif ext in ZIP_EXTENSIONS:
        clean_zip(file_path)
    else:
        # Check if zip file by signature
        if zipfile.is_zipfile(file_path):
            clean_zip(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

def show_notification(title, message, is_error=False):
    """Display modern dark floating notification at bottom-right of primary screen."""
    try:
        import tkinter as tk

        if is_error:
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            try:
                messagebox.showerror(title, message, parent=root)
            finally:
                root.destroy()
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        root.configure(bg='#18181b')

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w, h = 340, 76
        x = screen_w - w - 24
        y = screen_h - h - 56
        root.geometry(f'{w}x{h}+{x}+{y}')

        border_color = '#ef4444' if is_error else '#3b82f6'
        accent_color = '#ef4444' if is_error else '#22c55e'

        frame = tk.Frame(root, bg='#18181b', highlightbackground=border_color, highlightthickness=1)
        frame.pack(fill='both', expand=True)

        lbl_title = tk.Label(frame, text=title, font=('Segoe UI', 10, 'bold'), fg=accent_color, bg='#18181b')
        lbl_title.pack(anchor='w', padx=14, pady=(8, 2))

        # Truncate message if too long
        display_msg = message if len(message) <= 45 else (message[:42] + '...')
        lbl_msg = tk.Label(frame, text=display_msg, font=('Segoe UI', 9), fg='#e4e4e7', bg='#18181b')
        lbl_msg.pack(anchor='w', padx=14, pady=(0, 8))

        # Close on click
        root.bind('<Button-1>', lambda e: root.destroy())
        frame.bind('<Button-1>', lambda e: root.destroy())
        lbl_title.bind('<Button-1>', lambda e: root.destroy())
        lbl_msg.bind('<Button-1>', lambda e: root.destroy())

        # Auto-destroy duration (longer for errors)
        timeout_ms = 4000 if is_error else 2500
        root.after(timeout_ms, root.destroy)
        root.mainloop()
    except Exception:
        pass

def main():
    if len(sys.argv) < 2:
        show_notification("Clear Meta Data", "No files specified.", is_error=True)
        return

    files = sys.argv[1:]
    successes = []
    errors = []

    for f in files:
        base_name = os.path.basename(f)
        try:
            clean_file(f)
            successes.append(base_name)
        except Exception as e:
            errors.append((base_name, str(e)))

    if errors and not successes:
        first_err = errors[0]
        show_notification("⚠ Error Clearing Metadata", f"{first_err[0]}: {first_err[1]}", is_error=True)
    elif errors and successes:
        show_notification("Clear Meta Data", f"{len(successes)} cleaned, {len(errors)} failed", is_error=True)
    elif len(successes) == 1:
        show_notification("✓ Clear Meta Data", f"Cleaned: {successes[0]}")
    elif len(successes) > 1:
        show_notification("✓ Clear Meta Data", f"Cleaned {len(successes)} files successfully")

if __name__ == '__main__':
    main()
