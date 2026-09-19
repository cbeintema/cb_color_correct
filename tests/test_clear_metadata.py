import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json

from tools.clear_metadata import clean_video


class ClearVideoMetadataTests(unittest.TestCase):
    def test_failure_preserves_original_and_reports_ffmpeg_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'original.mp4'
            path.write_bytes(b'original video')
            failure = subprocess.CompletedProcess([], 1, '', 'Specific FFmpeg error')
            with patch('tools.clear_metadata.subprocess.run', return_value=failure):
                with self.assertRaisesRegex(RuntimeError, 'Specific FFmpeg error'):
                    clean_video(str(path))
            self.assertEqual(path.read_bytes(), b'original video')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_mp4_timecode_track_is_removed_without_reencoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'timecode & spaces.mp4'
            subprocess.run([
                shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i',
                'color=size=16x16:rate=24', '-t', '1', '-c:v', 'libx264',
                '-timecode', '01:00:00:00', '-metadata', 'title=Private title', str(path),
            ], check=True, capture_output=True)
            def probe():
                return json.loads(subprocess.check_output([
                    shutil.which('ffprobe'), '-v', 'error', '-show_streams',
                    '-show_format', '-of', 'json', str(path),
                ]))
            before = probe()
            self.assertTrue(any(s['codec_type'] == 'data' for s in before['streams']))
            clean_video(str(path))
            after = probe()
            self.assertEqual([s['codec_type'] for s in after['streams']], ['video'])
            self.assertEqual(before['streams'][0]['nb_frames'], after['streams'][0]['nb_frames'])
            self.assertEqual(before['streams'][0]['codec_name'], after['streams'][0]['codec_name'])
            self.assertNotIn('title', after['format'].get('tags', {}))
            self.assertNotIn('timecode', after['streams'][0].get('tags', {}))
