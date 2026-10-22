import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import FrameExtractor
import main


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, relative, name=None):
        folder = self.root / relative
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / (name or f"{folder.name}_4000fps.mp4")
        video.write_bytes(b"video")
        return folder

    def test_recursive_discovery_stops_at_video_folder(self):
        direct = self.video("direct", "direct.mp4")
        nested = self.video("2026-10-20/session/tilt-25-rep-002")
        self.video("2026-10-20/session/tilt-25-rep-002/side", "side.mp4")
        self.video("wrong", "unrelated.mp4")
        self.video("wrong_prefix", "wrong_prefix_extra.mp4")
        (self.root / "empty").mkdir()
        self.assertEqual(main.discover_folders(str(self.root)), sorted(map(str, [
            direct, nested, self.root / "wrong_prefix",
        ])))

    def test_wildcard_roots(self):
        first = self.video("2026-10-20/first")
        second = self.video("2026-10-21/deeper/second")
        self.video("archive/old")
        self.assertEqual(
            main.discover_folders(str(self.root / "2026-*")),
            sorted(map(str, [first, second])),
        )

    def test_parent_name_boundary_and_extensions(self):
        for name in ("trial.mp4", "trial_4000fps.MP4", "trial_0000.avi"):
            with self.subTest(name=name):
                self.assertTrue(FrameExtractor.video_matches_folder("/root/trial/", name))
        for name in ("trial2.mp4", "other.mp4", ".trial.mp4", "trial.mp4.part", "trial.csv"):
            with self.subTest(name=name):
                self.assertFalse(FrameExtractor.video_matches_folder("/root/trial", name))

    def test_discovery_signature_and_extractor_agree(self):
        folder = self.video("trial")
        self.video("trial", "unrelated.mp4")
        self.video("trial", ".trial.mp4")
        self.assertEqual(main.discover_folders(str(self.root)), [str(folder)])
        signature = main.video_signature(str(folder))
        self.assertIsNotNone(signature)
        self.assertEqual([row[0] for row in signature], ["trial_4000fps.mp4"])
        extractor = FrameExtractor.FrameExtractor.__new__(FrameExtractor.FrameExtractor)
        self.assertEqual(
            extractor.resolve_video_path(str(folder)),
            str(folder / "trial_4000fps.mp4"),
        )
        self.video("trial", "trial_0000.mov")
        with self.assertRaises(FileNotFoundError):
            extractor.resolve_video_path(str(folder))

    def test_new_nested_upload_is_found_by_next_queue_scan(self):
        pattern = str(self.root)
        original_discovery = main.discover_folders
        with patch.object(main, "discover_folders", side_effect=lambda: original_discovery(pattern)), \
                patch.object(main, "INPUT_ROOT", str(self.root)), \
                patch.object(main, "OUTPUT_ROOT", str(self.root.parent / (self.root.name + "_out"))):
            uploads = main.UploadQueue(0)
            uploads.scan()
            self.assertEqual(uploads.qsize(), 0)
            folder = self.video("new/date/trial")
            uploads.scan()
            self.assertEqual(uploads.get(timeout=0), str(folder))
            uploads.scan()
            self.assertEqual(uploads.qsize(), 0)

    def test_symbolic_link_video_is_not_discovered(self):
        folder = self.root / "linked"
        folder.mkdir()
        target = self.root / "target.mp4"
        target.write_bytes(b"video")
        os.symlink(target, folder / "linked.mp4")
        self.assertEqual(main.discover_folders(str(self.root)), [])


if __name__ == "__main__":
    unittest.main()
