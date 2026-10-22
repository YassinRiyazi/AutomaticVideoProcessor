import io
import os
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Roots(unittest.TestCase):
    """Temporary input / work / output roots wired into main."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        self.input, self.tmp, self.output = base / "Samba-Share" / "Clips", base / "tmp_processing", base / "Samba-Share" / "Output"
        self.input.mkdir(parents=True)
        for name, value in (("INPUT_ROOT", self.input), ("OUTPUT_ROOT", self.output), ("PROCESSING_ROOT", self.tmp)):
            patcher = patch.object(main, name, str(value))
            patcher.start()
            self.addCleanup(patcher.stop)

    def video_folder(self, relative: str) -> Path:
        folder = self.input / relative
        write(folder / f"{folder.name}_4000fps.mp4", "video")
        write(folder / f"{folder.name}_4000fps_camera.json", "{}")
        return folder


class PathAndMirrorTests(Roots):
    def test_paths_mirror_the_input_structure(self):
        folder = self.video_folder("2026-10-20_15-20-34/tilt-25-rep-001")
        self.assertEqual(main.work_folder(str(folder)), str(self.tmp / "2026-10-20_15-20-34/tilt-25-rep-001"))
        self.assertEqual(main.output_folder(str(folder)), str(self.output / "2026-10-20_15-20-34/tilt-25-rep-001"))
        with self.assertRaises(ValueError):
            main.output_folder(str(self.tmp))

    def test_mirror_copies_only_csv_logs_and_markers(self):
        work, out = self.tmp / "a", self.output / "a"
        for name in ("result.csv", ".done", "post_run_error.txt", "side/result.csv", "side/.done",
                     "side/databases/detections.csv", "side/databases/error_log.txt", "front/log_1.log"):
            write(work / name)
        for name in ("side/SR_edge/frame_000001.png", "side/frames_rotated/frame_000001.png", "a.mp4"):
            write(work / name)
        main.mirror_results(str(work), str(out))
        mirrored = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        self.assertEqual(mirrored, sorted([".done", "front/log_1.log", "post_run_error.txt", "result.csv", "side/.done",
                                           "side/databases/detections.csv", "side/databases/error_log.txt", "side/result.csv"]))

        # Unchanged files are not copied again; changed ones are replaced
        before = os.stat(out / "result.csv").st_ino
        main.mirror_results(str(work), str(out))
        self.assertEqual(os.stat(out / "result.csv").st_ino, before)
        write(work / "result.csv", "changed")
        main.mirror_results(str(work), str(out))
        self.assertEqual((out / "result.csv").read_text(), "changed")

    def test_videos_are_hard_linked_into_the_work_folder(self):
        folder = self.video_folder("x/trial")
        work = Path(main.work_folder(str(folder)))
        write(work / "trial_old.mp4", "stale")
        links = main.link_videos(str(folder), str(work))
        self.assertEqual([os.path.basename(p) for p in links], ["trial_4000fps.mp4"])
        self.assertFalse((work / "trial_old.mp4").exists())
        self.assertEqual(os.stat(links[0]).st_ino, os.stat(folder / "trial_4000fps.mp4").st_ino)


class ProcessingSpaceTests(Roots):
    def make(self, rel: str, files: int, size: int, age: float) -> Path:
        folder = self.tmp / rel
        for i in range(files):
            write(folder / "side" / "SR_edge" / f"frame_{i:06d}.png", "x" * size)
        os.utime(folder, (age, age))
        return folder

    def test_oldest_folders_are_deleted_first_and_active_is_kept(self):
        old = self.make("day1/a", 4, 10, 1000)
        mid = self.make("day1/b", 4, 10, 2000)
        new = self.make("day2/c", 4, 10, 3000)
        space = main.ProcessingSpace(str(self.tmp), max_files=8, max_bytes=10**9)
        self.assertEqual(space.totals(), (12, 120))
        with contextlib.redirect_stdout(io.StringIO()):
            space.enforce(keep=str(old))
        self.assertTrue(old.exists())
        self.assertFalse(mid.exists())
        self.assertTrue(new.exists())
        self.assertEqual(space.totals(), (8, 80))

        with contextlib.redirect_stdout(io.StringIO()):
            space.max_bytes = 50
            space.enforce(keep=str(new))
        self.assertFalse(old.exists())
        self.assertFalse((self.tmp / "day1").exists())
        self.assertTrue(new.exists())

    def test_usage_index_survives_restart_and_is_reconciled(self):
        a = self.make("a", 2, 5, 1000)
        main.ProcessingSpace(str(self.tmp), 100, 10**9)
        self.assertTrue((self.tmp / main.ProcessingSpace.INDEX).is_file())
        b = self.make("b", 3, 5, 2000)
        os.system(f"rm -r '{a}'")
        space = main.ProcessingSpace(str(self.tmp), 100, 10**9)
        self.assertEqual(set(space.usage), {"b"})
        self.assertEqual(space.totals(), (3, 15))
        write(b / "front" / "x.png", "y" * 5)
        space.update(str(b))
        self.assertEqual(space.totals(), (4, 20))


class ProcessFolderTests(Roots):
    FRAMES = range(2301, 2306)

    def fake_side(self, work, _pipeline):
        address = Path(work) / main.SIDE
        self.assertTrue(any(p.suffix == ".mp4" for p in Path(work).iterdir()), "video must be linked into the work folder")
        rows = ["file number,time (s),adv (degree)"]
        for i, n in enumerate(self.FRAMES):
            write(address / main.ROTATED_FRAMES_FOLDER / f"frame_{n:06d}.png")
            write(address / "SR_edge" / f"frame_{n:06d}.png")
            if n != self.FRAMES[-1]:
                rows.append(f"frame_{n:06d}.png,{i / main.FPS_EXPERIMENT},100")
        write(address / "result.csv", "\n".join(rows) + "\n")
        write(address / main.DATABASES_FOLDER / "detections.csv", "image\n")

    def fake_front(self, work, _pipeline):
        address = Path(work) / main.FRONT
        rows = ["file number,height (cm)"]
        for n in self.FRAMES:
            write(address / main.ROTATED_FRAMES_FOLDER / f"frame_{n:06d}.png")
            write(address / "SR_edge" / f"frame_{n:06d}.png")
            rows.append(f"frame_{n:06d}.png,0.2")
        write(address / "result.csv", "\n".join(rows) + "\n")

    def run_folder(self, folder: Path, front=None):
        pipeline = SimpleNamespace(space=main.ProcessingSpace(str(self.tmp), 10**6, 10**12))
        with patch.object(main, "process_side", side_effect=self.fake_side) as side, \
                patch.object(main, "process_front", side_effect=front or self.fake_front) as front_mock, \
                contextlib.redirect_stdout(io.StringIO()):
            main.process_folder(str(folder), pipeline)
        return side.call_count, front_mock.call_count, pipeline.space

    def test_outputs_go_to_tmp_and_only_csv_logs_to_the_mirror(self):
        folder = self.video_folder("2026-10-20/tilt-25-rep-001")
        source_before = sorted(p.name for p in folder.iterdir())
        self.assertTrue(main.needs_processing(str(folder)))

        self.assertEqual(self.run_folder(folder)[:2], (1, 1))

        out, work = Path(main.output_folder(str(folder))), Path(main.work_folder(str(folder)))
        self.assertEqual(sorted(p.name for p in folder.iterdir()), source_before)
        mirrored = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        self.assertEqual(mirrored, sorted([".done", "result.csv", "front/.done", "front/result.csv",
                                           "side/.done", "side/result.csv", "side/databases/detections.csv"]))
        self.assertEqual(len((out / "result.csv").read_text().splitlines()), 1 + len(self.FRAMES) - 1)
        self.assertEqual(len(list((work / "side" / "SR_edge").glob("*.png"))), len(self.FRAMES))
        self.assertFalse(list(work.glob("*.mp4")), "the video link is removed after processing")
        self.assertFalse(main.needs_processing(str(folder)))
        self.assertEqual(self.run_folder(folder)[:2], (0, 0))

    def test_failed_view_log_is_mirrored_and_not_retried(self):
        folder = self.video_folder("day/trial")
        with contextlib.redirect_stderr(io.StringIO()):
            self.run_folder(folder, front=lambda work, p: (_ for _ in ()).throw(RuntimeError("boom")))
        out = Path(main.output_folder(str(folder)))
        self.assertTrue(list((out / "front").glob("*.log")))
        self.assertFalse((out / "result.csv").exists())
        self.assertFalse(main.needs_processing(str(folder)))

    def test_views_are_redone_when_tmp_was_trimmed_mid_folder(self):
        folder = self.video_folder("day/trial")
        self.run_folder(folder)
        out, work = Path(main.output_folder(str(folder))), Path(main.work_folder(str(folder)))
        # Simulate: interrupted after the side view, then the work folder was evicted
        for name in ("front", "result.csv", ".done"):
            os.system(f"rm -r '{out / name}'")
        os.system(f"rm -r '{work}'")
        self.assertTrue(main.needs_processing(str(folder)))
        self.assertEqual(self.run_folder(folder)[:2], (1, 1))
        self.assertTrue((out / "result.csv").is_file())

    def test_clean_start_resets_work_and_mirror(self):
        folder = self.video_folder("day/trial")
        self.run_folder(folder)
        main.cleanStart([str(folder)], views=(main.FRONT,))
        out = Path(main.output_folder(str(folder)))
        self.assertFalse((out / "front").exists())
        self.assertFalse((out / "result.csv").exists())
        self.assertTrue((out / "side" / ".done").exists())
        self.assertEqual(self.run_folder(folder)[:2], (0, 1))


if __name__ == "__main__":
    unittest.main()
