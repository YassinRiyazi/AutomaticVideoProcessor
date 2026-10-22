import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main


class TiltTests(unittest.TestCase):
    def test_tilt_from_file_name(self):
        self.assertEqual(main.tilt_from_name("tilt-20-rep-001_4000fps_camera.json"), 20.0)
        self.assertEqual(main.tilt_from_name("/a/b/TILT-27.5-rep-1"), 27.5)
        self.assertIsNone(main.tilt_from_name("Fast025_50-73751_500_26-10-21--11-26-18"))
        self.assertEqual(main.tilt_from_name("Fast025", "tilt-35-rep-002"), 35.0)


class ParameterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / "tilt-20-rep-001"
        self.folder.mkdir()

    def table(self):
        parameters, issues = main.experiment_parameters(str(self.folder))
        return parameters.set_index("parameter"), issues

    def test_config_fallback_and_tilt_from_json_name(self):
        (self.folder / "tilt-20-rep-001_4000fps_camera.json").write_text(json.dumps({"fps": main.FPS_EXPERIMENT}))
        p, issues = self.table()
        config = main.BaseUtils.config["Experiment_Parameters"]
        self.assertEqual(issues, [])
        for key in ("mass", "gravity", "viscosity", "mStarmRatio"):
            self.assertEqual(p.loc[key, "value"], float(config[key]))
            self.assertEqual(p.loc[key, "source"], "config.yaml")
        self.assertEqual(p.loc["tilt", "value"], 20.0)
        self.assertEqual(p.loc["tilt", "source"], "tilt-20-rep-001_4000fps_camera.json")
        self.assertEqual(p.loc["mass", "unit"], "kg")

    def test_json_values_take_precedence(self):
        (self.folder / "tilt-20-rep-001_4000fps_camera.json").write_text(json.dumps({"mass": 2e-5, "viscosity": 5e-3, "fps": 2000}))
        p, issues = self.table()
        self.assertEqual(p.loc["mass", "value"], 2e-5)
        self.assertEqual(p.loc["mass", "source"], "tilt-20-rep-001_4000fps_camera.json")
        self.assertEqual(p.loc["gravity", "source"], "config.yaml")
        self.assertTrue(any("fps = 2000" in issue for issue in issues))

    def test_missing_tilt_is_reported(self):
        folder = self.folder.parent / "Fast025_50-73751_500"
        folder.mkdir()
        parameters, issues = main.experiment_parameters(str(folder))
        self.assertTrue(np.isnan(parameters.set_index("parameter").loc["tilt", "value"]))
        self.assertTrue(any("tilt" in issue for issue in issues))


class AccelerationTests(unittest.TestCase):
    def test_constant_acceleration_is_recovered_despite_noise_free_gaps(self):
        fps, a = 4000.0, 0.9
        frames = [n for n in range(2301, 2401) if n not in (2330, 2331, 2370)]
        t = (np.array(frames) - 2301) / fps
        x_cm = pd.Series(100 * (0.015 + 0.3 * t + 0.5 * a * t ** 2))
        names = pd.Series([f"frame_{n:06d}.png" for n in frames])
        result = main.smoothed_acceleration(names, x_cm, window=21, fps=fps)
        np.testing.assert_allclose(result, a, rtol=1e-6)

    def test_short_series_gives_nan(self):
        names = pd.Series(["frame_000001.png", "frame_000002.png", "frame_000003.png"])
        self.assertTrue(np.isnan(main.smoothed_acceleration(names, pd.Series([1.0, 1.1, 1.2]))).all())

    def test_noise_is_reduced_on_window(self):
        rng = np.random.default_rng(0)
        fps, n = 4000.0, 400
        t = np.arange(n) / fps
        x_cm = pd.Series(100 * (0.3 * t + 0.5 * 0.9 * t ** 2) + rng.normal(0, 1e-4, n))
        names = pd.Series([f"frame_{i:06d}.png" for i in range(n)])
        raw = np.diff(np.diff(x_cm.to_numpy() / 100)) * fps ** 2
        smooth = main.smoothed_acceleration(names, x_cm, window=21, fps=fps)
        self.assertLess(smooth.std(), raw.std() / 50)


class FrictionInPostRunTests(unittest.TestCase):
    def test_friction_formula_units_and_sign(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name) / "tilt-30-rep-001"
        fps, a, frames = main.FPS_EXPERIMENT, 0.9, list(range(1000, 1101))
        t = (np.array(frames) - frames[0]) / fps
        x_cm = 100 * (0.015 + 0.3 * t + 0.5 * a * t ** 2)
        for view in main.VIEWS:
            (folder / view / main.ROTATED_FRAMES_FOLDER).mkdir(parents=True)
            (folder / view / main.DONE).write_text("")
            for n in frames:
                (folder / view / main.ROTATED_FRAMES_FOLDER / f"frame_{n:06d}.png").write_text("")
        names = [f"frame_{n:06d}.png" for n in frames]
        pd.DataFrame({"file number": names[:-1], "time (s)": t[:-1], "x_center (cm)": x_cm[:-1]}).to_csv(folder / "side" / "result.csv", index=False)
        pd.DataFrame({"file number": names, "height (cm)": 0.2}).to_csv(folder / "front" / "result.csv", index=False)

        with patch("builtins.print"):
            merged = main.post_run(str(folder))
        config = main.BaseUtils.config["Experiment_Parameters"]
        m, g, ratio = float(config["mass"]), float(config["gravity"]), float(config["mStarmRatio"])
        expected_uN = (m * g * np.sin(np.radians(30)) - m * ratio * a) * 1e6
        np.testing.assert_allclose(merged["a (m/s^2)"], a, rtol=1e-6)
        np.testing.assert_allclose(merged["friction force (uN)"], expected_uN, rtol=1e-6)
        self.assertFalse((folder / main.POST_RUN_ERROR).exists())
        parameters = pd.read_csv(folder / main.PARAMETERS_CSV).set_index("parameter")
        self.assertEqual(parameters.loc["tilt", "value"], 30)


if __name__ == "__main__":
    unittest.main()
