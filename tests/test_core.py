import inspect
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qti_unified import qti_math
from qti_unified.config import PathoRunConfig, resolve_data_root
from qti_unified.patho import available_scenarios, generate_patho_case
from qti_unified.pipeline import discover_prediction_cases, generate_patho_data, validate_gt
from qti_unified.signals import patho_output_paths


class CoreWorkflowTests(unittest.TestCase):
    def test_gt_metric_payload_has_required_keys(self):
        case = generate_patho_case("needle_sphere_2", "needle_sphere_2tensors", n_tensors=8, seed=7)
        gt = qti_math.gt_scalars_from_params(case.params)

        for key in ["MD", "MD_um2_per_ms", "FA", "uFA", "C_MD", "C_c"]:
            self.assertIn(key, gt)
            self.assertTrue(gt[key] is None or np.isfinite(gt[key]))

    def test_data_root_prefers_argument_then_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "explicit"
            self.assertEqual(resolve_data_root(explicit), explicit.resolve())

            old_value = os.environ.get("QTI_DATA_ROOT")
            try:
                env_root = Path(tmp) / "env-root"
                os.environ["QTI_DATA_ROOT"] = str(env_root)
                self.assertEqual(resolve_data_root(None), env_root.resolve())
            finally:
                if old_value is None:
                    os.environ.pop("QTI_DATA_ROOT", None)
                else:
                    os.environ["QTI_DATA_ROOT"] = old_value

    def test_patho_output_naming_is_patho_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = patho_output_paths(tmp, "spheres_2", "spheres_csf_1500", snr=30)
            joined = "\n".join(str(path) for path in paths.values())
            self.assertIn("DTDs_cov_suite_2_patho", joined)
            self.assertIn("Results_2_MLP_patho", joined)
            self.assertIn("Results_SNR_fit_2_MLP_patho", joined)
            self.assertNotIn("sweep", joined.lower())

    def test_tiny_generation_writes_gt_and_noisy_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = PathoRunConfig(data_root=Path(tmp), n_tensors=8, n_realizations=2, snr=30.0, seed=3)
            generated = generate_patho_data(config, scenarios=["needle_sphere_2"])

            self.assertEqual(len(generated), 1)
            item = generated[0]
            gt_path = Path(item.paths["gt_json"])
            self.assertTrue(gt_path.exists())
            payload = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertIn("gt", payload)
            self.assertIn("MD", payload["gt"])
            self.assertEqual(item.noisy_signals.shape[0], 2)
            self.assertEqual(item.exact_signal.shape, item.cumexp_signal.shape)
            self.assertFalse(np.allclose(item.noisy_signals[0], item.noisy_signals[1]))

            discovered = discover_prediction_cases(tmp, snr_folder="SNR30")
            self.assertEqual(len(discovered), 2)

    def test_single_case_scenarios_available(self):
        scenarios = set(available_scenarios())
        self.assertIn("needle_sphere_2", scenarios)
        self.assertIn("spheres_2", scenarios)
        self.assertIn("crossing_needles_3_xyz", scenarios)

    def test_validate_gt_matches_stored_metrics_when_pandas_is_available(self):
        try:
            importlib.import_module("pandas")
        except ModuleNotFoundError:
            self.skipTest("pandas is not installed in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            config = PathoRunConfig(data_root=Path(tmp), n_tensors=8, n_realizations=0, snr=None, seed=3)
            generate_patho_data(config, scenarios=["needle_sphere_2"])

            validation = validate_gt(tmp)
            self.assertFalse(validation.empty)
            self.assertTrue(validation["ok"].all())


class RepositoryPolicyTests(unittest.TestCase):
    def test_gitignore_keeps_large_external_data_out(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / ".gitignore").read_text(encoding="utf-8")
        for pattern in ["data/", "runs/", "patient_data/", "*.nii", "*.nii.gz", "*.pt", "*.pth", "*_zscore.csv"]:
            self.assertIn(pattern, text)

    def test_public_functions_and_classes_have_docstrings(self):
        module_names = [
            "qti_unified.cli",
            "qti_unified.config",
            "qti_unified.mlp",
            "qti_unified.patient",
            "qti_unified.patho",
            "qti_unified.pipeline",
            "qti_unified.plotting",
            "qti_unified.signals",
        ]
        modules = [qti_math]
        skipped = []
        for module_name in module_names:
            try:
                modules.append(importlib.import_module(module_name))
            except ModuleNotFoundError as exc:
                skipped.append(f"{module_name}: missing {exc.name}")
        if skipped:
            print("Skipped optional docstring imports:", "; ".join(skipped))

        missing = []
        for module in modules:
            self.assertIsNotNone(module.__doc__, module.__name__)
            for name, obj in vars(module).items():
                if name.startswith("_"):
                    continue
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    if getattr(obj, "__module__", None) != module.__name__:
                        continue
                    if not inspect.getdoc(obj):
                        missing.append(f"{module.__name__}.{name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
