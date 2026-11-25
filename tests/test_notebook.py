"""Execute every notebook cell in a fresh kernel against synthetic CPU data.

Only mounts, accelerator, and experiment scale are substituted; the notebook's
source discovery, copy/import, training, evaluation, and plotting flow runs as written.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from test_baseline import synthetic_dataset

try:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    HAS_NOTEBOOK_DEPS = True
except ImportError:
    HAS_NOTEBOOK_DEPS = False


@unittest.skipUnless(HAS_NOTEBOOK_DEPS, "Install requirements-dev.txt for fresh-kernel notebook checks")
class NotebookTests(unittest.TestCase):
    def test_run_all_on_synthetic_cpu_data(self):
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            root, split_file = synthetic_dataset(temporary)
            input_dir, working = temporary / "input", temporary / "working"
            code_dataset = input_dir / "project_code"
            for name in ("src", "configs"):
                shutil.copytree(name, code_dataset / name, ignore=shutil.ignore_patterns("__pycache__"))
            nb = nbformat.read("notebooks/kaggle_baseline.ipynb", as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                source = source.replace("/kaggle/input/datasets/abelnathanaelh/agriculture-vision-2017/data2017_miniscale", root.as_posix())
                source = source.replace("/kaggle/input/datasets/abelnathanaelh/data2017-splits/data2017_splits.json", split_file.as_posix())
                source = source.replace("/kaggle/input", input_dir.as_posix())
                source = source.replace("/kaggle/working", working.as_posix())
                source = source.replace('config["device"] = "cuda"', 'config["device"] = "cpu"')
                source = source.replace('validate_config(config)',
                    'config.update(image_size=32, native_size=32, base_channels=2, batch_size=2, '
                    'num_workers=0, epochs=2, expected_counts={"train": 2, "val": 1, "test": 1})\n'
                    'torch.set_num_threads(1)\nvalidate_config(config)')
                source = source.replace('if not torch.cuda.is_available():', 'if False:')
                source = source.replace('torch.cuda.get_device_name(0)', '"CPU notebook smoke test"')
                source = source.replace('.cuda()', '.to("cpu")').replace('device="cuda"', 'device="cpu"')
                source = source.replace('cuda=True', 'cuda=False')
                cell.source = source
            final_results = next(cell for cell in nb.cells
                                 if cell.cell_type == "code" and "metrics_path =" in cell.source)
            nb.cells.append(nbformat.v4.new_code_cell(
                'def evaluate(*args, **kwargs):\n'
                '    raise AssertionError("Cached final results must not rerun test inference")\n'
                + final_results.source))
            kernel_dir = temporary / "kernels" / "baseline-smoke"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Baseline CPU smoke", "language": "python"}))
            manager = KernelManager(kernel_name="baseline-smoke",
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]),
                                    connection_file=str(temporary / "connection.json"))
            env = os.environ.copy()
            env.update(IPYTHONDIR=str(temporary / "ipython"),
                       JUPYTER_RUNTIME_DIR=str(temporary / "runtime"),
                       MPLCONFIGDIR=str(temporary / "matplotlib"))
            try:
                NotebookClient(nb, km=manager, timeout=120,
                               resources={"metadata": {"path": str(Path.cwd())}}).execute(env=env)
            finally:
                # NotebookClient leaves caller-owned kernel managers running.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            metrics = json.loads((working / "agriculture_segmentation/test/metrics.json").read_text())
            self.assertEqual(metrics["split"], "test")
            self.assertGreater(metrics["valid_pixels"], 0)
            code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
            self.assertTrue(all(cell.execution_count is not None for cell in code_cells))
            self.assertTrue(any("Reusing existing final metrics" in output.get("text", "")
                                for output in nb.cells[-1].outputs))


if __name__ == "__main__":
    unittest.main()
