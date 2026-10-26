from pathlib import Path
import csv
import importlib.util


ROOT = Path(__file__).parents[1]


def test_required_documents_exist():
    required = [
        "README.md",
        "docs/reproduction_protocol.md",
        "docs/implementation_differences.md",
        "docs/paper_to_code.md",
        "docs/cluster_setup.md",
        "docs/HANDOFF.md",
        "configs/repro_dreamer_v1.yaml",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_no_fake_results_are_recorded():
    text = (ROOT / "docs/reproduction_protocol.md").read_text()
    assert "Not yet verified" in text
    assert "833.66" in text


def test_non_log_boundary_is_reported(tmp_path):
    module_path = ROOT / "scripts" / "kaggle_pytorch_pilot.py"
    spec = importlib.util.spec_from_file_location("kaggle_pytorch_pilot", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    progress = tmp_path / "progress.csv"
    with progress.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Diagnostics/CumSteps", "Diagnostics/CumUpdates",
        ])
        writer.writeheader()
        writer.writerow({"Diagnostics/CumSteps": "1000", "Diagnostics/CumUpdates": "0"})
    result = module.progress_boundary_metadata(progress, 1007)
    assert result["last_logged_decision_steps"] == 1000
    assert result["last_logged_updates"] == 0
    assert result["non_log_boundary"] is True
