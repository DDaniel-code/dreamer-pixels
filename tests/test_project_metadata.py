from pathlib import Path


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
