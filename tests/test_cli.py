"""CLI smoke tests (model-free paths)."""
import importlib.util

import pytest

from graphcodebert_interpretability.cli import build_parser, main

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_parser_builds_all_subcommands():
    parser = build_parser()
    # Parsing a valid subcommand should not raise.
    args = parser.parse_args(["report", "--no-neural"])
    assert args.command == "report"


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "gcbi" in capsys.readouterr().out


def test_report_no_neural_runs(tmp_path):
    code = main(["report", "--no-neural", "--outdir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "report.md").exists()


@pytest.mark.skipif(TORCH_AVAILABLE, reason="torch installed; model path would run")
def test_compare_without_model_fails_cleanly(tmp_path, capsys):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x): return x\n", encoding="utf-8")
    b.write_text("def g(y): return y\n", encoding="utf-8")
    code = main(["compare", str(a), str(b)])
    assert code == 2
    assert "error:" in capsys.readouterr().err
