"""Pruebas del punto de entrada reproducible."""

from pathlib import Path

import pytest
from test_pipeline import synthetic_csv

from conformal_fault_inference_with_abstention.__main__ import (
    build_parser,
    config_from_args,
    main,
)
from conformal_fault_inference_with_abstention.contracts import LabelPolicy


def test_defaults_match_reported_run():
    config = config_from_args(build_parser().parse_args([]))
    assert config.data_path == Path("data/raw/ai4i2020.csv")
    assert config.output_dir.parent == Path("artifacts")
    assert config.output_dir.name.startswith("run_") and config.output_dir.name.endswith(
        "_seed42_hgb-mlp"
    )
    assert config.overwrite is False
    assert config.label_policy == LabelPolicy(("TWF", "PWF", "OSF", "HDF"), "exclude_only_rnf")
    assert (config.calibration_size, config.test_size) == (0.30, 0.20)
    assert config.alphas == (0.05, 0.10, 0.20)
    assert config.random_state == 42
    assert config.models == ("hgb", "mlp")
    assert config.mlp_epochs == 200
    assert config.mlp_benchmark_repetitions == 3


def test_arguments_override_defaults():
    args = build_parser().parse_args(
        "--models hgb --alphas 0.1 --seed 7 --priority HDF OSF PWF TWF".split()
    )
    config = config_from_args(args)
    assert config.models == ("hgb",)
    assert config.alphas == (0.1,)
    assert config.random_state == 7
    assert config.label_policy.priority == ("HDF", "OSF", "PWF", "TWF")


def test_invalid_priority_exits():
    args = build_parser().parse_args(["--priority", "TWF", "TWF", "PWF", "OSF"])
    with pytest.raises(SystemExit):
        config_from_args(args)


def test_main_runs_end_to_end_with_hgb(tmp_path, capsys):
    csv = tmp_path / "synthetic.csv"
    synthetic_csv(csv)
    code = main(
        [
            "--data",
            str(csv),
            "--output",
            str(tmp_path / "run"),
            "--models",
            "hgb",
            "--alphas",
            "0.1",
            "0.3",
            "--benchmark",
            "0",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "metrics.csv").exists()
    out = capsys.readouterr().out
    assert "mondrian" in out and "split" in out


def test_main_refuses_non_empty_output_without_overwrite(tmp_path):
    csv = tmp_path / "synthetic.csv"
    synthetic_csv(csv)
    out = tmp_path / "run"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    base = ["--data", str(csv), "--output", str(out), "--models", "hgb", "--benchmark", "0"]
    with pytest.raises(FileExistsError):
        main(base)
    assert main([*base, "--overwrite"]) == 0
