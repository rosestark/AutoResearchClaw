from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from researchclaw.agents.figure_agent import renderer as figure_renderer
from researchclaw.agents.figure_agent.renderer import RendererAgent
from researchclaw.llm.acp_client import ACPClient, ACPConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline._helpers import StageResult
from researchclaw.pipeline.experiment_diagnosis import assess_experiment_quality
from researchclaw.pipeline.stages import Stage, StageStatus


def test_pipeline_summary_does_not_report_proceed_on_pre_decision_failure(tmp_path: Path) -> None:
    results = [
        StageResult(stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)),
        StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.FAILED,
            artifacts=(),
            decision="retry",
            error="ACP prompt failed (exit 1): Queue owner disconnected before prompt completion",
        ),
    ]

    summary = rc_runner._build_pipeline_summary(
        run_id="nightly-failed-before-decision",
        results=results,
        from_stage=Stage.TOPIC_INIT,
        run_dir=tmp_path,
    )

    assert summary["final_status"] == "failed"
    assert summary["final_decision"] != "proceed"
    assert summary["final_decision"] == "failed"
    assert "Queue owner disconnected" in summary["stall_reason"]


def test_acp_queue_owner_disconnect_is_retried(monkeypatch: Any, tmp_path: Path) -> None:
    client = ACPClient(
        ACPConfig(
            agent="codex",
            cwd=str(tmp_path),
            acpx_command="/bin/echo",
            session_name="arc-test-session",
        )
    )
    attempts = {"count": 0, "reconnects": 0}

    monkeypatch.setattr(client, "_ensure_session", lambda: None)
    monkeypatch.setattr(client, "_resolve_acpx", lambda: "/bin/echo")

    def fake_send_prompt_cli(acpx: str, prompt: str) -> str:
        assert acpx == "/bin/echo"
        assert prompt == "hello"
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError(
                "ACP prompt failed (exit 1): [acpx] session arc-test · agent connected\n"
                "Queue owner disconnected before prompt completion"
            )
        return "ok"

    def fake_force_reconnect() -> None:
        attempts["reconnects"] += 1

    monkeypatch.setattr(client, "_send_prompt_cli", fake_send_prompt_cli)
    monkeypatch.setattr(client, "_force_reconnect", fake_force_reconnect)

    assert client._send_prompt("hello") == "ok"
    assert attempts == {"count": 2, "reconnects": 1}


def test_renderer_auto_mode_falls_back_when_docker_image_is_unavailable(monkeypatch: Any) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="No such image: researchclaw/experiment:latest",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(figure_renderer.subprocess, "run", fake_run)

    agent = RendererAgent(llm=None, use_docker=None, docker_image="researchclaw/experiment:latest")

    assert agent._use_docker is False
    assert "image unavailable" in agent._docker_unavailable_reason


def test_identical_ablation_outputs_are_not_full_paper_quality() -> None:
    summary = {
        "condition_summaries": {
            "baseline": {"metrics": {"primary_metric": 0.5}},
            "proposed": {"metrics": {"primary_metric": 0.5}},
            "ablation_no_gate": {"metrics": {"primary_metric": 0.5}},
        },
        "best_run": {
            "metrics": {
                "baseline/0/primary_metric": 0.5,
                "baseline/1/primary_metric": 0.5,
                "proposed/0/primary_metric": 0.5,
                "proposed/1/primary_metric": 0.5,
                "ablation_no_gate/0/primary_metric": 0.5,
                "ablation_no_gate/1/primary_metric": 0.5,
            }
        },
        "ablation_warnings": [
            "ABLATION FAILURE: Conditions 'baseline' and 'proposed' produce identical outputs across all 1 metrics."
        ],
    }

    quality = assess_experiment_quality(summary, min_conditions=3, min_seeds=2)

    assert quality.sufficient is False
    assert quality.mode.value == "technical_report"
    assert any(d.type.value == "identical_conditions" and d.severity == "critical" for d in quality.deficiencies)


def test_stage14_summary_selection_prefers_current_over_stale_version(tmp_path: Path) -> None:
    stale = tmp_path / "stage-14_v1"
    stale.mkdir()
    (stale / "experiment_summary.json").write_text(
        json.dumps({"source": "stale"}), encoding="utf-8"
    )
    current = tmp_path / "stage-14"
    current.mkdir()
    (current / "experiment_summary.json").write_text(
        json.dumps({"source": "current"}), encoding="utf-8"
    )

    selected = rc_runner._select_stage14_experiment_summary(tmp_path)

    assert selected == current / "experiment_summary.json"


def test_stage14_hard_quality_blocker_detects_ablation_integrity_failure(tmp_path: Path) -> None:
    stage14 = tmp_path / "stage-14"
    stage14.mkdir()
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(
            {
                "ablation_warnings": [
                    "ABLATION FAILURE: Conditions 'a' and 'b' produce identical outputs across all 3 metrics."
                ]
            }
        ),
        encoding="utf-8",
    )

    blocker = rc_runner._stage14_hard_quality_blocker(tmp_path)

    assert blocker is not None
    assert "ablation integrity failure" in blocker
    assert "stage-14/experiment_summary.json" in blocker
