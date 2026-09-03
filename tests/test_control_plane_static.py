from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_codex_hooks_do_not_run_local_runtime_verification() -> None:
    """Break caught: laptop stop/post hooks invoke Python, pytest, or dependency work."""
    for relative_path in (".codex/hooks/stop.ps1", ".codex/hooks/post_tool_use.ps1"):
        script = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()

        assert "python" not in script
        assert "pytest" not in script
        assert "pip install" not in script


def test_codex_stop_hook_has_json_only_status_payloads() -> None:
    """Break caught: stop hook emits plain command output instead of Codex JSON."""
    script = (PROJECT_ROOT / ".codex/hooks/stop.ps1").read_text(encoding="utf-8")

    assert "ConvertTo-Json" in script
    assert "hookSpecificOutput" in script
    assert "permissionDecision" in script


def test_pre_tool_hook_denies_malformed_json_with_json_payload() -> None:
    """Break caught: malformed hook input fails open without a Codex JSON decision."""
    script = (PROJECT_ROOT / ".codex/hooks/pre_tool_use.ps1").read_text(encoding="utf-8")

    assert "Deny-Json" in script
    assert "malformed hook input" in script
    assert "exit 2" in script
