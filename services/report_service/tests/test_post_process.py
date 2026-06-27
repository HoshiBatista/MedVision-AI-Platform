"""Unit tests for report post-processing."""

from app.services.report_generator import _DISCLAIMER, _post_process


def test_post_process_trims_incomplete_tail_and_appends_disclaimer():
    result = _post_process("  This is   complete. Trailing incomplete")
    assert _DISCLAIMER in result
    assert "This is complete." in result
    assert "Trailing incomplete" not in result  # incomplete tail dropped


def test_post_process_collapses_whitespace():
    result = _post_process("a\n\n  b    c.")
    body = result.replace(_DISCLAIMER, "")
    assert "  " not in body  # no doubled spaces left


def test_post_process_always_appends_disclaimer():
    assert _post_process("no terminal punctuation here").endswith(_DISCLAIMER)
