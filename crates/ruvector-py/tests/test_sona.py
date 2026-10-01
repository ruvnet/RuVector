"""Tests for ``ruvector.SonaEngine`` — the inference-only SONA binding over
``ruvector_sona::SonaEngine``.

Scope note (see ``src/sona.rs`` module doc comment for the full rationale):
this binds only the forward-pass half (``apply_micro_lora``,
``apply_base_lora``, ``stats``, ``save_state``/``load_state``). Because
both LoRA forward passes are residual and every projection weight is
zero-initialised at construction, a fresh engine's ``apply_*`` calls are
an *exact* identity transform — not merely "close to one" — so these
tests assert exact equality rather than just finite/no-NaN, which is a
stronger (and still honest) claim about untrained behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ruvector


def test_construction_rejects_zero_hidden_dim() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be > 0"):
        ruvector.SonaEngine(0)


def test_construction_and_properties() -> None:
    engine = ruvector.SonaEngine(64)
    assert engine.hidden_dim == 64
    assert engine.num_layers == 12  # fixed default, see src/sona.rs
    assert engine.is_enabled is True


def test_is_enabled_round_trips_through_the_setter() -> None:
    engine = ruvector.SonaEngine(32)
    assert engine.is_enabled is True
    engine.is_enabled = False
    assert engine.is_enabled is False
    engine.is_enabled = True
    assert engine.is_enabled is True


def test_apply_micro_lora_fresh_engine_is_exact_identity() -> None:
    engine = ruvector.SonaEngine(64)
    input_vec = np.arange(64, dtype=np.float32)
    output = engine.apply_micro_lora(input_vec)

    assert output.shape == (64,)
    assert np.all(np.isfinite(output))
    # Zero-initialised up-projection + residual forward pass => exact
    # identity on an untrained engine (see module doc comment).
    np.testing.assert_array_equal(output, input_vec)


def test_apply_base_lora_fresh_engine_is_exact_identity() -> None:
    engine = ruvector.SonaEngine(64)
    input_vec = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    output = engine.apply_base_lora(0, input_vec)

    assert output.shape == (64,)
    assert np.all(np.isfinite(output))
    np.testing.assert_array_equal(output, input_vec)


def test_apply_micro_lora_rejects_wrong_length() -> None:
    engine = ruvector.SonaEngine(64)
    bad_input = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="hidden_dim"):
        engine.apply_micro_lora(bad_input)


def test_apply_base_lora_rejects_wrong_length() -> None:
    engine = ruvector.SonaEngine(64)
    bad_input = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="hidden_dim"):
        engine.apply_base_lora(0, bad_input)


def test_apply_micro_lora_rejects_non_contiguous_input() -> None:
    engine = ruvector.SonaEngine(8)
    base = np.arange(16, dtype=np.float32)
    non_contig = base[::2]  # a strided view, not C-contiguous, len 8
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError, match="C-contiguous"):
        engine.apply_micro_lora(non_contig)


def test_apply_base_lora_rejects_out_of_range_layer() -> None:
    engine = ruvector.SonaEngine(64)
    input_vec = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError, match="num_layers"):
        engine.apply_base_lora(engine.num_layers, input_vec)


def test_stats_returns_sane_dict() -> None:
    engine = ruvector.SonaEngine(64)
    stats = engine.stats()

    expected_keys = {
        "trajectories_recorded",
        "trajectories_buffered",
        "trajectories_dropped",
        "buffer_success_rate",
        "patterns_stored",
        "patterns_learned",
        "ewc_tasks",
        "instant_enabled",
        "background_enabled",
    }
    assert expected_keys <= set(stats.keys())
    # A fresh engine has recorded nothing yet.
    assert stats["trajectories_recorded"] == 0
    assert stats["trajectories_buffered"] == 0
    assert stats["patterns_stored"] == 0
    assert stats["instant_enabled"] is True
    assert stats["background_enabled"] is True


def test_save_load_state_round_trip_on_fresh_engine() -> None:
    engine = ruvector.SonaEngine(32)
    state = engine.save_state()
    assert isinstance(state, str)

    # Honest claim: a fresh engine has no patterns, so loading its own
    # state back restores zero patterns (not persisted here: LoRA weights
    # -- see module doc comment).
    restored_count = engine.load_state(state)
    assert restored_count == 0

    # Round-tripped JSON must at least carry the documented keys.
    parsed = json.loads(state)
    assert "patterns" in parsed
    assert "ewc_task_count" in parsed


def test_load_state_rejects_malformed_json() -> None:
    engine = ruvector.SonaEngine(32)
    with pytest.raises(ruvector.RuVectorError):
        engine.load_state("not valid json {{{")


def test_repr_is_diagnostic() -> None:
    engine = ruvector.SonaEngine(48)
    r = repr(engine)
    assert "SonaEngine" in r
    assert "hidden_dim=48" in r
