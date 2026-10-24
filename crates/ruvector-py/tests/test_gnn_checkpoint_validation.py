import json

import numpy as np
import pytest
import ruvector


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_dim", 0),
        ("input_dim", 3),
        ("hidden_dim", 0),
        ("hidden_dim", 16),
        ("heads", 0),
        ("heads", 3),
        ("dropout", -1),
        ("dropout", 2),
    ],
)
def test_checkpoint_refuses_inconsistent_envelope(field, value):
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    envelope[field] = value
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


@pytest.mark.parametrize("field,value", [("head_dim", 0), ("num_heads", 3)])
def test_checkpoint_refuses_inconsistent_attention(field, value):
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    envelope["layer"]["attention"][field] = value
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


@pytest.mark.parametrize(
    "path",
    [
        ("w_msg", "weights"),
        ("w_msg", "bias"),
        ("w_agg", "weights"),
        ("w_update", "w_z", "weights"),
        ("w_update", "u_h", "weights"),
        ("attention", "q_linear", "weights"),
        ("attention", "out_linear", "bias"),
        ("norm", "gamma"),
        ("norm", "beta"),
    ],
)
def test_checkpoint_refuses_inconsistent_tensor_shape(path):
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    tensor = envelope["layer"]
    for part in path:
        tensor = tensor[part]
    tensor["dim"][0] -= 1
    tensor["data"] = tensor["data"][: int(np.prod(tensor["dim"]))]
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


@pytest.mark.parametrize("epsilon", [0.0, -1.0])
def test_checkpoint_refuses_invalid_normalization_epsilon(epsilon):
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    envelope["layer"]["norm"]["eps"] = epsilon
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


def test_checkpoint_refuses_inner_dropout_mismatch():
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    envelope["layer"]["dropout"] = 0.5
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


@pytest.mark.parametrize("path", [("w_msg", "weights"), ("norm", "beta")])
def test_checkpoint_refuses_parameter_overflow(path):
    envelope = json.loads(ruvector.GnnLayer(4, 8, 2).to_json())
    tensor = envelope["layer"]
    for part in path:
        tensor = tensor[part]
    tensor["data"][0] = 1e308
    with pytest.raises((ValueError, ruvector.RuVectorError)):
        ruvector.GnnLayer.from_json(json.dumps(envelope))


def inputs():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), np.array(
        [[0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )


def test_unmodified_checkpoint_roundtrip_control():
    layer = ruvector.GnnLayer(4, 8, 2)
    restored = ruvector.GnnLayer.from_json(layer.to_json())
    node, neighbors = inputs()
    np.testing.assert_array_equal(
        restored.forward(node, neighbors), layer.forward(node, neighbors)
    )


def test_valid_modified_weights_survive_checkpoint_roundtrip():
    original = ruvector.GnnLayer(4, 8, 2)
    envelope = json.loads(original.to_json())
    envelope["layer"]["norm"]["beta"]["data"][0] = 2.0
    envelope["layer"]["w_msg"]["weights"]["data"][0] = 0.125
    loaded = ruvector.GnnLayer.from_json(json.dumps(envelope))
    restored = ruvector.GnnLayer.from_json(loaded.to_json())
    assert json.loads(restored.to_json()) == json.loads(loaded.to_json())
    node, neighbors = inputs()
    np.testing.assert_array_equal(
        restored.forward(node, neighbors), loaded.forward(node, neighbors)
    )
    assert not np.array_equal(
        loaded.forward(node, neighbors), original.forward(node, neighbors)
    )
