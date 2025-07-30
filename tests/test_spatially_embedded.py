import torch
import pytest
import numpy as np

from bioplnn.models.spatially_embedded import (
    SpatiallyEmbeddedArea,
    SpatiallyEmbeddedAreaConfig,
    SpatiallyEmbeddedRNN,
)


def test_spatially_embedded_area_initialization():
    # Test initialization with default parameters
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    area = SpatiallyEmbeddedArea(config)
    assert area is not None


def test_spatially_embedded_rnn_initialization():
    # Test initialization with default parameters
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    rnn = SpatiallyEmbeddedRNN(num_areas=1, area_configs=[config])
    assert rnn is not None


def test_spatially_embedded_area_forward():
    # Test forward pass
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    area = SpatiallyEmbeddedArea(config)
    input_tensor = torch.rand(1, 3, 32, 32)
    neuron_state = area.init_neuron_state(1)
    output, new_neuron_state = area.forward(input_tensor, neuron_state)
    assert output.shape == (1, 16, 32, 32)


def test_spatially_embedded_rnn_forward():
    # Test forward pass
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    rnn = SpatiallyEmbeddedRNN(num_areas=1, area_configs=[config])
    input_tensor = torch.rand(5, 1, 3, 32, 32)
    output_states, neuron_states, feedback_states = rnn.forward(input_tensor)
    assert output_states[0].shape == (5, 1, 16, 32, 32)


def test_spatially_embedded_area_connectivity():
    """Test that SpatiallyEmbeddedArea correctly routes signals based on connectivity matrix."""
    # Create a simple area with 2 neuron types
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(4, 4),
        in_channels=2,
        out_channels=1,
        num_neuron_types=2,
        num_neuron_subtypes=[2, 2],
        neuron_type_class=["excitatory", "inhibitory"],
        inter_neuron_type_connectivity=np.array([
            [1, 1, 0],  # input connects to both neurons
            [1, 0, 0],  # neuron 0 connects to neuron 0
            [1, 0, 0],  # neuron 1 connects to neuron 0
        ]),
        inter_neuron_type_nonlinearity=np.array([
            ["relu", "relu", "relu"],
            ["relu", "relu", "relu"],
            ["relu", "relu", "relu"],
        ]),
    )
    area = SpatiallyEmbeddedArea(config)
    
    # Create input and initial states
    batch_size = 2
    x = torch.randn(batch_size, 2, 4, 4)
    h0 = area.init_neuron_state(batch_size)
    
    # Forward pass
    out, h1 = area(x, h0)
    
    # Verify output shape
    assert out.shape == (batch_size, 1, 4, 4)
    
    # Verify neuron state shapes
    assert len(h1) == 2
    assert h1[0].shape == (batch_size, 2, 4, 4)
    assert h1[1].shape == (batch_size, 2, 4, 4)
    
    # Verify that the conv layers exist with correct names
    assert "0->0" in area.convs  # input -> neuron 0
    assert "0->1" in area.convs  # input -> neuron 1
    assert "1->0" in area.convs  # neuron 0 -> neuron 0
    assert "2->0" in area.convs  # neuron 1 -> neuron 0
    assert "1->out" in area.out_convs  # neuron 0 -> output


def test_spatially_embedded_rnn_connectivity():
    """Test that SpatiallyEmbeddedRNN correctly routes signals between areas."""
    # Create a simple RNN with 2 areas
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            in_size=(4, 4),
            in_channels=2,
            out_channels=2,
            num_neuron_types=1,
            num_neuron_subtypes=[2],
            neuron_type_class=["excitatory"],
            inter_neuron_type_connectivity=np.array([
                [1, 1],  # input -> neuron 0 and output
                [1, 0],  # neuron 0 -> neuron 0
            ]),
            inter_neuron_type_nonlinearity=np.array([
                ["relu", "relu"],
                ["relu", "relu"],
            ]),
        ),
        SpatiallyEmbeddedAreaConfig(
            in_size=(4, 4),
            in_channels=2,
            out_channels=1,
            num_neuron_types=1,
            num_neuron_subtypes=[2],
            neuron_type_class=["excitatory"],
            inter_neuron_type_connectivity=np.array([
                [1, 1],  # input -> neuron 0 and output
                [1, 0],  # neuron 0 -> neuron 0
            ]),
            inter_neuron_type_nonlinearity=np.array([
                ["relu", "relu"],
                ["relu", "relu"],
            ]),
        ),
    ]
    
    rnn = SpatiallyEmbeddedRNN(
        num_areas=2,
        area_configs=area_configs,
    )
    
    # Create input and initial states
    batch_size = 2
    x = torch.randn(batch_size, 2, 4, 4)
    h0 = rnn.init_neuron_states(batch_size)
    
    # Forward pass
    outs, h1, _ = rnn(x, num_steps=1)
    
    # Verify output shapes
    assert len(outs) == 2
    assert outs[0].shape == (batch_size, 2, 4, 4)
    assert outs[1].shape == (batch_size, 1, 4, 4)
    
    # Verify neuron state shapes
    assert len(h1) == 2
    assert len(h1[0]) == 1
    assert len(h1[1]) == 1
    assert h1[0][0].shape == (batch_size, 2, 4, 4)
    assert h1[1][0].shape == (batch_size, 2, 4, 4)


def test_spatially_embedded_area_feedback():
    """Test that SpatiallyEmbeddedArea correctly handles feedback connections."""
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(4, 4),
        in_channels=2,
        out_channels=1,
        feedback_channels=2,
        num_neuron_types=1,
        num_neuron_subtypes=[2],
        neuron_type_class=["excitatory"],
        inter_neuron_type_connectivity=np.array([
            [1, 0],  # input -> neuron 0
            [1, 0],  # feedback -> neuron 0
            [1, 1],  # neuron 0 -> neuron 0 and output
        ]),
        inter_neuron_type_nonlinearity=np.array([
            ["relu", "relu"],
            ["relu", "relu"],
            ["relu", "relu"],
        ]),
    )
    area = SpatiallyEmbeddedArea(config)
    
    # Create input and initial states
    batch_size = 2
    x = torch.randn(batch_size, 2, 4, 4)
    h0 = area.init_neuron_state(batch_size)
    fb0 = area.init_feedback_state(batch_size)
    
    # Forward pass
    out, h1 = area(x, h0, fb0)
    
    # Verify output shape
    assert out.shape == (batch_size, 1, 4, 4)
    
    # Verify neuron state shape
    assert len(h1) == 1
    assert h1[0].shape == (batch_size, 2, 4, 4)


def test_spatially_embedded_rnn_feedback():
    """Test that SpatiallyEmbeddedRNN correctly handles feedback between areas."""
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            in_size=(4, 4),
            in_channels=2,
            out_channels=2,
            feedback_channels=2,
            num_neuron_types=1,
            num_neuron_subtypes=[2],
            neuron_type_class=["excitatory"],
            inter_neuron_type_connectivity=np.array([
                [1, 0],  # input -> neuron 0
                [1, 0],  # feedback -> neuron 0
                [1, 1],  # neuron 0 -> neuron 0 and output
            ]),
            inter_neuron_type_nonlinearity=np.array([
                ["relu", "relu"],
                ["relu", "relu"],
                ["relu", "relu"],
            ]),
        ),
        SpatiallyEmbeddedAreaConfig(
            in_size=(4, 4),
            in_channels=2,
            out_channels=2,
            num_neuron_types=1,
            num_neuron_subtypes=[2],
            neuron_type_class=["excitatory"],
            inter_neuron_type_connectivity=np.array([[1, 0], [1, 1]]),
            inter_neuron_type_nonlinearity=np.array([["relu", "relu"], ["relu", "relu"]]),
        ),
    ]
    
    rnn = SpatiallyEmbeddedRNN(
        num_areas=2,
        area_configs=area_configs,
        inter_area_feedback_connectivity=np.array([[0, 0], [1, 0]]),
    )
    
    # Create input and initial states
    batch_size = 2
    x = torch.randn(batch_size, 2, 4, 4)
    h0 = rnn.init_neuron_states(batch_size)
    fb0 = rnn.init_feedback_states(batch_size)
    
    # Forward pass
    outs, h1, fbs = rnn(x, num_steps=1)
    
    # Verify output shapes
    assert len(outs) == 2
    assert outs[0].shape == (batch_size, 2, 4, 4)
    assert outs[1].shape == (batch_size, 2, 4, 4)
    
    # Verify neuron state shapes
    assert len(h1) == 2
    assert len(h1[0]) == 1
    assert len(h1[1]) == 1
    assert h1[0][0].shape == (batch_size, 2, 4, 4)
    assert h1[1][0].shape == (batch_size, 2, 4, 4)
    
    # Verify feedback shapes
    assert len(fbs) == 2
    assert fbs[0] is not None
    assert fbs[1] is None
    assert fbs[0].shape == (batch_size, 2, 4, 4)


# TODO: More extensive testing needs to be done for the SpatiallyEmbeddedRNN, SpatiallyEmbeddedArea, and SpatiallyEmbeddedAreaConfig
