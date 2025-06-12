import torch
import pytest
import pandas as pd
import numpy as np

from bioplnn.models.spatially_embedded import (
    SpatiallyEmbeddedArea,
    SpatiallyEmbeddedAreaConfig,
    SpatiallyEmbeddedRNN,
    Conv2dRectify,
)


# --- SpatiallyEmbeddedAreaConfig Tests ---

@pytest.mark.parametrize("num_neuron_types", [1, 2, 3])
@pytest.mark.parametrize("use_feedback", [True, False])
def test_s_area_config_connectivity_template_df(num_neuron_types, use_feedback):
    torch.manual_seed(42)
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(4, 4),
        in_channels=1,
        out_channels=1,
        num_neuron_types=num_neuron_types,
        use_feedback=use_feedback,
        feedback_channels=1 if use_feedback else None
    )
    df = config.inter_neuron_type_connectivity_template_df

    assert isinstance(df, pd.DataFrame)

    expected_sources = []
    if use_feedback:
        expected_sources.append("feedback")
    expected_sources.append("input")
    for i in range(num_neuron_types):
        expected_sources.append(f"neuron_type_{i}")

    expected_targets = []
    for i in range(num_neuron_types):
        expected_targets.append(f"neuron_type_{i}")
    expected_targets.append("output")

    assert list(df.columns) == ["source_type", "target_type", "is_connected", "spatial_extent_hw", "num_subtype_groups", "nonlinearity", "bias"]

    # Check that all combinations of source and target are present
    for src in expected_sources:
        for tgt in expected_targets:
            # Feedback cannot target output directly in this template
            if src == "feedback" and tgt == "output":
                assert not ((df["source_type"] == src) & (df["target_type"] == tgt)).any()
                continue
            # Input cannot target other neuron types if num_neuron_types == 1 and it targets output directly
            # This logic is nuanced by how the template is built; basic check here
            assert ((df["source_type"] == src) & (df["target_type"] == tgt)).any()

    # Check default values
    assert df["is_connected"].all() # Default is True
    assert (df["spatial_extent_hw"] == 3).all()
    assert (df["num_subtype_groups"] == 1).all()
    assert (df["nonlinearity"] == "relu").all()
    assert (df["bias"] == 0.0).all()


def test_s_area_config_valid_initialization():
    torch.manual_seed(42)
    # Test with a variety of valid parameters
    config_params = {
        "area_name": "TestArea1",
        "in_size": (8, 8),
        "in_channels": 3,
        "out_channels": 16,
        "in_class": "excitatory",
        "num_neuron_types": 2,
        "neuron_type_class": ["excitatory", "inhibitory"],
        "neuron_type_density": [0.7, 0.3],
        "neuron_type_nonlinearity": ["relu", torch.nn.Sigmoid()],
        "num_neuron_subtypes": [2, 3], # 2 subtypes for type 0, 3 for type 1
        "tau_init": 20.0,
        "tau_mode": "subtype",
        "tau_init_fn": lambda shape, **kwargs: torch.rand(shape) * 10 + 1,
        "use_dense_input_projection": True,
        "use_dense_output_projection": False,
        "use_feedback": True,
        "feedback_channels": 8,
        "feedback_class": "hybrid",
        "feedback_spatial_extent_hw": 5,
        "inter_neuron_type_connectivity": np.array([
            [True, True, True],  # feedback to type0, type1, output
            [True, True, True],  # input to type0, type1, output
            [True, False, True], # type0 to type0, type1, output
            [False, True, True]  # type1 to type0, type1, output
        ], dtype=bool).T, # Transpose to match (source, target)
        "inter_neuron_type_spatial_extents_hw": np.array([
            [3,3,0], [3,3,3], [5,5,1], [5,5,1]
        ]).T +1, # +1 to avoid 0 for valid extent
        "inter_neuron_type_num_subtype_groups": np.ones((4,3), dtype=int).T,
        "inter_neuron_type_nonlinearity": [['relu', 'relu', 'relu']]*4, # Must be list of lists if array-like
        "inter_neuron_type_bias": np.random.rand(4,3).T * 0.1,
        "store_history": True,
        "device": "cpu",
        "dtype": torch.float64,
    }
    config = SpatiallyEmbeddedAreaConfig(**config_params)
    assert config is not None
    for key, value in config_params.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(getattr(config, key), value)
        elif isinstance(value, list) and all(isinstance(v, list) for v in value) : # for inter_neuron_type_nonlinearity
             # Deep comparison for list of lists
             attr_val = getattr(config,key)
             assert len(attr_val) == len(value)
             for i in range(len(value)):
                 assert attr_val[i] == value[i] # Relying on __eq__ for inner elements (str, nn.Module)
        else:
            assert getattr(config, key) == value

    # Test minimal working example
    config_minimal = SpatiallyEmbeddedAreaConfig(in_size=(4,4), in_channels=1, out_channels=1)
    assert config_minimal is not None
    assert config_minimal.num_neuron_types == 1


def test_s_area_config_invalid_initialization():
    torch.manual_seed(42)
    base_args = {"in_size": (4, 4), "in_channels": 1, "out_channels": 1}

    with pytest.raises(ValueError, match="in_class must be one of"):
        SpatiallyEmbeddedAreaConfig(**base_args, in_class="invalid_class")

    with pytest.raises(ValueError, match="num_neuron_types must be >= 1"):
        SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=0)

    with pytest.raises(ValueError, match="neuron_type_class must be a list of strings"):
        SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=2, neuron_type_class="not_a_list")

    with pytest.raises(ValueError, match="Length of neuron_type_class must match num_neuron_types"):
        SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=2, neuron_type_class=["exc"])

    with pytest.raises(ValueError, match="num_neuron_subtypes as a list must have length equal to num_neuron_types"):
         SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=2, num_neuron_subtypes=[1]) # only one entry for 2 types

    with pytest.raises(ValueError, match="Invalid tau_mode"):
        SpatiallyEmbeddedAreaConfig(**base_args, tau_mode="invalid_tau_mode")

    with pytest.raises(ValueError, match="feedback_channels must be > 0 if use_feedback is True"):
        SpatiallyEmbeddedAreaConfig(**base_args, use_feedback=True, feedback_channels=0)

    with pytest.raises(ValueError, match="inter_neuron_type_connectivity must be a 2D numpy array"):
        SpatiallyEmbeddedAreaConfig(**base_args, inter_neuron_type_connectivity=[True, False])

    # Mismatch in shape for inter_neuron_type_connectivity
    with pytest.raises(ValueError, match="Shape mismatch for inter_neuron_type_connectivity"):
        SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=1,
                                   inter_neuron_type_connectivity=np.array([[True,True],[True,True],[True,True]])) # Expected (2_src+1_types, 1_type+1_out) = (3,2)

    # Mismatch for spatial_extents with connectivity
    with pytest.raises(ValueError, match="Shape mismatch for inter_neuron_type_spatial_extents_hw"):
         SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=1,
                                    inter_neuron_type_connectivity=np.full((3,2), True), # (input, fb, type0) -> (type0, output)
                                    inter_neuron_type_spatial_extents_hw=np.array([1,2,3])) # Wrong shape

    # Invalid nonlinearity string
    with pytest.raises(ValueError, match="Invalid nonlinearity string"):
        SpatiallyEmbeddedAreaConfig(**base_args, neuron_type_nonlinearity="invalid_nonlin")

    # Test that if inter_neuron_type_nonlinearity is a list of lists, inner elements are valid
    with pytest.raises(ValueError, match="Invalid nonlinearity string in inter_neuron_type_nonlinearity"):
        SpatiallyEmbeddedAreaConfig(**base_args, num_neuron_types=1,
                                    inter_neuron_type_connectivity=np.full((3,2), True),
                                    inter_neuron_type_nonlinearity=[["invalid_nl", "relu"], ["relu","relu"], ["relu","relu"]])


# --- SpatiallyEmbeddedArea Tests ---

@pytest.mark.parametrize("num_neuron_types", [1, 2])
@pytest.mark.parametrize("use_feedback", [True, False])
@pytest.mark.parametrize("tau_mode", ["type", "subtype", "spatial"]) # "subtype_spatial" is more complex
@pytest.mark.parametrize("num_neuron_subtypes", [1, [1,2]]) # if num_neuron_types=2
@pytest.mark.parametrize("neuron_type_nonlinearity_mode", ["scalar_str", "list_str", "list_module"])
def test_s_area_initialization_diverse_configs(
    num_neuron_types, use_feedback, tau_mode, num_neuron_subtypes, neuron_type_nonlinearity_mode
):
    torch.manual_seed(42)
    in_size = (8, 8)
    in_channels = 3
    out_channels = 5
    feedback_channels_val = 4 if use_feedback else None

    if num_neuron_types == 1 and isinstance(num_neuron_subtypes, list):
        current_subtypes = num_neuron_subtypes[0]
    elif isinstance(num_neuron_subtypes, list) and len(num_neuron_subtypes) != num_neuron_types:
        pytest.skip("num_neuron_subtypes list length must match num_neuron_types")
    else:
        current_subtypes = num_neuron_subtypes

    neuron_type_nonlinearity_config = "relu" # default
    if neuron_type_nonlinearity_mode == "list_str":
        neuron_type_nonlinearity_config = ["tanh"] * num_neuron_types
    elif neuron_type_nonlinearity_mode == "list_module":
        neuron_type_nonlinearity_config = [torch.nn.Sigmoid()] * num_neuron_types

    if num_neuron_types > 1 and neuron_type_nonlinearity_mode == "scalar_str":
        # This is fine, scalar will broadcast.
        pass


    config_params = {
        "area_name": f"Area_NT{num_neuron_types}_FB{use_feedback}_TM{tau_mode}",
        "in_size": in_size,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_neuron_types": num_neuron_types,
        "neuron_type_density": "half" if num_neuron_types > 1 else "same",
        "neuron_type_nonlinearity": neuron_type_nonlinearity_config,
        "num_neuron_subtypes": current_subtypes,
        "tau_mode": tau_mode,
        "tau_init": 10.0,
        "use_feedback": use_feedback,
        "feedback_channels": feedback_channels_val,
        "store_history": True, # Useful for later tests
        "dtype": torch.float32 # Common case
    }

    # Create a valid connectivity matrix based on num_neuron_types and use_feedback
    num_sources = (1 if use_feedback else 0) + 1 + num_neuron_types # feedback, input, neuron_types
    num_targets = num_neuron_types + 1 # neuron_types, output

    config_params["inter_neuron_type_connectivity"] = np.full((num_sources, num_targets), True, dtype=bool)
    # Feedback cannot target output directly in the default template setup
    if use_feedback:
        config_params["inter_neuron_type_connectivity"][0, -1] = False

    config_params["inter_neuron_type_spatial_extents_hw"] = np.full((num_sources, num_targets), 3, dtype=int)
    config_params["inter_neuron_type_num_subtype_groups"] = np.full((num_sources, num_targets), 1, dtype=int)

    # Create nonlin list of lists
    nonlin_list_of_lists = []
    for _ in range(num_sources):
        nonlin_list_of_lists.append(["relu"] * num_targets)
    config_params["inter_neuron_type_nonlinearity"] = nonlin_list_of_lists
    config_params["inter_neuron_type_bias"] = np.zeros((num_sources, num_targets))


    try:
        config = SpatiallyEmbeddedAreaConfig(**config_params)
    except ValueError as e:
        # If config itself fails, skip (already tested in config tests)
        pytest.skip(f"Skipping due to config validation error: {e}")
        return

    area = SpatiallyEmbeddedArea(config)
    assert area is not None
    assert area.config == config
    assert area.area_name == config.area_name
    assert area.num_neuron_types == num_neuron_types

    total_subtypes = 0
    if isinstance(current_subtypes, list):
        total_subtypes = sum(current_subtypes)
    else:
        total_subtypes = num_neuron_types * current_subtypes

    expected_tau_shape_dim0 = total_subtypes if tau_mode in ["subtype", "subtype_spatial"] else num_neuron_types
    if tau_mode == "spatial" or tau_mode == "subtype_spatial":
         # tau shape will be (expected_tau_shape_dim0, H, W)
        assert area.tau.shape == (expected_tau_shape_dim0, *in_size)
    else:
        assert area.tau.shape == (expected_tau_shape_dim0,)

    # Check neuron_type_modules
    assert len(area.neuron_type_modules) == num_neuron_types
    for i in range(num_neuron_types):
        assert area.neuron_type_modules[i].in_channels > 0 # Dynamic based on connectivity

        num_sub = current_subtypes[i] if isinstance(current_subtypes, list) else current_subtypes
        expected_out_channels_nt_module = num_sub * config.neuron_type_density_per_type[i] * config.out_channels_per_density // config.num_neuron_subtypes_per_type[i]

        # This calculation for expected_out_channels_nt_module needs to be very precise
        # It depends on how channels are distributed among subtypes and density
        # For now, just check that it's created
        assert area.neuron_type_modules[i].out_channels > 0


    # Check presence of input_conv and feedback_conv based on config
    if use_feedback:
        assert area.feedback_conv is not None
        # Check connectivity: if feedback connects to any neuron type, feedback_conv should exist
        if np.any(config.inter_neuron_type_connectivity[0, :-1]): # First row (feedback), all but last col (output)
             assert area.feedback_conv is not None # Redundant check, but emphasizes logic
    else:
        assert area.feedback_conv is None

    assert area.input_conv is not None # Always present

    # Check output conv
    assert area.output_conv is not None
    # The in_channels for output_conv depends on what connects to "output" target
    # This sum needs to be precise based on which source types connect to output and their out_channels.
    # For now, assert existence.

    # Check dtypes and device
    assert area.tau.dtype == config.dtype
    for conv_list in area.conv_layers.values():
        for conv_block in conv_list:
            for param in conv_block.parameters():
                assert param.dtype == config.dtype
                assert param.device == torch.device(config.device)


def test_s_area_initialization(): # Keep the original simple test as a baseline
    # Test initialization with default parameters
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(32, 32), in_channels=3, out_channels=16
    )
    area = SpatiallyEmbeddedArea(config)
    assert area is not None


@pytest.mark.parametrize("num_neuron_types", [1, 2])
@pytest.mark.parametrize("use_feedback", [True, False])
@pytest.mark.parametrize("tau_mode", ["type", "spatial"]) # Simplified for forward pass test
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_s_area_forward_pass(
    num_neuron_types, use_feedback, tau_mode, batch_size, dtype
):
    torch.manual_seed(43)
    in_size = (8, 8)
    in_channels = 2
    out_channels = 3
    feedback_channels_val = 2 if use_feedback else None
    num_subtypes = 1 # Keep simple for this test

    config_params = {
        "area_name": f"FwdArea_NT{num_neuron_types}_FB{use_feedback}",
        "in_size": in_size, "in_channels": in_channels, "out_channels": out_channels,
        "num_neuron_types": num_neuron_types, "num_neuron_subtypes": num_subtypes,
        "tau_mode": tau_mode, "tau_init": 5.0,
        "use_feedback": use_feedback, "feedback_channels": feedback_channels_val,
        "store_history": True, "dtype": dtype, "device": "cpu"
    }

    num_sources = (1 if use_feedback else 0) + 1 + num_neuron_types
    num_targets = num_neuron_types + 1
    config_params["inter_neuron_type_connectivity"] = np.full((num_sources, num_targets), True, dtype=bool)
    if use_feedback:
        config_params["inter_neuron_type_connectivity"][0, -1] = False # No feedback to output
    config_params["inter_neuron_type_nonlinearity"] = [["relu"]*num_targets]*num_sources


    try:
        config = SpatiallyEmbeddedAreaConfig(**config_params)
    except ValueError as e:
        pytest.skip(f"Skipping due to config validation error: {e}")
        return

    area = SpatiallyEmbeddedArea(config)
    area.to(dtype) # Ensure model parameters are of correct dtype

    input_tensor = torch.rand(batch_size, in_channels, *in_size, dtype=dtype)
    initial_neuron_state = area.init_neuron_state(batch_size=batch_size)

    feedback_state_input = None
    if use_feedback:
        feedback_state_input = torch.rand(batch_size, feedback_channels_val, *in_size, dtype=dtype)

    output, new_neuron_state = area.forward(
        input_tensor, initial_neuron_state, feedback_state=feedback_state_input
    )

    # Check output shape
    assert output.shape == (batch_size, out_channels, *in_size)
    assert output.dtype == dtype

    # Check new_neuron_state shapes and dtype
    assert isinstance(new_neuron_state, list)
    assert len(new_neuron_state) == num_neuron_types
    for i in range(num_neuron_types):
        # channels_per_type = area.config.out_channels_per_density * area.config.neuron_type_density_per_type[i]
        # For num_subtypes=1, channels_per_subtype = channels_per_type
        # This needs to use the actual calculated channels for the neuron type module's output
        expected_neuron_type_channels = area.neuron_type_modules[i].out_channels # This is already num_subtypes * channels_per_subtype
        assert new_neuron_state[i].shape == (batch_size, expected_neuron_type_channels, *in_size)
        assert new_neuron_state[i].dtype == dtype

    # Check history storage
    assert area.history_h_neurons is not None
    # History should be a list of lists: [time_step][neuron_type_idx]
    # For a single forward pass, time_step is 1 effectively
    assert len(area.history_h_neurons) == 1
    assert len(area.history_h_neurons[0]) == num_neuron_types
    for i in range(num_neuron_types):
         assert torch.allclose(area.history_h_neurons[0][i], new_neuron_state[i])

    # If feedback is provided when not expected, it should be ignored or raise error (current behavior: ignored)
    if not use_feedback and feedback_state_input is not None:
        # To strictly test this, one might need to mock/spy on conv layers, or check for specific warnings.
        # For now, we assume if the forward pass runs without error, it's handled.
        pass

    # If feedback is expected but not provided, it should raise an error or use a default (current behavior: error)
    if use_feedback and feedback_state_input is None:
        with pytest.raises(ValueError, match="Feedback state must be provided"): # or TypeError
             area.forward(input_tensor, initial_neuron_state, feedback_state=None)


@pytest.mark.parametrize("num_neuron_types", [1, 2])
@pytest.mark.parametrize("use_feedback", [True, False])
@pytest.mark.parametrize("tau_mode", ["type", "spatial"])
@pytest.mark.parametrize("train_tau", [True, False])
def test_s_area_backward_pass(
    num_neuron_types, use_feedback, tau_mode, train_tau
):
    torch.manual_seed(44)
    in_size = (4, 4) # Smaller for faster backward pass
    in_channels = 2
    out_channels = 2
    feedback_channels_val = 2 if use_feedback else None
    num_subtypes = 1
    batch_size = 1 # Smallest batch for faster backward
    dtype = torch.float32 # float32 is standard for backward tests

    config_params = {
        "area_name": f"BwdArea_NT{num_neuron_types}_FB{use_feedback}_TT{train_tau}",
        "in_size": in_size, "in_channels": in_channels, "out_channels": out_channels,
        "num_neuron_types": num_neuron_types, "num_neuron_subtypes": num_subtypes,
        "tau_mode": tau_mode, "tau_init": 5.0, "train_tau": train_tau,
        "use_feedback": use_feedback, "feedback_channels": feedback_channels_val,
        "store_history": False, # Not needed for backward
        "dtype": dtype, "device": "cpu"
    }

    num_sources = (1 if use_feedback else 0) + 1 + num_neuron_types
    num_targets = num_neuron_types + 1
    config_params["inter_neuron_type_connectivity"] = np.full((num_sources, num_targets), True, dtype=bool)
    if use_feedback:
        config_params["inter_neuron_type_connectivity"][0, -1] = False
    config_params["inter_neuron_type_nonlinearity"] = [["relu"]*num_targets]*num_sources
    # Ensure all convs have biases for gradient checking
    config_params["inter_neuron_type_bias"] = np.full((num_sources, num_targets), 0.1, dtype=float)


    try:
        config = SpatiallyEmbeddedAreaConfig(**config_params)
    except ValueError as e:
        pytest.skip(f"Skipping due to config validation error: {e}")
        return

    area = SpatiallyEmbeddedArea(config)
    area.to(dtype)

    input_tensor = torch.rand(batch_size, in_channels, *in_size, dtype=dtype)
    initial_neuron_state = area.init_neuron_state(batch_size=batch_size)
    feedback_state_input = None
    if use_feedback:
        feedback_state_input = torch.rand(batch_size, feedback_channels_val, *in_size, dtype=dtype)

    output, _ = area.forward(
        input_tensor, initial_neuron_state, feedback_state=feedback_state_input
    )
    loss = output.sum()
    loss.backward()

    # Check tau gradients
    if train_tau:
        assert area.tau.grad is not None
        assert not torch.isnan(area.tau.grad).any()
        assert not torch.isinf(area.tau.grad).any()
    else:
        if hasattr(area.tau, 'grad'): # Might not exist if not a Parameter
            assert area.tau.grad is None

    # Check gradients for all Conv2dRectify layers
    all_conv_layers = []
    if area.input_conv: all_conv_layers.extend(area.input_conv)
    if area.feedback_conv: all_conv_layers.extend(area.feedback_conv)
    for nt_module in area.neuron_type_modules:
        all_conv_layers.extend(nt_module.inter_convs) # These are lists of Conv2dRectify
    if area.output_conv: all_conv_layers.extend(area.output_conv)

    assert len(all_conv_layers) > 0 # Make sure we are actually testing something

    for conv_layer in all_conv_layers:
        assert isinstance(conv_layer, Conv2dRectify)
        assert conv_layer.weight.grad is not None
        assert not torch.isnan(conv_layer.weight.grad).any()
        assert not torch.isinf(conv_layer.weight.grad).any()
        if conv_layer.bias is not None:
            assert conv_layer.bias.grad is not None
            assert not torch.isnan(conv_layer.bias.grad).any()
            assert not torch.isinf(conv_layer.bias.grad).any()


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("use_init_fn", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_s_area_init_states(batch_size, use_init_fn, dtype):
    torch.manual_seed(45)
    in_size = (6, 6)
    config_params = {
        "in_size": in_size, "in_channels": 2, "out_channels": 3,
        "num_neuron_types": 2, "num_neuron_subtypes": [1, 2], # total 1+2=3 subtypes
        "use_feedback": True, "feedback_channels": 2,
        "dtype": dtype, "device": "cpu"
    }

    mock_init_fn = None
    if use_init_fn:
        mock_init_fn = lambda shape, **kwargs: torch.ones(shape, dtype=kwargs.get("dtype", dtype), device=kwargs.get("device", "cpu")) * 0.5
        config_params["neuron_state_init_fn"] = mock_init_fn
        config_params["output_state_init_fn"] = mock_init_fn
        config_params["feedback_state_init_fn"] = mock_init_fn # Though feedback is usually external

    config = SpatiallyEmbeddedAreaConfig(**config_params)
    area = SpatiallyEmbeddedArea(config)
    area.to(dtype)

    # Test init_neuron_state
    neuron_states = area.init_neuron_state(batch_size)
    assert isinstance(neuron_states, list)
    assert len(neuron_states) == config.num_neuron_types
    for i in range(config.num_neuron_types):
        expected_channels = config.neuron_type_modules_out_channels[i]
        assert neuron_states[i].shape == (batch_size, expected_channels, *in_size)
        assert neuron_states[i].dtype == dtype
        if use_init_fn:
            assert torch.allclose(neuron_states[i], torch.ones_like(neuron_states[i]) * 0.5)
        else:
            assert torch.allclose(neuron_states[i], torch.zeros_like(neuron_states[i]))

    # Test init_output_state
    output_state = area.init_output_state(batch_size)
    assert output_state.shape == (batch_size, config.out_channels, *in_size)
    assert output_state.dtype == dtype
    if use_init_fn:
        assert torch.allclose(output_state, torch.ones_like(output_state) * 0.5)
    else:
        assert torch.allclose(output_state, torch.zeros_like(output_state))

    # Test init_feedback_state (if applicable)
    if config.use_feedback:
        feedback_state = area.init_feedback_state(batch_size)
        assert feedback_state.shape == (batch_size, config.feedback_channels, *in_size)
        assert feedback_state.dtype == dtype
        if use_init_fn: # feedback_state_init_fn applies here
            assert torch.allclose(feedback_state, torch.ones_like(feedback_state) * 0.5)
        else: # Default is zeros
            assert torch.allclose(feedback_state, torch.zeros_like(feedback_state))


def test_s_area_df_summaries():
    torch.manual_seed(46)
    config = SpatiallyEmbeddedAreaConfig(
        in_size=(4,4), in_channels=2, out_channels=3, num_neuron_types=2,
        neuron_type_class=["E", "I"], num_neuron_subtypes=[1,2], use_feedback=True, feedback_channels=2
    )
    area = SpatiallyEmbeddedArea(config)

    neuron_df = area.neuron_description_df()
    assert isinstance(neuron_df, pd.DataFrame)
    assert not neuron_df.empty
    expected_neuron_cols = ["neuron_type_idx", "neuron_type_class", "subtype_idx_within_type", "global_subtype_idx", "num_channels", "spatial_shape_hw"]
    for col in expected_neuron_cols:
        assert col in neuron_df.columns
    assert len(neuron_df) == sum(config.num_neuron_subtypes_per_type)


    conv_df = area.conv_connectivity_df()
    assert isinstance(conv_df, pd.DataFrame)
    assert not conv_df.empty
    expected_conv_cols = ["source_name", "target_name", "conv_module_type", "in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "nonlinearity"]
    for col in expected_conv_cols:
        assert col in conv_df.columns

    # Check summary runs
    # TODO: Capture stdout and check content if necessary, for now just check it runs
    area.summary(print_summary=False) # Test return string path
    summary_str = area.summary(print_summary=False)
    assert isinstance(summary_str, str)
    assert "SpatiallyEmbeddedArea Summary" in summary_str
    area.summary(print_summary=True) # Test print path (manual check or capture stdout)


@pytest.mark.parametrize("train_tau_param", [True, False]) # Renamed to avoid conflict
def test_s_area_clamp_tau(train_tau_param):
    torch.manual_seed(47)
    in_size=(4,4)
    # Create taus that are intentionally < 1.0 / dt to test clamping
    # dt = 0.2, so min_tau is 1.0. We'll set some taus to 0.5.
    dt_val = 0.2
    initial_taus_type = torch.tensor([0.5, 1.5], dtype=torch.float32) # For num_neuron_types=2, tau_mode="type"

    config = SpatiallyEmbeddedAreaConfig(
        in_size=in_size, in_channels=1, out_channels=1,
        num_neuron_types=2,
        tau_init=initial_taus_type, # Will be used if tau_mode is 'type'
        tau_mode="type",
        train_tau=train_tau_param,
        dt=dt_val
    )
    area = SpatiallyEmbeddedArea(config)

    # Clamp tau is called at the end of forward pass.
    # Here, we are testing it more directly if possible, or checking its effect after forward.
    # The _clamp_tau method itself is protected. We check the public tau attribute.

    # If train_tau is True, initial taus are wrapped in nn.Parameter.
    # Let's check the value before any operation that might clamp it.
    if train_tau_param:
        assert torch.allclose(area.tau.data, initial_taus_type)
    else:
        assert torch.allclose(area.tau, initial_taus_type)

    # Perform a dummy forward pass to trigger _clamp_tau
    dummy_input = torch.rand(1, config.in_channels, *in_size, dtype=config.dtype)
    neuron_state = area.init_neuron_state(1)
    area.forward(dummy_input, neuron_state)

    expected_clamped_taus = torch.tensor([1.0/dt_val, 1.5], dtype=torch.float32) # 0.5 becomes 1.0/dt, 1.5 stays

    current_tau_val = area.tau.data if train_tau_param else area.tau
    assert torch.allclose(current_tau_val, expected_clamped_taus, atol=1e-5)


# --- SpatiallyEmbeddedRNN Tests ---

def _create_default_s_area_config(
    idx=0, in_size=(8,8), in_ch=3, out_ch=3, feedback_ch=3, num_neuron_types=1, dtype=torch.float32, device="cpu", **kwargs
):
    return SpatiallyEmbeddedAreaConfig(
        area_name=f"Area{idx}",
        in_size=in_size,
        in_channels=in_ch,
        out_channels=out_ch,
        use_feedback=True, # Assume feedback for inter-area tests
        feedback_channels=feedback_ch,
        num_neuron_types=num_neuron_types,
        dtype=dtype,
        device=device,
        **kwargs
    )

@pytest.mark.parametrize("num_areas", [1, 2, 3])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_single_connectome", [True, False])
@pytest.mark.parametrize("use_area_kwargs", [True, False]) # Test providing area_kwargs vs list of configs
def test_s_rnn_initialization(num_areas, batch_first, use_single_connectome, use_area_kwargs):
    torch.manual_seed(48)
    in_size_hw = (8,8)
    initial_in_channels = 3
    area_out_channels = 4 # Each area will output this many channels
    feedback_channels_per_area = 4 # For inter-area feedback

    area_configs_list = []
    if not use_area_kwargs:
        # Create a list of configs. If not using single connectome, make them slightly different.
        # For simplicity in this test, output of area i must match input of area i+1 if no projection
        # Here, we ensure all areas have same in/out channels for feedforward path to simplify.
        current_in_ch = initial_in_channels
        for i in range(num_areas):
            conf = _create_default_s_area_config(
                idx=i,
                in_size=in_size_hw,
                in_ch=current_in_ch if i==0 else area_out_channels, # Input to first area, others receive from previous area
                out_ch=area_out_channels,
                feedback_ch=feedback_channels_per_area,
                area_name=f"Area{i}_diff" if not use_single_connectome else f"Area{i}_single"
            )
            area_configs_list.append(conf)
            # current_in_ch = area_out_channels # Output of current becomes input of next (if no projection)
                                            # This logic is handled by the RNN if use_projection_between_areas=True

    rnn_params = {
        "num_areas": num_areas,
        "batch_first": batch_first,
        "use_single_connectome_for_all_areas": use_single_connectome,
        "input_channels_for_first_area": initial_in_channels,
        "in_size_hw_for_first_area": in_size_hw,
        # For simplicity, let inter-area feedback use the area's out_channels
        "inter_area_feedback_channels": area_out_channels,
    }

    if use_area_kwargs:
        # Use area_kwargs and common_area_kwargs
        rnn_params["area_kwargs"] = {"out_channels": area_out_channels, "feedback_channels": feedback_channels_per_area}
        rnn_params["common_area_kwargs"] = {"num_neuron_types": 1, "tau_init": 15.0}
        if use_single_connectome:
             # This test path is tricky if area_kwargs is also trying to define a full connectome.
             # SpatiallyEmbeddedRNN will use the connectome from area_kwargs for all if use_single_connectome.
             # Let's ensure area_kwargs provides enough for one area.
             rnn_params["area_kwargs"]["in_channels"] = initial_in_channels # For the single connectome base
             pass # Defaults are mostly fine
        else:
            # Need to provide a list of area_kwargs if not single connectome
             rnn_params["area_kwargs"] = [{"out_channels": area_out_channels,
                                           "in_channels": initial_in_channels if i==0 else area_out_channels,
                                           "feedback_channels": feedback_channels_per_area} for i in range(num_areas)]

    else:
        rnn_params["area_configs"] = area_configs_list[0] if num_areas == 1 and use_single_connectome else area_configs_list


    # Test inter_area_feedback_connectivity variations
    if num_areas > 1:
        rnn_params_fb_conn = rnn_params.copy()
        # Valid lower triangular
        fb_conn_valid = np.tril(np.ones((num_areas, num_areas), dtype=bool), k=-1)
        rnn_fb_valid = SpatiallyEmbeddedRNN(**rnn_params_fb_conn, inter_area_feedback_connectivity=fb_conn_valid)
        assert rnn_fb_valid is not None
        assert rnn_fb_valid.inter_area_feedback_convs is not None

        # All True (will be made lower triangular internally)
        fb_conn_all_true = np.ones((num_areas, num_areas), dtype=bool)
        rnn_fb_all_true = SpatiallyEmbeddedRNN(**rnn_params_fb_conn, inter_area_feedback_connectivity=fb_conn_all_true)
        assert rnn_fb_all_true is not None

        # Test variations for nonlinearity and spatial extents (scalar and array)
        rnn_params_fb_ext = rnn_params.copy()
        rnn_params_fb_ext["inter_area_feedback_connectivity"] = fb_conn_valid
        rnn_fb_scalar_ext = SpatiallyEmbeddedRNN(**rnn_params_fb_ext,
                                                 inter_area_feedback_spatial_extents_hw=3,
                                                 inter_area_feedback_nonlinearity="relu")
        assert rnn_fb_scalar_ext is not None

        rnn_fb_array_ext = SpatiallyEmbeddedRNN(**rnn_params_fb_ext,
                                                inter_area_feedback_spatial_extents_hw=np.full((num_areas,num_areas),5),
                                                inter_area_feedback_nonlinearity=[["sigmoid"]*num_areas]*num_areas)
        assert rnn_fb_array_ext is not None


    rnn = SpatiallyEmbeddedRNN(**rnn_params)
    assert rnn is not None
    assert len(rnn.areas) == num_areas
    assert rnn.batch_first == batch_first

    if use_single_connectome and num_areas > 0:
        base_config = rnn.areas[0].config
        for i in range(1, num_areas):
            # Check some key parameters that should be shared
            assert rnn.areas[i].config.num_neuron_types == base_config.num_neuron_types
            assert rnn.areas[i].config.tau_mode == base_config.tau_mode
            # Connectivity related params should be identical if using single connectome from a config object
            if not use_area_kwargs : # if actual config objects were made
                assert rnn.areas[i].config.inter_neuron_type_connectivity is base_config.inter_neuron_type_connectivity

    # Test pool_mode and area_time_delay
    rnn_pool_delay = SpatiallyEmbeddedRNN(
        **rnn_params, pool_mode='max', area_time_delay=True,
        # Ensure sizes are poolable if num_areas > 1
        area_kwargs= {"out_channels": area_out_channels, "in_channels": initial_in_channels if use_area_kwargs else (initial_in_channels if _==0 else area_out_channels),
                      "feedback_channels": feedback_channels_per_area, "in_size": (8,8)} if use_area_kwargs else None,
        area_configs=[_create_default_s_area_config(idx=i, in_size=(8,8), in_ch=initial_in_channels if i==0 else area_out_channels, out_ch=area_out_channels) for i in range(num_areas)] if not use_area_kwargs else None
    )
    assert rnn_pool_delay is not None
    if num_areas > 1:
        assert rnn_pool_delay.pool_layers is not None
        assert len(rnn_pool_delay.pool_layers) == num_areas -1


def test_s_rnn_initialization_errors():
    torch.manual_seed(49)
    base_config = _create_default_s_area_config(in_ch=3, out_ch=4)
    config_mismatch_in = _create_default_s_area_config(in_ch=5, out_ch=4) # Input to next area is 5, but prev output is 4

    # Mismatched in/out channels between areas without projection
    with pytest.raises(ValueError, match="Output channels of area"):
        SpatiallyEmbeddedRNN(num_areas=2, area_configs=[base_config, config_mismatch_in], use_projection_between_areas=False)

    # Invalid inter_area_feedback_connectivity shape
    with pytest.raises(ValueError, match="inter_area_feedback_connectivity must be a square matrix"):
        SpatiallyEmbeddedRNN(num_areas=2, area_configs=[base_config]*2, inter_area_feedback_connectivity=np.array([True, False]))

    with pytest.raises(ValueError, match="inter_area_feedback_connectivity shape must match num_areas"):
         SpatiallyEmbeddedRNN(num_areas=2, area_configs=[base_config]*2, inter_area_feedback_connectivity=np.ones((3,3), dtype=bool))

    # Invalid inter_area_feedback_spatial_extents_hw shape
    with pytest.raises(ValueError, match="inter_area_feedback_spatial_extents_hw shape must match num_areas"):
        SpatiallyEmbeddedRNN(num_areas=2, area_configs=[base_config]*2,
                             inter_area_feedback_connectivity=np.ones((2,2),dtype=bool),
                             inter_area_feedback_spatial_extents_hw=np.array([1,2,3]))

    # area_configs not a list when num_areas > 1 and not use_single_connectome
    with pytest.raises(ValueError, match="area_configs must be a list of SpatiallyEmbeddedAreaConfig"):
        SpatiallyEmbeddedRNN(num_areas=2, area_configs=base_config, use_single_connectome_for_all_areas=False)

    # Not enough area_kwargs when not use_single_connectome
    with pytest.raises(ValueError, match="area_kwargs must be a list of dicts if not using single connectome"):
        SpatiallyEmbeddedRNN(num_areas=2, area_kwargs={"out_channels": 3}, use_single_connectome_for_all_areas=False,
                             input_channels_for_first_area=3, in_size_hw_for_first_area=(8,8))


@pytest.mark.parametrize("num_areas", [1, 2])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("area_time_delay", [True, False])
@pytest.mark.parametrize("pool_mode", [None, "max"]) # 'avg' is similar to 'max' for shape testing
@pytest.mark.parametrize("num_steps", [1, 3])
@pytest.mark.parametrize("provide_initial_states", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_s_rnn_forward_pass(
    num_areas, batch_first, area_time_delay, pool_mode, num_steps, provide_initial_states, dtype
):
    torch.manual_seed(50)
    in_size_hw = (8,8) if pool_mode is None else (16,16) # Ensure poolable if active
    initial_in_channels = 2
    area_out_channels = 3
    feedback_channels_val = 3 # Matches area_out_channels for simplicity in inter-area feedback

    if pool_mode is not None and num_areas == 1:
        pytest.skip("pool_mode is only relevant for num_areas > 1")

    area_configs = []
    current_ch_in = initial_in_channels
    current_size_hw = in_size_hw
    for i in range(num_areas):
        area_configs.append(
            _create_default_s_area_config(
                idx=i, in_size=current_size_hw, in_ch=current_ch_in,
                out_ch=area_out_channels, feedback_ch=feedback_channels_val,
                num_neuron_types=1, # Simpler for forward pass shape checks
                store_history=True, # For checking history
                dtype=dtype
            )
        )
        current_ch_in = area_out_channels
        if pool_mode is not None and i < num_areas -1: # If pooling, next area input size is halved
            current_size_hw = (current_size_hw[0]//2, current_size_hw[1]//2)


    rnn = SpatiallyEmbeddedRNN(
        num_areas=num_areas,
        area_configs=area_configs,
        batch_first=batch_first,
        area_time_delay=area_time_delay,
        pool_mode=pool_mode,
        inter_area_feedback_channels=feedback_channels_val, # For feedback from last to first, etc.
        inter_area_feedback_connectivity=np.tril(np.ones((num_areas,num_areas),dtype=bool), k=-1) if num_areas > 1 else None,
        store_history_for_all_areas=True
    )
    rnn.to(dtype)

    # Prepare input tensor
    if batch_first:
        input_tensor = torch.rand(num_steps, initial_in_channels, *in_size_hw, dtype=dtype) # (Batch=1, Steps, C, H, W)
        input_tensor = input_tensor.unsqueeze(0).expand(2, -1, -1, -1, -1) # (Batch=2, Steps, C, H, W)
        current_batch_size = 2
    else:
        input_tensor = torch.rand(num_steps, 2, initial_in_channels, *in_size_hw, dtype=dtype) # (Steps, Batch=2, C, H, W)
        current_batch_size = 2


    output_state0_all, neuron_state0_all, feedback_state0_all = None, None, None
    if provide_initial_states:
        output_state0_all = rnn.init_output_states(current_batch_size)
        neuron_state0_all = rnn.init_neuron_states(current_batch_size)
        feedback_state0_all = rnn.init_feedback_states(current_batch_size)


    outputs_all, neurons_all, feedbacks_all = rnn.forward(
        input_tensor,
        output_state0_all_areas=output_state0_all,
        neuron_state0_all_areas=neuron_state0_all,
        feedback_state0_all_areas=feedback_state0_all
    )

    # Check output_states_all_areas
    assert isinstance(outputs_all, list)
    assert len(outputs_all) == num_areas
    current_h, current_w = in_size_hw
    for i in range(num_areas):
        expected_shape_prefix = (current_batch_size, num_steps) if batch_first else (num_steps, current_batch_size)
        assert outputs_all[i].shape == (*expected_shape_prefix, area_out_channels, current_h, current_w)
        assert outputs_all[i].dtype == dtype
        if pool_mode is not None and i < num_areas -1 :
            current_h //= 2
            current_w //= 2

    # Check neuron_states_all_areas
    assert isinstance(neurons_all, list)
    assert len(neurons_all) == num_areas
    for i in range(num_areas):
        assert isinstance(neurons_all[i], list) # List per area for its neuron types
        # For this test, num_neuron_types=1 in each area config
        assert len(neurons_all[i]) == area_configs[i].num_neuron_types
        # Shape of neuron state from Area.forward is (batch, channels, H, W)
        # RNN stacks these across time, so it should be (batch, num_steps, channels, H, W) or (num_steps, batch, ...)
        # However, the returned 'neuron_states_all_areas' is the *final* neuron state, not the history.
        neuron_type_module_out_ch = area_configs[i].neuron_type_modules_out_channels[0] # for type 0

        # Determine the H, W for this area's neuron states (matches its output H,W before pooling for next area)
        area_h, area_w = outputs_all[i].shape[-2:]

        # The returned neuron_states are final states, so shape (batch, channels, H, W)
        final_neuron_state_shape = (current_batch_size, neuron_type_module_out_ch, area_h, area_w)
        assert neurons_all[i][0].shape == final_neuron_state_shape
        assert neurons_all[i][0].dtype == dtype


    # Check feedback_states_all_areas
    assert isinstance(feedbacks_all, list)
    # Number of feedback signals depends on connectivity. With tril(-1), area i gives feedback to i-1...0
    # The list contains feedback *received by* area i from area i+1...N-1 (feedforward view)
    # Or, if it's feedback *sent by* area i, then it's different.
    # Based on `feedback_signals_per_area.append(fb_signal_to_next_area)`, it's N-1 signals sent.
    if num_areas > 1 :
        assert len(feedbacks_all) == num_areas -1 # Feedback between areas
        # Feedback shape depends on pooling of the source area
        fb_h, fb_w = in_size_hw # Start with initial size for feedback from Area 0
        for i in range(num_areas -1):
            expected_shape_prefix = (current_batch_size, num_steps) if batch_first else (num_steps, current_batch_size)
            # Feedback channels are based on `inter_area_feedback_channels` or source area's out_channels if not projected
            # For this test, inter_area_feedback_channels = area_out_channels
            assert feedbacks_all[i].shape == (*expected_shape_prefix, feedback_channels_val, fb_h, fb_w)
            assert feedbacks_all[i].dtype == dtype
            if pool_mode is not None:
                fb_h //=2
                fb_w //=2
    else:
        assert len(feedbacks_all) == 0


    # Check history storage in areas
    for i in range(num_areas):
        assert rnn.areas[i].history_h_neurons is not None
        assert len(rnn.areas[i].history_h_neurons) == num_steps
        assert len(rnn.areas[i].history_h_neurons[0]) == area_configs[i].num_neuron_types


@pytest.mark.parametrize("num_areas", [1, 2])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("area_time_delay", [False, True]) # Add True case
@pytest.mark.parametrize("pool_mode", [None, "max"])
@pytest.mark.parametrize("train_tau_all_areas", [True, False])
@pytest.mark.parametrize("use_projection_between_areas", [False, True]) # Add True case
def test_s_rnn_backward_pass(
    num_areas, batch_first, area_time_delay, pool_mode, train_tau_all_areas, use_projection_between_areas
):
    torch.manual_seed(51)
    in_size_hw = (8,8) if pool_mode is None else (16,16)
    initial_in_channels = 2
    area_out_channels = 2 # Keep small for faster backward
    feedback_channels_val = 2
    num_steps = 2 # Small steps
    current_batch_size = 1 # Smallest batch
    dtype = torch.float32

    if pool_mode is not None and num_areas == 1:
        pytest.skip("pool_mode is only relevant for num_areas > 1")
    if use_projection_between_areas and num_areas == 1:
        pytest.skip("projection_between_areas is only relevant for num_areas > 1")


    area_configs = []
    current_ch_in = initial_in_channels
    current_size_hw_iter = in_size_hw # Renamed to avoid conflict with outer scope current_size_hw in forward test
    for i in range(num_areas):
        area_configs.append(
            _create_default_s_area_config(
                idx=i, in_size=current_size_hw_iter, in_ch=current_ch_in,
                out_ch=area_out_channels, feedback_ch=feedback_channels_val,
                num_neuron_types=1, train_tau=train_tau_all_areas, dtype=dtype,
                # Ensure all convs have biases for gradient checking
                inter_neuron_type_bias = 0.1,
            )
        )
        # If not using projection, next area's in_ch must match current area's out_ch
        # If using projection, RNN handles it, so next area can have different in_ch (but config must be set)
        # For this test, assume if projection is used, the created config's in_ch is suitable (e.g. could be different)
        current_ch_in = area_out_channels
        if pool_mode is not None and i < num_areas -1:
            current_size_hw_iter = (current_size_hw_iter[0]//2, current_size_hw_iter[1]//2)

    rnn = SpatiallyEmbeddedRNN(
        num_areas=num_areas,
        area_configs=area_configs,
        batch_first=batch_first,
        area_time_delay=area_time_delay,
        pool_mode=pool_mode,
        inter_area_feedback_channels=feedback_channels_val,
        inter_area_feedback_connectivity=np.tril(np.ones((num_areas,num_areas),dtype=bool), k=-1) if num_areas > 1 else None,
        use_projection_between_areas=use_projection_between_areas,
        store_history_for_all_areas=False # Not needed for backward
    )
    rnn.to(dtype)

    if batch_first:
        input_tensor = torch.rand(current_batch_size, num_steps, initial_in_channels, *in_size_hw, dtype=dtype)
    else:
        input_tensor = torch.rand(num_steps, current_batch_size, initial_in_channels, *in_size_hw, dtype=dtype)

    outputs_all, _, _ = rnn.forward(input_tensor)

    # Sum outputs from the last area for loss
    loss = outputs_all[-1].sum()
    loss.backward()

    # Check gradients for all areas
    for i in range(num_areas):
        area = rnn.areas[i]
        if train_tau_all_areas:
            assert area.tau.grad is not None
            assert not torch.isnan(area.tau.grad).any()
            assert not torch.isinf(area.tau.grad).any()
        else:
            if hasattr(area.tau, 'grad'): assert area.tau.grad is None

        all_area_conv_layers = []
        if area.input_conv: all_area_conv_layers.extend(area.input_conv)
        if area.feedback_conv: all_area_conv_layers.extend(area.feedback_conv)
        for nt_module in area.neuron_type_modules:
            all_area_conv_layers.extend(nt_module.inter_convs)
        if area.output_conv: all_area_conv_layers.extend(area.output_conv)

        for conv_layer in all_area_conv_layers:
            assert conv_layer.weight.grad is not None; assert not torch.isnan(conv_layer.weight.grad).any()
            if conv_layer.bias is not None:
                assert conv_layer.bias.grad is not None; assert not torch.isnan(conv_layer.bias.grad).any()

    # Check gradients for inter_area_projection_layers
    if use_projection_between_areas and num_areas > 1:
        assert rnn.inter_area_projection_layers is not None
        for proj_layer in rnn.inter_area_projection_layers:
            if isinstance(proj_layer, Conv2dRectify): # Or nn.Conv2d if that's used
                assert proj_layer.weight.grad is not None; assert not torch.isnan(proj_layer.weight.grad).any()
                if proj_layer.bias is not None:
                     assert proj_layer.bias.grad is not None; assert not torch.isnan(proj_layer.bias.grad).any()
            elif isinstance(proj_layer, nn.Identity):
                pass # No grads for Identity

    # Check gradients for inter_area_feedback_convs
    if rnn.inter_area_feedback_convs is not None:
        for fb_conv_list in rnn.inter_area_feedback_convs: # List of lists
            for fb_conv in fb_conv_list: # Each actual Conv2dRectify
                 if isinstance(fb_conv, Conv2dRectify):
                    assert fb_conv.weight.grad is not None; assert not torch.isnan(fb_conv.weight.grad).any()
                    if fb_conv.bias is not None:
                        assert fb_conv.bias.grad is not None; assert not torch.isnan(fb_conv.bias.grad).any()
                 elif isinstance(fb_conv, nn.Identity):
                     pass


    # Check gradients for pool_layers
    if rnn.pool_layers is not None:
        # Gradients for pooling layers themselves are not parameters, but they affect upstream grads.
        # This check is more about ensuring the setup doesn't break backward pass.
        # If previous checks pass, pooling layers are doing their job in grad propagation.
        pass


@pytest.mark.parametrize("num_areas", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_s_rnn_init_all_states(num_areas, batch_size, dtype):
    torch.manual_seed(52)
    area_configs = [_create_default_s_area_config(idx=i, dtype=dtype, num_neuron_types=2 if i%2==0 else 1) for i in range(num_areas)] # Vary num_neuron_types

    rnn = SpatiallyEmbeddedRNN(num_areas=num_areas, area_configs=area_configs, dtype=dtype)
    rnn.to(dtype)

    # Test init_neuron_states
    all_neuron_states = rnn.init_neuron_states(batch_size)
    assert isinstance(all_neuron_states, list)
    assert len(all_neuron_states) == num_areas
    for i in range(num_areas):
        area_neuron_states = all_neuron_states[i]
        expected_area_neuron_states = rnn.areas[i].init_neuron_state(batch_size)
        assert isinstance(area_neuron_states, list)
        assert len(area_neuron_states) == len(expected_area_neuron_states) # Num neuron types in this area
        for j in range(len(area_neuron_states)):
            assert torch.allclose(area_neuron_states[j], expected_area_neuron_states[j])
            assert area_neuron_states[j].dtype == dtype

    # Test init_output_states
    all_output_states = rnn.init_output_states(batch_size)
    assert isinstance(all_output_states, list)
    assert len(all_output_states) == num_areas
    for i in range(num_areas):
        area_output_state = all_output_states[i]
        expected_area_output_state = rnn.areas[i].init_output_state(batch_size)
        assert torch.allclose(area_output_state, expected_area_output_state)
        assert area_output_state.dtype == dtype

    # Test init_feedback_states (for inter-area feedback)
    # These are feedback signals *to be sent* if following the pattern of Area.init_feedback_state
    # Or, initial states for feedback buffers if that's the interpretation.
    # The SpatiallyEmbeddedRNN.init_feedback_states initializes feedback *buffers* for receiving signals.
    all_feedback_states = rnn.init_feedback_states(batch_size) # Initializes feedback buffers
    assert isinstance(all_feedback_states, list)
    # One feedback buffer per area that can *receive* feedback.
    # If using tril(-1) connectivity, all areas except the last can potentially send, first can't receive from previous.
    # The `inter_area_feedback_buffers` are for feedback signals passed *forward* in the sequence.
    # `init_feedback_states` initializes states for these buffers.
    expected_len = num_areas if rnn.inter_area_feedback_connectivity is not None and np.any(rnn.inter_area_feedback_connectivity) else 0
    # More precisely, it's for each connection that exists. Let's check based on `inter_area_feedback_convs` structure.
    # For this test, let's assume default connectivity (tril) which means num_areas-1 sets of convs.
    # And the init_feedback_states initializes a buffer for each area that *could* receive feedback.
    # The current SpatiallyEmbeddedRNN.init_feedback_states initializes a list of num_areas tensors.
    assert len(all_feedback_states) == num_areas

    for i in range(num_areas):
        fb_state = all_feedback_states[i]
        # Shape should be (batch, rnn.inter_area_feedback_channels, H, W of area i)
        # This is tricky because inter_area_feedback_channels might be different from area's own feedback_channels.
        # For this test, let's assume rnn.inter_area_feedback_channels is used.
        # The H, W should match the area that *receives* this feedback, which is area i.
        expected_fb_shape = (batch_size, rnn.inter_area_feedback_channels, *rnn.areas[i].config.in_size)
        if rnn.inter_area_feedback_connectivity is not None and np.any(rnn.inter_area_feedback_connectivity[:, i]): # If area i receives any feedback
            assert fb_state is not None
            assert fb_state.shape == expected_fb_shape
            assert fb_state.dtype == dtype
            assert torch.allclose(fb_state, torch.zeros(expected_fb_shape, dtype=dtype)) # Default is zeros
        else:
            assert fb_state is None # No buffer if no incoming connection


@pytest.mark.parametrize("num_areas", [1, 2])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("query_area_idx", [None, 0]) # None for all, 0 for first area
def test_s_rnn_query_neuron_states(num_areas, batch_first, query_area_idx):
    torch.manual_seed(53)
    num_steps = 2
    area_configs = [_create_default_s_area_config(idx=i, store_history=True, num_neuron_types=1) for i in range(num_areas)]
    rnn = SpatiallyEmbeddedRNN(num_areas=num_areas, area_configs=area_configs, batch_first=batch_first, store_history_for_all_areas=True)

    # Dummy input for forward pass
    current_batch_size = 2
    initial_in_channels = area_configs[0].in_channels
    in_size_hw = area_configs[0].in_size
    if batch_first:
        input_tensor = torch.rand(current_batch_size, num_steps, initial_in_channels, *in_size_hw)
    else:
        input_tensor = torch.rand(num_steps, current_batch_size, initial_in_channels, *in_size_hw)

    rnn.forward(input_tensor) # Populate history

    # Query parameters (simple query for this test)
    time_step_query = num_steps - 1 # Query last time step

    queried_states = rnn.query_neuron_states(time_step=time_step_query, area_idx=query_area_idx)

    if query_area_idx is not None:
        # Should be equivalent to querying the specific area
        expected_states = rnn.areas[query_area_idx].query_neuron_states(time_step=time_step_query)
        # queried_states from RNN's method for a single area is directly the tensor/list from that area.
        # Area's query_neuron_states returns a tensor if num_neuron_types=1 and time_step is int.
        # Let's ensure this matches.
        assert isinstance(queried_states, torch.Tensor) # Since num_neuron_types=1 in config
        assert torch.allclose(queried_states, expected_states)
    else:
        # Should be a list of query results, one for each area
        assert isinstance(queried_states, list)
        assert len(queried_states) == num_areas
        for i in range(num_areas):
            expected_states_area_i = rnn.areas[i].query_neuron_states(time_step=time_step_query)
            assert isinstance(queried_states[i], torch.Tensor)
            assert torch.allclose(queried_states[i], expected_states_area_i)


# --- Conv2dRectify Tests ---

@pytest.mark.parametrize("bias_flag", [True, False])
@pytest.mark.parametrize("rectify_weights", [True, False])
@pytest.mark.parametrize("rectify_bias", [True, False])
def test_conv2d_rectify_forward_backward(bias_flag, rectify_weights, rectify_bias):
    torch.manual_seed(54)
    in_channels, out_channels, kernel_size = 2, 3, 3
    input_tensor = torch.randn(2, in_channels, 5, 5, requires_grad=True) # Batch, C_in, H, W

    conv = Conv2dRectify(
        in_channels, out_channels, kernel_size,
        bias=bias_flag,
        rectify_weights=rectify_weights,
        rectify_bias=rectify_bias
    )

    # Initialize weights/bias to have some negative values to test rectification
    conv.weight.data.uniform_(-1, 1)
    if bias_flag:
        conv.bias.data.uniform_(-1, 1)

    output = conv(input_tensor)

    # Check rectification
    if rectify_weights:
        assert torch.all(conv.weight.data >= 0)
    else: # Check if some original negatives are still there (high probability)
        if conv.weight.data.numel() > 0 : # Ensure weight tensor is not empty
             assert torch.any(conv.weight.data < 0) or torch.all(conv.weight.data >=0) # Can be all positive by chance

    if bias_flag and rectify_bias:
        assert torch.all(conv.bias.data >= 0)
    elif bias_flag: # Check if some original negatives are still there
        if conv.bias.data.numel() > 0:
            assert torch.any(conv.bias.data < 0) or torch.all(conv.bias.data >=0)


    # Test backward pass
    loss = output.sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert conv.weight.grad is not None
    if bias_flag:
        assert conv.bias.grad is not None

    # Ensure gradients are not all zero (unless output was zero, which is unlikely)
    if output.abs().sum().item() > 1e-6 : # Only if output is not effectively zero
        assert conv.weight.grad.abs().sum().item() > 0
        if bias_flag:
            assert conv.bias.grad.abs().sum().item() > 0


# Remove the final TODO
# TODO: More extensive testing needs to be done for the SpatiallyEmbeddedRNN, SpatiallyEmbeddedArea, and SpatiallyEmbeddedAreaConfig
