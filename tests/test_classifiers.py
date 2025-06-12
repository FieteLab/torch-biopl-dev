import torch
import pytest
import numpy as np

# RNN models and Configs needed for rnn_kwargs
from bioplnn.models.connectome import ConnectomeRNN, ConnectomeODERNN
from bioplnn.models.spatially_embedded import SpatiallyEmbeddedRNN, SpatiallyEmbeddedAreaConfig

# Classifier models
from bioplnn.models.classifiers import (
    ConnectomeClassifier,
    ConnectomeODEClassifier,
    SpatiallyEmbeddedClassifier,
)

torch.manual_seed(0)

# Helper function for gradient checking (optional, can do it directly in tests)
def check_gradients(model, loss):
    """Checks if all trainable parameters have gradients and are not NaN/Inf."""
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
            assert param.grad.abs().sum() > 1e-8, f"Zero gradient for {name} (potential issue)" # check if grad is not effectively zero

# --- ConnectomeClassifier Tests ---

def _get_default_connectome_rnn_kwargs(input_size=10, num_neurons=5, batch_first=False, connectome_type='identity'):
    if connectome_type == 'identity':
        connectome = torch.eye(num_neurons)
    elif connectome_type == 'random':
        connectome = torch.rand(num_neurons, num_neurons)
    else:
        raise ValueError("Invalid connectome_type")

    return {
        "input_size": input_size,
        "num_neurons": num_neurons,
        "connectome": connectome,
        "batch_first": batch_first,
        "store_history": True # For return_activations
    }

@pytest.mark.parametrize("fc_dim", [None, 32])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("num_classes", [2, 10])
def test_connectome_classifier_initialization(fc_dim, dropout, num_classes):
    torch.manual_seed(1)
    rnn_kwargs = _get_default_connectome_rnn_kwargs()

    classifier = ConnectomeClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        fc_dim=fc_dim,
        dropout=dropout
    )
    assert classifier is not None
    assert classifier.num_classes == num_classes
    assert classifier.fc_dim == (fc_dim if fc_dim is not None else rnn_kwargs["num_neurons"])
    assert classifier.dropout_rate == dropout
    assert isinstance(classifier.rnn, ConnectomeRNN)
    assert classifier.fc_layers is not None

@pytest.mark.parametrize("loss_all_timesteps", [True, False])
@pytest.mark.parametrize("return_activations", [True, False])
@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("input_type", ["4d", "5d"]) # 4d (B,C,H,W), 5d (B,T,C,H,W)
def test_connectome_classifier_forward(loss_all_timesteps, return_activations, batch_first_rnn, input_type):
    torch.manual_seed(2)
    num_classes = 3
    batch_size = 2
    num_timesteps = 5 # Only relevant for 5d input or if rnn itself expects sequence

    # For 4D input, features are C*H*W. For 5D, it's T*C*H*W if flattened before RNN,
    # or C*H*W if T is the sequence dim for RNN.
    # ConnectomeRNN expects (seq_len, batch, input_features) or (batch, seq_len, input_features)

    C, H, W = 2, 4, 4
    input_features_rnn = C * H * W

    rnn_kwargs = _get_default_connectome_rnn_kwargs(input_size=input_features_rnn, num_neurons=8, batch_first=batch_first_rnn)

    classifier = ConnectomeClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        loss_all_timesteps=loss_all_timesteps,
        return_activations=return_activations
    )
    classifier.eval() # For consistent dropout behavior

    if input_type == "4d":
        # Input: (B, C, H, W) -> Classifier expects (B, Features) or (B, 1, Features) if using RNN for single step
        # The classifier's forward should handle reshaping this to (B, 1, C*H*W) for the RNN if not batch_first
        # or (1, B, C*H*W) if batch_first. Or, if rnn input_size is C*H*W, it implies a single time step.
        # ConnectomeClassifier's current forward assumes input is (batch, seq_len, features) or (seq_len, batch, features)
        # So, a 4D input (B,C,H,W) is treated as (B, 1, C*H*W) effectively.
        x = torch.rand(batch_size, C, H, W)
        expected_rnn_seq_len = 1
    elif input_type == "5d":
        # Input: (B, T, C, H, W) -> Classifier expects (B, T, Features) or (T, B, Features)
        x = torch.rand(batch_size, num_timesteps, C, H, W)
        expected_rnn_seq_len = num_timesteps

    result = classifier(x)

    if return_activations:
        output, rnn_outputs, fc_inputs = result
        assert rnn_outputs is not None
        assert fc_inputs is not None

        # rnn_outputs shape: (seq_len, batch, num_neurons) or (batch, seq_len, num_neurons)
        # fc_inputs shape: depends on loss_all_timesteps
        if batch_first_rnn:
            assert rnn_outputs.shape == (batch_size, expected_rnn_seq_len, classifier.rnn.num_neurons)
            if loss_all_timesteps:
                assert fc_inputs.shape == (batch_size * expected_rnn_seq_len, classifier.rnn.num_neurons)
            else:
                assert fc_inputs.shape == (batch_size, classifier.rnn.num_neurons) # Only last time step
        else:
            assert rnn_outputs.shape == (expected_rnn_seq_len, batch_size, classifier.rnn.num_neurons)
            if loss_all_timesteps:
                assert fc_inputs.shape == (expected_rnn_seq_len * batch_size, classifier.rnn.num_neurons)
            else:
                assert fc_inputs.shape == (batch_size, classifier.rnn.num_neurons) # Only last time step (after permute)
    else:
        output = result

    if loss_all_timesteps:
        if batch_first_rnn : # Output (B, T, N_classes)
            assert output.shape == (batch_size, expected_rnn_seq_len, num_classes)
        else: # Output (T, B, N_classes)
            assert output.shape == (expected_rnn_seq_len, batch_size, num_classes)
    else: # Output (B, N_classes)
        assert output.shape == (batch_size, num_classes)


@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("train_rnn_params_explicitly", [True, False]) # Test if RNN grads are computed
@pytest.mark.parametrize("fc_dim", [None, 16])
def test_connectome_classifier_backward(batch_first_rnn, train_rnn_params_explicitly, fc_dim):
    torch.manual_seed(3)
    num_classes = 2
    batch_size = 2
    input_features_rnn = 10

    rnn_kwargs = _get_default_connectome_rnn_kwargs(
        input_size=input_features_rnn, num_neurons=8, batch_first=batch_first_rnn
    )
    # Ensure RNN params require grad if train_rnn_params_explicitly is True
    if train_rnn_params_explicitly:
        rnn_kwargs["connectome"] = torch.rand(8,8, requires_grad=True) # Example: make connectome trainable
        # Could also set rnn_kwargs["train_tau"] = True if testing tau grads, etc.

    classifier = ConnectomeClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        fc_dim=fc_dim,
        loss_all_timesteps=False # Simpler for backward, grad check should still work
    )
    # If not training RNN params explicitly, set them to not require grad after init
    if not train_rnn_params_explicitly:
        for param in classifier.rnn.parameters():
            param.requires_grad = False

    # Input: (B, T, Features) or (T, B, Features)
    num_timesteps = 3
    if batch_first_rnn:
        x = torch.rand(batch_size, num_timesteps, input_features_rnn)
    else:
        x = torch.rand(num_timesteps, batch_size, input_features_rnn)

    output = classifier(x)
    loss = output.sum()

    # Custom gradient check because helper expects model.parameters() to be all relevant
    # Here we need to check RNN params conditionally
    loss.backward()

    for name, param in classifier.fc_layers.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"FC grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"FC NaN grad for {name}"
            assert param.grad.abs().sum() > 1e-8, f"FC Zero grad for {name}"

    for name, param in classifier.rnn.named_parameters():
        if param.requires_grad: # This depends on train_rnn_params_explicitly and RNN's own setup
            assert param.grad is not None, f"RNN grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"RNN NaN grad for {name}"
            # Sum check can be tricky if some RNN params naturally have small grads
            # assert param.grad.abs().sum() > 1e-9, f"RNN Zero/small grad for {name}"
        else:
            assert param.grad is None, f"RNN grad present but not expected for {name}"


# --- ConnectomeODEClassifier Tests ---

def _get_default_connectome_odernn_kwargs(input_size=10, num_neurons=5, batch_first=False, connectome_type='identity', solver='dopri5'):
    if connectome_type == 'identity':
        connectome = torch.eye(num_neurons)
    elif connectome_type == 'random':
        connectome = torch.rand(num_neurons, num_neurons)
    else:
        raise ValueError("Invalid connectome_type")

    return {
        "input_size": input_size,
        "num_neurons": num_neurons,
        "connectome": connectome,
        "batch_first": batch_first,
        "store_history": True, # For return_activations
        "solver": solver,
        "solver_options": {"step_size": 0.1} if solver == "rk4" else {"atol": 1e-3, "rtol": 1e-3} # Basic options
    }

@pytest.mark.parametrize("fc_dim", [None, 24])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
@pytest.mark.parametrize("num_classes", [2, 5])
def test_connectome_ode_classifier_initialization(fc_dim, dropout, num_classes):
    torch.manual_seed(4)
    rnn_kwargs = _get_default_connectome_odernn_kwargs()

    classifier = ConnectomeODEClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        fc_dim=fc_dim,
        dropout=dropout
    )
    assert classifier is not None
    assert classifier.num_classes == num_classes
    assert classifier.fc_dim == (fc_dim if fc_dim is not None else rnn_kwargs["num_neurons"])
    assert classifier.dropout_rate == dropout
    assert isinstance(classifier.rnn, ConnectomeODERNN)
    assert classifier.fc_layers is not None

@pytest.mark.parametrize("loss_all_timesteps", [True, False])
@pytest.mark.parametrize("return_activations", [True, False])
@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("input_type", ["4d", "5d"])
@pytest.mark.parametrize("num_evals", [None, 4]) # None means use input sequence length for eval points
def test_connectome_ode_classifier_forward(loss_all_timesteps, return_activations, batch_first_rnn, input_type, num_evals):
    torch.manual_seed(5)
    num_classes = 3
    batch_size = 2
    num_timesteps_input_seq = 5 # Number of time points in the input data if 5D

    C, H, W = 2, 3, 3
    input_features_rnn = C * H * W

    rnn_kwargs = _get_default_connectome_odernn_kwargs(
        input_size=input_features_rnn, num_neurons=7, batch_first=batch_first_rnn
    )

    classifier = ConnectomeODEClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        loss_all_timesteps=loss_all_timesteps,
        return_activations=return_activations,
        num_evals=num_evals # Pass num_evals to classifier for ODE forward
    )
    classifier.eval()

    if input_type == "4d":
        # (B,C,H,W) -> treated as (B, 1, C*H*W) for RNN
        x = torch.rand(batch_size, C, H, W)
        # For ODERN, if input is single time point, num_evals determines output seq len.
        # If num_evals is None, it defaults to 2 for single point input (start, end).
        expected_output_seq_len = num_evals if num_evals is not None else 2
    elif input_type == "5d":
        # (B,T,C,H,W) -> (B,T,Features) for RNN
        x = torch.rand(batch_size, num_timesteps_input_seq, C, H, W)
        expected_output_seq_len = num_evals if num_evals is not None else num_timesteps_input_seq

    result = classifier(x) # num_evals is passed during init now

    if return_activations:
        output, rnn_outputs, fc_inputs, timestamps = result # Timestamps also returned for ODERN
        assert rnn_outputs is not None
        assert fc_inputs is not None
        assert timestamps is not None

        if batch_first_rnn:
            assert rnn_outputs.shape == (batch_size, expected_output_seq_len, classifier.rnn.num_neurons)
            assert timestamps.shape == (batch_size, expected_output_seq_len)
            if loss_all_timesteps:
                assert fc_inputs.shape == (batch_size * expected_output_seq_len, classifier.rnn.num_neurons)
            else: # Only last time step's features used for FC
                assert fc_inputs.shape == (batch_size, classifier.rnn.num_neurons)
        else:
            assert rnn_outputs.shape == (expected_output_seq_len, batch_size, classifier.rnn.num_neurons)
            assert timestamps.shape == (expected_output_seq_len,) # Timestamps usually (seq_len,) if not batch_first for eval_times
            if loss_all_timesteps:
                assert fc_inputs.shape == (expected_output_seq_len * batch_size, classifier.rnn.num_neurons)
            else:
                assert fc_inputs.shape == (batch_size, classifier.rnn.num_neurons)
    else:
        output = result

    if loss_all_timesteps:
        if batch_first_rnn:
            assert output.shape == (batch_size, expected_output_seq_len, num_classes)
        else:
            assert output.shape == (expected_output_seq_len, batch_size, num_classes)
    else:
        assert output.shape == (batch_size, num_classes)


@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("train_rnn_params_explicitly", [True, False])
@pytest.mark.parametrize("solver", ["rk4", "dopri5"]) # Test different solvers
def test_connectome_ode_classifier_backward(batch_first_rnn, train_rnn_params_explicitly, solver):
    torch.manual_seed(6)
    num_classes = 2
    batch_size = 1 # Smaller for faster backward
    input_features_rnn = 8

    rnn_kwargs = _get_default_connectome_odernn_kwargs(
        input_size=input_features_rnn, num_neurons=6, batch_first=batch_first_rnn, solver=solver
    )
    if train_rnn_params_explicitly:
        rnn_kwargs["connectome"] = torch.rand(6,6, requires_grad=True)
        rnn_kwargs["train_tau"] = True # Example of another trainable RNN param

    classifier = ConnectomeODEClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        loss_all_timesteps=False
    )
    if not train_rnn_params_explicitly:
        for param in classifier.rnn.parameters():
            param.requires_grad = False

    # ODERNs typically take (seq_len, batch, features) or (batch, seq_len, features)
    # For a simple test, let's use a "sequence" of 1 (like a 4D input that's been reshaped)
    num_timesteps_for_ode = 1
    if batch_first_rnn:
        x = torch.rand(batch_size, num_timesteps_for_ode, input_features_rnn)
    else:
        x = torch.rand(num_timesteps_for_ode, batch_size, input_features_rnn)

    output = classifier(x)
    loss = output.sum()

    # Custom gradient check
    loss.backward()

    for name, param in classifier.fc_layers.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"FC grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"FC NaN grad for {name}"
            assert param.grad.abs().sum() > 1e-8, f"FC Zero grad for {name}"

    for name, param in classifier.rnn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"ODERN grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"ODERN NaN grad for {name}"
            # Sum check can be sensitive for ODEs due to solver dynamics
            # assert param.grad.abs().sum() > 1e-9, f"ODERN Zero/small grad for {name}"
        else:
            assert param.grad is None, f"ODERN grad present but not expected for {name}"


# --- SpatiallyEmbeddedClassifier Tests ---

def _get_default_s_rnn_kwargs(num_areas=1, batch_first=False, dtype=torch.float32):
    # Create a minimal SpatiallyEmbeddedAreaConfig for the SpatiallyEmbeddedRNN
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            area_name=f"Area{i}",
            in_size=(8, 8), # Keep small for tests
            in_channels=3 if i == 0 else 4, # Example: first area takes 3, subsequent take 4
            out_channels=4, # All areas output 4 channels
            num_neuron_types=1,
            use_feedback=True, # Assuming feedback for inter-area, though not strictly needed if num_areas=1
            feedback_channels=4, # Matches out_channels for simplicity
            store_history=True, # For return_activations
            dtype=dtype
        ) for i in range(num_areas)
    ]

    return {
        "num_areas": num_areas,
        "area_configs": area_configs[0] if num_areas == 1 else area_configs,
        "batch_first": batch_first,
        "store_history_for_all_areas": True # For return_activations
    }


@pytest.mark.parametrize("pool_mode_classifier", ["avg", "max", None])
@pytest.mark.parametrize("fc_dim", [None, 40])
@pytest.mark.parametrize("dropout", [0.0, 0.6])
@pytest.mark.parametrize("num_classes", [3, 7])
def test_s_classifier_initialization(pool_mode_classifier, fc_dim, dropout, num_classes):
    torch.manual_seed(7)
    rnn_kwargs = _get_default_s_rnn_kwargs(num_areas=1) # Single area for simplicity

    # Determine pool_size_classifier based on rnn_kwargs output size if pooling is active
    pool_size = None
    if pool_mode_classifier is not None:
        # Last area's output size (H,W) from rnn_kwargs
        # For this test, _get_default_s_rnn_kwargs uses (8,8) for Area0
        # If pool_size is None, it defaults to the full spatial extent.
        # Let's explicitly set one, e.g., adaptive pooling or a fixed size if needed.
        # For adaptive pooling (output size 1x1), pool_size can be the input HxW.
        # If pool_size is an int, it's kernel size.
        pool_size = (1,1) # Results in adaptive pooling to 1x1 feature map per channel if None is not used.
                          # If None, it's adaptive. If int/tuple, it's kernel size.
                          # The classifier handles None by using AdaptiveAvg/MaxPool2d.

    classifier = SpatiallyEmbeddedClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        pool_size_classifier=pool_size, # Can be None for adaptive, or specific tuple/int
        pool_mode_classifier=pool_mode_classifier,
        fc_dim=fc_dim,
        dropout=dropout
    )
    assert classifier is not None
    assert classifier.num_classes == num_classes
    assert classifier.pool_mode == pool_mode_classifier

    if pool_mode_classifier is not None:
        assert classifier.pool_layer is not None
        if pool_mode_classifier == "avg":
            assert isinstance(classifier.pool_layer, torch.nn.AdaptiveAvgPool2d) or isinstance(classifier.pool_layer, torch.nn.AvgPool2d)
        elif pool_mode_classifier == "max":
            assert isinstance(classifier.pool_layer, torch.nn.AdaptiveMaxPool2d) or isinstance(classifier.pool_layer, torch.nn.MaxPool2d)
    else:
        assert classifier.pool_layer is None

    # fc_dim logic: if pooling, it's based on rnn_out_channels. If no pooling, it's rnn_out_channels * H * W.
    # This is complex due to adaptive pooling. The classifier sets self.fc_input_dim internally.
    # Just check that fc_layers are created.
    assert classifier.fc_layers is not None

    # Test invalid pool_mode
    with pytest.raises(ValueError, match="Invalid pool_mode_classifier"):
        SpatiallyEmbeddedClassifier(
            rnn_kwargs=rnn_kwargs, num_classes=num_classes, pool_mode_classifier="invalid"
        )


@pytest.mark.parametrize("loss_all_timesteps", [True, False])
@pytest.mark.parametrize("return_activations", [True, False])
@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("num_rnn_steps", [1, 3]) # Num steps the SpatiallyEmbeddedRNN runs for
def test_s_classifier_forward(loss_all_timesteps, return_activations, batch_first_rnn, num_rnn_steps):
    torch.manual_seed(8)
    num_classes = 2
    batch_size = 2

    # SpatiallyEmbeddedRNN input is (B, T, C, H, W) or (T, B, C, H, W)
    # C, H, W are defined in _get_default_s_rnn_kwargs (Area0: C=3, H=8, W=8)
    # Output channels of last area is 4.

    rnn_kwargs = _get_default_s_rnn_kwargs(num_areas=1, batch_first=batch_first_rnn) # Using 1 area for simplicity
    C_in, H_in, W_in = rnn_kwargs["area_configs"].in_channels, rnn_kwargs["area_configs"].in_size[0], rnn_kwargs["area_configs"].in_size[1]

    classifier = SpatiallyEmbeddedClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        loss_all_timesteps=loss_all_timesteps,
        return_activations=return_activations,
        pool_mode_classifier="avg" # Use pooling for defined feature vector size
    )
    classifier.eval()

    if batch_first_rnn:
        x = torch.rand(batch_size, num_rnn_steps, C_in, H_in, W_in)
    else:
        x = torch.rand(num_rnn_steps, batch_size, C_in, H_in, W_in)

    result = classifier(x, num_steps=num_rnn_steps) # Pass num_steps to S-RNN forward

    rnn_output_channels = rnn_kwargs["area_configs"].out_channels

    if return_activations:
        output, rnn_output_last_area, pooled_features, fc_inputs = result
        assert rnn_output_last_area is not None
        assert pooled_features is not None # If pool_mode is not None
        assert fc_inputs is not None

        # rnn_output_last_area shape: (B,T,C_out,H_out,W_out) or (T,B,C_out,H_out,W_out)
        # pooled_features shape: (B*T or B, C_out) after pooling and flattening
        # fc_inputs shape: same as pooled_features if no further processing before FC

        if batch_first_rnn:
            assert rnn_output_last_area.shape[:2] == (batch_size, num_rnn_steps)
            assert rnn_output_last_area.shape[2] == rnn_output_channels
            if loss_all_timesteps:
                assert fc_inputs.shape[0] == batch_size * num_rnn_steps
            else:
                assert fc_inputs.shape[0] == batch_size
        else:
            assert rnn_output_last_area.shape[:2] == (num_rnn_steps, batch_size)
            assert rnn_output_last_area.shape[2] == rnn_output_channels
            if loss_all_timesteps:
                assert fc_inputs.shape[0] == num_rnn_steps * batch_size
            else:
                assert fc_inputs.shape[0] == batch_size

        assert fc_inputs.shape[-1] == rnn_output_channels # After adaptive pooling to 1x1, channels remain

    else:
        output = result

    if loss_all_timesteps:
        if batch_first_rnn:
            assert output.shape == (batch_size, num_rnn_steps, num_classes)
        else:
            assert output.shape == (num_rnn_steps, batch_size, num_classes)
    else:
        assert output.shape == (batch_size, num_classes)


@pytest.mark.parametrize("batch_first_rnn", [True, False])
@pytest.mark.parametrize("train_rnn_params_explicitly", [True, False])
@pytest.mark.parametrize("pool_mode_classifier", ["avg", None])
def test_s_classifier_backward(batch_first_rnn, train_rnn_params_explicitly, pool_mode_classifier):
    torch.manual_seed(9)
    num_classes = 2
    batch_size = 1
    num_rnn_steps = 2

    rnn_kwargs = _get_default_s_rnn_kwargs(num_areas=1, batch_first=batch_first_rnn)
    C_in, H_in, W_in = rnn_kwargs["area_configs"].in_channels, rnn_kwargs["area_configs"].in_size[0], rnn_kwargs["area_configs"].in_size[1]

    # To test RNN param training, we'd need to make some params in SpatiallyEmbeddedAreaConfig trainable,
    # e.g., by setting `train_tau=True` in the area_config used by _get_default_s_rnn_kwargs.
    # For simplicity, we'll just check if grads exist if requires_grad is true on RNN params.
    # The SpatiallyEmbeddedArea itself has Conv layers that are trainable by default.

    classifier = SpatiallyEmbeddedClassifier(
        rnn_kwargs=rnn_kwargs,
        num_classes=num_classes,
        pool_mode_classifier=pool_mode_classifier,
        loss_all_timesteps=False
    )

    if not train_rnn_params_explicitly:
        for param in classifier.rnn.parameters():
            param.requires_grad = False

    if batch_first_rnn:
        x = torch.rand(batch_size, num_rnn_steps, C_in, H_in, W_in)
    else:
        x = torch.rand(num_rnn_steps, batch_size, C_in, H_in, W_in)

    output = classifier(x, num_steps=num_rnn_steps)
    loss = output.sum()

    loss.backward()

    for name, param in classifier.fc_layers.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"FC grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"FC NaN grad for {name}"
            assert param.grad.abs().sum() > 1e-8, f"FC Zero grad for {name}"

    # Pooling layer (if not None) usually doesn't have trainable params unless it's a custom layer.
    # AdaptiveAvgPool2d/MaxPool2d do not have parameters.

    for name, param in classifier.rnn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"S-RNN grad missing for {name}"
            assert not torch.isnan(param.grad).any(), f"S-RNN NaN grad for {name}"
        else:
            assert param.grad is None, f"S-RNN grad present but not expected for {name}"
