import torch
import pytest

from bioplnn.models.connectome import ConnectomeODERNN, ConnectomeRNN


def test_connectome_rnn_initialization():
    # Test initialization with default parameters
    connectome = torch.eye(10)
    rnn = ConnectomeRNN(input_size=10, num_neurons=10, connectome=connectome)
    assert rnn is not None
    assert rnn.connectome.shape == (10, 10)
    assert rnn.input_size == 10
    assert rnn.num_neurons == 10


def test_connectome_rnn_initialization_detailed():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    # Test with num_neuron_types > 0
    rnn_nt = ConnectomeRNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        num_neuron_types=2,
        neuron_type=torch.randint(0, 2, (10,)),
    )
    assert rnn_nt is not None
    assert rnn_nt.num_neuron_types == 2
    assert rnn_nt.neuron_type.shape == (10,)

    # Test with neuron_class and neuron_class_mode='per_neuron'
    rnn_nc_pn = ConnectomeRNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        neuron_class=["A", "B"] * 5,  # list of strings
        neuron_class_mode="per_neuron",
    )
    assert rnn_nc_pn is not None
    assert len(rnn_nc_pn.neuron_class) == 10

    # Test with neuron_class and neuron_class_mode='per_neuron_type'
    rnn_nc_pnt = ConnectomeRNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        num_neuron_types=2,
        neuron_type=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        neuron_class={0: "Excitatory", 1: "Inhibitory"},  # mapping
        neuron_class_mode="per_neuron_type",
    )
    assert rnn_nc_pnt is not None
    assert len(rnn_nc_pnt.neuron_class) == 2

    # Test with neuron_class as a single string
    rnn_nc_single = ConnectomeRNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        neuron_class="DefaultNeuron",
        neuron_class_mode="per_neuron", # mode should adapt or be implicitly per_neuron
    )
    assert rnn_nc_single is not None
    assert len(rnn_nc_single.neuron_class) == 10 # Should broadcast
    assert all(nc == "DefaultNeuron" for nc in rnn_nc_single.neuron_class)


def test_connectome_rnn_initialization_parameters():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    input_size = 5
    num_neurons = 10

    # Test neuron_tau_init variations
    rnn_tau_scalar = ConnectomeRNN(input_size, num_neurons, connectome, neuron_tau_init=2.0)
    assert torch.allclose(rnn_tau_scalar.tau, torch.tensor([2.0] * num_neurons))

    rnn_tau_list = ConnectomeRNN(input_size, num_neurons, connectome, neuron_tau_init=[1.0 + i for i in range(num_neurons)])
    assert rnn_tau_list.tau.shape == (num_neurons,)

    rnn_tau_tensor = ConnectomeRNN(input_size, num_neurons, connectome, neuron_tau_init=torch.arange(1.0, num_neurons + 1.0))
    assert rnn_tau_tensor.tau.shape == (num_neurons,)

    # Test train_tau
    rnn_train_tau_true = ConnectomeRNN(input_size, num_neurons, connectome, train_tau=True)
    assert rnn_train_tau_true.tau.requires_grad
    rnn_train_tau_false = ConnectomeRNN(input_size, num_neurons, connectome, train_tau=False)
    assert not rnn_train_tau_false.tau.requires_grad

    # Test neuron_nonlinearity variations
    rnn_nl_str = ConnectomeRNN(input_size, num_neurons, connectome, neuron_nonlinearity='relu')
    assert isinstance(rnn_nl_str.neuron_nonlinearity_fn[0], torch.nn.ReLU)

    rnn_nl_module = ConnectomeRNN(input_size, num_neurons, connectome, neuron_nonlinearity=torch.nn.Sigmoid())
    assert isinstance(rnn_nl_module.neuron_nonlinearity_fn[0], torch.nn.Sigmoid)

    # Test per_neuron_type nonlinearity
    neuron_type_tensor = torch.randint(0, 2, (num_neurons,))
    rnn_nl_map_pnt = ConnectomeRNN(
        input_size, num_neurons, connectome,
        num_neuron_types=2, neuron_type=neuron_type_tensor,
        neuron_nonlinearity={0: 'tanh', 1: 'relu'},
        neuron_class_mode='per_neuron_type' # Though not directly for nonlinearity, it aligns with type-based setup
    )
    assert isinstance(rnn_nl_map_pnt.neuron_nonlinearity_fn[0], torch.nn.Tanh) # Assuming first neuron is type 0
    assert isinstance(rnn_nl_map_pnt.neuron_nonlinearity_fn[1], torch.nn.ReLU) # Assuming second type is type 1

    # Test use_dense_input_projection and use_dense_output_projection
    rnn_dense_in = ConnectomeRNN(input_size, num_neurons, connectome, use_dense_input_projection=True)
    assert rnn_dense_in.input_projection is not None
    assert rnn_dense_in.input_projection.weight.shape == (num_neurons, input_size)

    rnn_dense_out = ConnectomeRNN(input_size, num_neurons, connectome, use_dense_output_projection=True)
    assert rnn_dense_out.output_projection is not None
    assert rnn_dense_out.output_projection.weight.shape == (input_size, num_neurons) # output_size is input_size if not specified

    # Test with provided input_projection and output_projection (tensor)
    custom_input_proj = torch.rand(num_neurons, input_size)
    rnn_custom_in_proj = ConnectomeRNN(input_size, num_neurons, connectome, input_projection=custom_input_proj)
    assert torch.allclose(rnn_custom_in_proj.input_projection.weight, custom_input_proj)

    custom_output_proj = torch.rand(input_size, num_neurons)
    rnn_custom_out_proj = ConnectomeRNN(input_size, num_neurons, connectome, output_projection=custom_output_proj)
    assert torch.allclose(rnn_custom_out_proj.output_projection.weight, custom_output_proj)

    # Test batch_first
    rnn_batch_first_true = ConnectomeRNN(input_size, num_neurons, connectome, batch_first=True)
    assert rnn_batch_first_true.batch_first
    rnn_batch_first_false = ConnectomeRNN(input_size, num_neurons, connectome, batch_first=False)
    assert not rnn_batch_first_false.batch_first


def test_connectome_rnn_initialization_error_handling():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    input_size = 5
    num_neurons = 10

    # Error: neuron_type provided when num_neuron_types is 0
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=0,
            neuron_type=torch.randint(0, 1, (num_neurons,)),
        )

    # Error: neuron_type not provided when num_neuron_types > 0
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size, num_neurons, connectome, num_neuron_types=2
        )

    # Error: neuron_type length mismatch with num_neurons
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.randint(0, 2, (num_neurons - 1,)),
        )

    # Error: neuron_class (list) length mismatch with num_neurons when mode is 'per_neuron'
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            neuron_class=["A"] * (num_neurons - 1),
            neuron_class_mode="per_neuron",
        )

    # Error: neuron_class (dict) key mismatch with neuron_type values when mode is 'per_neuron_type'
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.zeros(num_neurons, dtype=torch.long), # all type 0
            neuron_class={1: "TypeOne"}, # Missing type 0
            neuron_class_mode="per_neuron_type",
        )

    # Error: neuron_tau_init (list) length mismatch
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            neuron_tau_init=[1.0] * (num_neurons -1),
            neuron_tau_mode="per_neuron",
        )

    # Error: neuron_tau_init (dict) key mismatch for 'per_neuron_type'
    with pytest.raises(ValueError):
        ConnectomeRNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.zeros(num_neurons, dtype=torch.long), # all type 0
            neuron_tau_init={1: 10.0}, # Missing type 0
            neuron_tau_mode="per_neuron_type",
        )

    # Error: Invalid neuron_tau_mode
    with pytest.raises(ValueError):
        ConnectomeRNN(input_size, num_neurons, connectome, neuron_tau_mode="invalid_mode")

    # Error: Invalid neuron_class_mode
    with pytest.raises(ValueError):
        ConnectomeRNN(input_size, num_neurons, connectome, neuron_class_mode="invalid_mode")


def test_connectome_odernn_initialization():
    # Test initialization with default parameters
    connectome = torch.eye(10)
    odenn = ConnectomeODERNN(
        input_size=10, num_neurons=10, connectome=connectome
    )
    assert odenn is not None
    assert odenn.connectome.shape == (10, 10)
    assert odenn.input_size == 10
    assert odenn.num_neurons == 10


def test_connectome_odernn_initialization_detailed():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    # Test with num_neuron_types > 0
    odernn_nt = ConnectomeODERNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        num_neuron_types=2,
        neuron_type=torch.randint(0, 2, (10,)),
    )
    assert odernn_nt is not None
    assert odernn_nt.num_neuron_types == 2
    assert odernn_nt.neuron_type.shape == (10,)

    # Test with neuron_class and neuron_class_mode='per_neuron'
    odernn_nc_pn = ConnectomeODERNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        neuron_class=["A", "B"] * 5,  # list of strings
        neuron_class_mode="per_neuron",
    )
    assert odernn_nc_pn is not None
    assert len(odernn_nc_pn.neuron_class) == 10

    # Test with neuron_class and neuron_class_mode='per_neuron_type'
    odernn_nc_pnt = ConnectomeODERNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        num_neuron_types=2,
        neuron_type=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        neuron_class={0: "Excitatory", 1: "Inhibitory"},  # mapping
        neuron_class_mode="per_neuron_type",
    )
    assert odernn_nc_pnt is not None
    assert len(odernn_nc_pnt.neuron_class) == 2

    # Test with neuron_class as a single string
    odernn_nc_single = ConnectomeODERNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        neuron_class="DefaultNeuron",
        neuron_class_mode="per_neuron",
    )
    assert odernn_nc_single is not None
    assert len(odernn_nc_single.neuron_class) == 10
    assert all(nc == "DefaultNeuron" for nc in odernn_nc_single.neuron_class)

    # Test specific ODERN solver and options
    odernn_solver = ConnectomeODERNN(
        input_size=5,
        num_neurons=10,
        connectome=connectome,
        solver="rk4",
        solver_options={"step_size": 0.1}
    )
    assert odernn_solver.solver == "rk4"
    assert odernn_solver.solver_options == {"step_size": 0.1}


def test_connectome_odernn_initialization_parameters():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    input_size = 5
    num_neurons = 10

    # Test neuron_tau_init variations
    odernn_tau_scalar = ConnectomeODERNN(input_size, num_neurons, connectome, neuron_tau_init=2.0)
    assert torch.allclose(odernn_tau_scalar.tau, torch.tensor([2.0] * num_neurons))

    odernn_tau_list = ConnectomeODERNN(input_size, num_neurons, connectome, neuron_tau_init=[1.0 + i for i in range(num_neurons)])
    assert odernn_tau_list.tau.shape == (num_neurons,)

    odernn_tau_tensor = ConnectomeODERNN(input_size, num_neurons, connectome, neuron_tau_init=torch.arange(1.0, num_neurons + 1.0))
    assert odernn_tau_tensor.tau.shape == (num_neurons,)

    # Test train_tau
    odernn_train_tau_true = ConnectomeODERNN(input_size, num_neurons, connectome, train_tau=True)
    assert odernn_train_tau_true.tau.requires_grad
    odernn_train_tau_false = ConnectomeODERNN(input_size, num_neurons, connectome, train_tau=False)
    assert not odernn_train_tau_false.tau.requires_grad

    # Test neuron_nonlinearity variations
    odernn_nl_str = ConnectomeODERNN(input_size, num_neurons, connectome, neuron_nonlinearity='relu')
    assert isinstance(odernn_nl_str.neuron_nonlinearity_fn[0], torch.nn.ReLU)

    odernn_nl_module = ConnectomeODERNN(input_size, num_neurons, connectome, neuron_nonlinearity=torch.nn.Sigmoid())
    assert isinstance(odernn_nl_module.neuron_nonlinearity_fn[0], torch.nn.Sigmoid)

    neuron_type_tensor = torch.randint(0, 2, (num_neurons,))
    odernn_nl_map_pnt = ConnectomeODERNN(
        input_size, num_neurons, connectome,
        num_neuron_types=2, neuron_type=neuron_type_tensor,
        neuron_nonlinearity={0: 'tanh', 1: 'relu'},
        neuron_class_mode='per_neuron_type'
    )
    assert isinstance(odernn_nl_map_pnt.neuron_nonlinearity_fn[0], torch.nn.Tanh)
    assert isinstance(odernn_nl_map_pnt.neuron_nonlinearity_fn[1], torch.nn.ReLU)

    # Test use_dense_input_projection and use_dense_output_projection
    odernn_dense_in = ConnectomeODERNN(input_size, num_neurons, connectome, use_dense_input_projection=True)
    assert odernn_dense_in.input_projection is not None
    assert odernn_dense_in.input_projection.weight.shape == (num_neurons, input_size)

    odernn_dense_out = ConnectomeODERNN(input_size, num_neurons, connectome, use_dense_output_projection=True)
    assert odernn_dense_out.output_projection is not None
    assert odernn_dense_out.output_projection.weight.shape == (input_size, num_neurons)

    # Test with provided input_projection and output_projection (tensor)
    custom_input_proj = torch.rand(num_neurons, input_size)
    odernn_custom_in_proj = ConnectomeODERNN(input_size, num_neurons, connectome, input_projection=custom_input_proj)
    assert torch.allclose(odernn_custom_in_proj.input_projection.weight, custom_input_proj)

    custom_output_proj = torch.rand(input_size, num_neurons)
    odernn_custom_out_proj = ConnectomeODERNN(input_size, num_neurons, connectome, output_projection=custom_output_proj)
    assert torch.allclose(odernn_custom_out_proj.output_projection.weight, custom_output_proj)

    # Test batch_first
    odernn_batch_first_true = ConnectomeODERNN(input_size, num_neurons, connectome, batch_first=True)
    assert odernn_batch_first_true.batch_first
    odernn_batch_first_false = ConnectomeODERNN(input_size, num_neurons, connectome, batch_first=False)
    assert not odernn_batch_first_false.batch_first


def test_connectome_odernn_initialization_error_handling():
    torch.manual_seed(42)
    connectome = torch.rand(10, 10)
    input_size = 5
    num_neurons = 10

    # Error: neuron_type provided when num_neuron_types is 0
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=0,
            neuron_type=torch.randint(0, 1, (num_neurons,)),
        )

    # Error: neuron_type not provided when num_neuron_types > 0
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size, num_neurons, connectome, num_neuron_types=2
        )

    # Error: neuron_type length mismatch with num_neurons
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.randint(0, 2, (num_neurons - 1,)),
        )

    # Error: neuron_class (list) length mismatch with num_neurons when mode is 'per_neuron'
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            neuron_class=["A"] * (num_neurons - 1),
            neuron_class_mode="per_neuron",
        )

    # Error: neuron_class (dict) key mismatch with neuron_type values when mode is 'per_neuron_type'
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.zeros(num_neurons, dtype=torch.long), # all type 0
            neuron_class={1: "TypeOne"}, # Missing type 0
            neuron_class_mode="per_neuron_type",
        )

    # Error: neuron_tau_init (list) length mismatch
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            neuron_tau_init=[1.0] * (num_neurons -1),
            neuron_tau_mode="per_neuron",
        )

    # Error: neuron_tau_init (dict) key mismatch for 'per_neuron_type'
    with pytest.raises(ValueError):
        ConnectomeODERNN(
            input_size,
            num_neurons,
            connectome,
            num_neuron_types=2,
            neuron_type=torch.zeros(num_neurons, dtype=torch.long), # all type 0
            neuron_tau_init={1: 10.0}, # Missing type 0
            neuron_tau_mode="per_neuron_type",
        )

    # Error: Invalid neuron_tau_mode
    with pytest.raises(ValueError):
        ConnectomeODERNN(input_size, num_neurons, connectome, neuron_tau_mode="invalid_mode")

    # Error: Invalid neuron_class_mode
    with pytest.raises(ValueError):
        ConnectomeODERNN(input_size, num_neurons, connectome, neuron_class_mode="invalid_mode")

    # Error: Invalid solver
    with pytest.raises(ValueError):
        ConnectomeODERNN(input_size, num_neurons, connectome, solver="invalid_solver_name")


def test_connectome_rnn_update_fn():
    # Test update function
    batch_size = 10
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    rnn = ConnectomeRNN(
        input_size=input_size, num_neurons=num_neurons, connectome=connectome
    )

    x_t = torch.rand(input_size, batch_size)
    h = torch.rand(num_neurons, batch_size)

    h_new = rnn.update_fn(x_t, h)
    assert h_new.shape == h.shape


def test_connectome_odernn_update_fn():
    # Test update function
    batch_size = 16
    num_steps = 5
    input_size = 10
    num_neurons = 10
    connectome = torch.eye(num_neurons)

    rnn = ConnectomeODERNN(
        input_size=input_size, num_neurons=num_neurons, connectome=connectome
    )

    t = torch.zeros(batch_size)
    h = torch.rand(batch_size, num_neurons)
    x = torch.rand(num_steps, input_size, batch_size)
    args = {"x": x, "start_time": 0.0, "end_time": 1.0}

    dhdt = rnn.update_fn(t, h, args)
    assert dhdt.shape == h.shape


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("provide_neuron_state0", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_steps", [1, 5])
def test_connectome_rnn_forward(batch_first, provide_neuron_state0, dtype, num_steps):
    torch.manual_seed(42)
    batch_size = 16
    input_size = 10
    num_neurons = 20
    connectome = torch.rand(num_neurons, num_neurons, dtype=dtype)

    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        batch_first=batch_first,
        neuron_tau_init=torch.rand(num_neurons, dtype=dtype) + 0.5, # Ensure tau > 0
        neuron_nonlinearity='relu'
    )
    rnn.to(dtype) # Move model parameters to the specified dtype

    if batch_first:
        x = torch.rand(batch_size, num_steps, input_size, dtype=dtype)
    else:
        x = torch.rand(num_steps, batch_size, input_size, dtype=dtype)

    neuron_state0 = None
    if provide_neuron_state0:
        if batch_first:
            neuron_state0 = torch.rand(batch_size, num_neurons, dtype=dtype)
        else:
            neuron_state0 = torch.rand(num_neurons, batch_size, dtype=dtype)

    outputs, hidden_states = rnn.forward(x, neuron_state0=neuron_state0)

    assert outputs is not None
    assert hidden_states is not None

    expected_output_dim1 = num_steps if batch_first else batch_size
    expected_output_dim0 = batch_size if batch_first else num_steps

    # Output is typically the state of the last layer, or a projection of it
    # If output_projection is None, output is effectively the hidden states
    if rnn.output_projection is None:
        expected_outputs_shape = (expected_output_dim0, expected_output_dim1, num_neurons) if batch_first else (expected_output_dim0, expected_output_dim1, num_neurons)
    else: # if there is an output projection, output size is input_size (by default if output_size not specified)
        expected_outputs_shape = (expected_output_dim0, expected_output_dim1, input_size) if batch_first else (expected_output_dim0, expected_output_dim1, input_size)

    # For outputs (all time steps)
    if batch_first:
        assert outputs.shape == (batch_size, num_steps, rnn.output_size)
    else:
        assert outputs.shape == (num_steps, batch_size, rnn.output_size)

    # For hidden_states (final time step)
    if batch_first:
        assert hidden_states.shape == (batch_size, num_neurons)
    else:
        assert hidden_states.shape == (num_neurons, batch_size)

    assert outputs.dtype == dtype
    assert hidden_states.dtype == dtype


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("provide_neuron_state0", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_evals", [None, 2, 5]) # None means eval_times must be provided
@pytest.mark.parametrize("provide_eval_times", [True, False])
def test_connectome_odernn_forward(batch_first, provide_neuron_state0, dtype, num_evals, provide_eval_times):
    torch.manual_seed(42)
    batch_size = 8 # Smaller batch for potentially slower ODE tests
    input_size = 4
    num_neurons = 6
    num_steps_input = 5 # Number of time points in the input tensor x

    # Ensure valid combination of num_evals and provide_eval_times
    if num_evals is None and not provide_eval_times:
        pytest.skip("num_evals cannot be None if provide_eval_times is False")
    if num_evals is not None and provide_eval_times:
        # Let's test this specific scenario: if both are provided, eval_times takes precedence.
        pass


    connectome = torch.rand(num_neurons, num_neurons, dtype=dtype)
    odernn = ConnectomeODERNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        batch_first=batch_first,
        neuron_tau_init=torch.rand(num_neurons, dtype=dtype) + 0.5,
        neuron_nonlinearity='tanh', # Common for ODEs
        solver='dopri5', # A common adaptive step solver
        solver_options={'atol': 1e-3, 'rtol': 1e-3} # Looser tolerances for faster tests
    )
    odernn.to(dtype)

    if batch_first:
        x = torch.rand(batch_size, num_steps_input, input_size, dtype=dtype)
    else:
        x = torch.rand(num_steps_input, batch_size, input_size, dtype=dtype)

    neuron_state0 = None
    if provide_neuron_state0:
        if batch_first:
            neuron_state0 = torch.rand(batch_size, num_neurons, dtype=dtype)
        else:
            neuron_state0 = torch.rand(num_neurons, batch_size, dtype=dtype)

    eval_times_tensor = None
    if provide_eval_times:
        eval_times_tensor = torch.linspace(0, num_steps_input -1 , (num_evals or num_steps_input), dtype=dtype) # num_evals or default if num_evals is None
        if batch_first: # eval_times can be (batch_size, num_eval_points) or (num_eval_points)
             eval_times_tensor = eval_times_tensor.unsqueeze(0).expand(batch_size, -1)


    outputs, hidden_states, timestamps = odernn.forward(
        x,
        neuron_state0=neuron_state0,
        num_evals=num_evals if not provide_eval_times else None, # Pass num_evals only if eval_times not given
        eval_times=eval_times_tensor
    )

    assert outputs is not None
    assert hidden_states is not None
    assert timestamps is not None

    expected_num_eval_points = len(eval_times_tensor) if provide_eval_times else num_evals
    if expected_num_eval_points is None : # if num_evals=None and eval_times=None (though skipped, good to be safe)
        expected_num_eval_points = num_steps_input


    # For outputs (all evaluated time steps)
    if batch_first:
        assert outputs.shape == (batch_size, expected_num_eval_points, odernn.output_size)
        assert timestamps.shape == (batch_size, expected_num_eval_points)
    else:
        assert outputs.shape == (expected_num_eval_points, batch_size, odernn.output_size)
        assert timestamps.shape == (expected_num_eval_points,) # Timestamps usually don't have batch dim if not batch_first for eval_times

    # For hidden_states (final time step's state)
    if batch_first:
        assert hidden_states.shape == (batch_size, num_neurons)
    else:
        assert hidden_states.shape == (num_neurons, batch_size)

    assert outputs.dtype == dtype
    assert hidden_states.dtype == dtype
    assert timestamps.dtype == dtype


@pytest.mark.parametrize("train_connectome", [True, False])
@pytest.mark.parametrize("train_tau", [True, False])
@pytest.mark.parametrize("use_dense_input_projection", [True, False])
@pytest.mark.parametrize("use_dense_output_projection", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_connectome_rnn_backward(
    train_connectome, train_tau, use_dense_input_projection, use_dense_output_projection, batch_first
):
    torch.manual_seed(42)
    input_size = 3
    num_neurons = 4
    batch_size = 2
    num_steps = 3
    dtype = torch.float32 # Backward pass tests are often done with float32 for simplicity

    connectome_tensor = torch.rand(num_neurons, num_neurons, dtype=dtype, requires_grad=train_connectome)

    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome_tensor,
        train_connectome=train_connectome,
        neuron_tau_init=torch.rand(num_neurons, dtype=dtype) + 0.5,
        train_tau=train_tau,
        use_dense_input_projection=use_dense_input_projection,
        use_dense_output_projection=use_dense_output_projection,
        output_size=input_size, # ensure output projection maps back to input_size for simplicity
        batch_first=batch_first,
        neuron_nonlinearity='relu'
    )
    rnn.to(dtype)

    if batch_first:
        x = torch.rand(batch_size, num_steps, input_size, dtype=dtype)
    else:
        x = torch.rand(num_steps, batch_size, input_size, dtype=dtype)

    outputs, _ = rnn.forward(x)
    loss = outputs.sum()
    loss.backward()

    # Check connectome gradients
    if train_connectome:
        assert rnn.connectome.grad is not None
        assert not torch.isnan(rnn.connectome.grad).any()
        assert not torch.isinf(rnn.connectome.grad).any()
    else:
        if hasattr(rnn.connectome, 'grad'): # It might be a non-leaf tensor if train_connectome is False from start
             assert rnn.connectome.grad is None


    # Check tau gradients
    if train_tau:
        assert rnn.tau.grad is not None
        assert not torch.isnan(rnn.tau.grad).any()
        assert not torch.isinf(rnn.tau.grad).any()
    else:
        assert rnn.tau.grad is None

    # Check input projection gradients
    if use_dense_input_projection:
        assert rnn.input_projection.weight.grad is not None
        assert not torch.isnan(rnn.input_projection.weight.grad).any()
        assert not torch.isinf(rnn.input_projection.weight.grad).any()
        if rnn.input_projection.bias is not None:
            assert rnn.input_projection.bias.grad is not None
            assert not torch.isnan(rnn.input_projection.bias.grad).any()
            assert not torch.isinf(rnn.input_projection.bias.grad).any()
    elif rnn.input_projection is not None and isinstance(rnn.input_projection, torch.nn.Module): # Custom module
        # This case requires knowing if the custom module has trainable params.
        # For this test, we assume if it's a module, its params should have grads if they require_grad.
        for param in rnn.input_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


    # Check output projection gradients
    if use_dense_output_projection:
        assert rnn.output_projection.weight.grad is not None
        assert not torch.isnan(rnn.output_projection.weight.grad).any()
        assert not torch.isinf(rnn.output_projection.weight.grad).any()
        if rnn.output_projection.bias is not None:
            assert rnn.output_projection.bias.grad is not None
            assert not torch.isnan(rnn.output_projection.bias.grad).any()
            assert not torch.isinf(rnn.output_projection.bias.grad).any()
    elif rnn.output_projection is not None and isinstance(rnn.output_projection, torch.nn.Module):
        for param in rnn.output_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


@pytest.mark.parametrize("train_connectome", [True, False])
@pytest.mark.parametrize("train_tau", [True, False])
@pytest.mark.parametrize("use_dense_input_projection", [True, False])
@pytest.mark.parametrize("use_dense_output_projection", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("solver", ["dopri5", "rk4"]) # Adjoint is used for backward for some
def test_connectome_odernn_backward(
    train_connectome, train_tau, use_dense_input_projection, use_dense_output_projection, batch_first, solver
):
    torch.manual_seed(42)
    input_size = 3
    num_neurons = 4
    batch_size = 2
    num_steps_input = 3 # Number of time points in input x
    num_evals = 3 # Number of evaluation points for ODE solution
    dtype = torch.float32

    connectome_tensor = torch.rand(num_neurons, num_neurons, dtype=dtype, requires_grad=train_connectome)

    odernn = ConnectomeODERNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome_tensor,
        train_connectome=train_connectome,
        neuron_tau_init=torch.rand(num_neurons, dtype=dtype) + 0.5,
        train_tau=train_tau,
        use_dense_input_projection=use_dense_input_projection,
        use_dense_output_projection=use_dense_output_projection,
        output_size=input_size,
        batch_first=batch_first,
        neuron_nonlinearity='tanh',
        solver=solver,
        solver_options={'atol': 1e-2, 'rtol': 1e-2} if solver != 'rk4' else {'step_size': 0.1} # Looser for speed
    )
    odernn.to(dtype)

    if batch_first:
        x = torch.rand(batch_size, num_steps_input, input_size, dtype=dtype)
    else:
        x = torch.rand(num_steps_input, batch_size, input_size, dtype=dtype)

    # For ODERN, backward through the ode_solve operation requires 'adjoint' method for some solvers,
    # or for the solver to be reversible. The `torchdiffeq` library handles this.
    # We are testing if the gradients reach our parameters.
    outputs, _, _ = odernn.forward(x, num_evals=num_evals)
    loss = outputs.sum()
    loss.backward()

    if train_connectome:
        assert odernn.connectome.grad is not None
        assert not torch.isnan(odernn.connectome.grad).any()
        assert not torch.isinf(odernn.connectome.grad).any()
    else:
        if hasattr(odernn.connectome, 'grad'):
             assert odernn.connectome.grad is None

    if train_tau:
        assert odernn.tau.grad is not None
        assert not torch.isnan(odernn.tau.grad).any()
        assert not torch.isinf(odernn.tau.grad).any()
    else:
        assert odernn.tau.grad is None

    if use_dense_input_projection:
        assert odernn.input_projection.weight.grad is not None
        assert not torch.isnan(odernn.input_projection.weight.grad).any()
        assert not torch.isinf(odernn.input_projection.weight.grad).any()
        if odernn.input_projection.bias is not None:
            assert odernn.input_projection.bias.grad is not None
    elif odernn.input_projection is not None and isinstance(odernn.input_projection, torch.nn.Module):
        for param in odernn.input_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

    if use_dense_output_projection:
        assert odernn.output_projection.weight.grad is not None
        assert not torch.isnan(odernn.output_projection.weight.grad).any()
        assert not torch.isinf(odernn.output_projection.weight.grad).any()
        if odernn.output_projection.bias is not None:
            assert odernn.output_projection.bias.grad is not None
    elif odernn.output_projection is not None and isinstance(odernn.output_projection, torch.nn.Module):
        for param in odernn.output_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


# Helper Method Tests
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("provide_neuron_state0", [True, False])
@pytest.mark.parametrize("use_init_fn", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_init_neuron_state(batch_first, provide_neuron_state0, use_init_fn, dtype):
    torch.manual_seed(42)
    num_neurons = 10
    batch_size = 4
    input_size = 5 # Not directly used by init_neuron_state but needed for RNN init
    connectome = torch.eye(num_neurons, dtype=dtype)

    init_fn_mock = None
    if use_init_fn:
        # Simple init_fn that returns ones, to check if it's called
        init_fn_mock = lambda shape, batch_first_flag, dtype_val, device_val: torch.ones(shape, dtype=dtype_val, device=device_val)


    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        batch_first=batch_first,
        neuron_state_init_fn=init_fn_mock
    )
    rnn.to(dtype)

    neuron_state0_input = None
    if provide_neuron_state0:
        if batch_first:
            neuron_state0_input = torch.rand(batch_size, num_neurons, dtype=dtype)
        else:
            neuron_state0_input = torch.rand(num_neurons, batch_size, dtype=dtype)

    # x_t is needed for device and sometimes shape inference if neuron_state0 is None
    # Its content doesn't matter here, only shape and device.
    if batch_first:
        x_t_dummy = torch.empty(batch_size, input_size, dtype=dtype, device=rnn.connectome.device)
    else:
        x_t_dummy = torch.empty(input_size, batch_size, dtype=dtype, device=rnn.connectome.device)


    initialized_state = rnn.init_neuron_state(neuron_state0_input, x_t_dummy, batch_size)

    expected_shape = (batch_size, num_neurons) if batch_first else (num_neurons, batch_size)
    assert initialized_state.shape == expected_shape
    assert initialized_state.dtype == dtype
    assert initialized_state.device == rnn.connectome.device

    if provide_neuron_state0:
        # If neuron_state0 is provided, it should be returned (possibly reshaped if dimensions were squeezed)
        # For this test, we assume neuron_state0_input already has the correct batch dimension order
        assert torch.allclose(initialized_state, neuron_state0_input.view(expected_shape))
    elif use_init_fn:
        assert torch.allclose(initialized_state, torch.ones(expected_shape, dtype=dtype, device=rnn.connectome.device))
    else:
        # Default initialization is zeros
        assert torch.allclose(initialized_state, torch.zeros(expected_shape, dtype=dtype, device=rnn.connectome.device))


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_neuron_types", [True, False])
def test_query_neuron_states(batch_first, use_neuron_types):
    torch.manual_seed(42)
    input_size = 3
    num_neurons = 6
    batch_size = 2
    num_steps = 4
    dtype = torch.float32

    neuron_type_tensor = None
    num_neuron_types = 0
    if use_neuron_types:
        num_neuron_types = 2
        # Simple assignment: first half type 0, second half type 1
        neuron_type_tensor = torch.tensor([0] * (num_neurons // 2) + [1] * (num_neurons - num_neurons // 2), dtype=torch.long)

    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=torch.rand(num_neurons, num_neurons, dtype=dtype),
        batch_first=batch_first,
        num_neuron_types=num_neuron_types,
        neuron_type=neuron_type_tensor,
        store_history=True # Crucial for query_neuron_states
    )
    rnn.to(dtype)

    if batch_first:
        x = torch.rand(batch_size, num_steps, input_size, dtype=dtype)
    else:
        x = torch.rand(num_steps, batch_size, input_size, dtype=dtype)

    rnn.forward(x) # Populate history

    # --- Test Cases for query_neuron_states ---
    history = rnn.history_h_neurons # (num_steps, batch_size, num_neurons) or (batch_size, num_steps, num_neurons)

    # Case 1: Get all states for a specific time_step
    queried_t1 = rnn.query_neuron_states(time_step=1)
    expected_t1_shape = (batch_size, num_neurons) if batch_first else (1, batch_size, num_neurons) # query returns (1,B,N) if time_step is int
    assert queried_t1.shape[-2:] == (batch_size, num_neurons) # Check last two dims
    if batch_first:
        assert torch.allclose(queried_t1.squeeze(0), history[:, 1, :])
    else:
        assert torch.allclose(queried_t1.squeeze(0), history[1, :, :])


    # Case 2: Get states for a specific batch index
    queried_b0 = rnn.query_neuron_states(batch_idx=0)
    expected_b0_shape = (num_steps, 1, num_neurons) if batch_first else (num_steps, 1, num_neurons)
    assert queried_b0.shape == expected_b0_shape
    if batch_first:
        assert torch.allclose(queried_b0, history[0, :, :].unsqueeze(0 if batch_first else 1).permute(1,0,2)) # Need to adjust permute for batch_first
    else:
         assert torch.allclose(queried_b0, history[:, 0, :].unsqueeze(1))


    # Case 3: Get all states for a specific neuron index
    queried_n2 = rnn.query_neuron_states(neuron_idx_query=2)
    expected_n2_shape = (num_steps, batch_size, 1)
    assert queried_n2.shape == expected_n2_shape
    if batch_first:
        assert torch.allclose(queried_n2, history[:, :, 2].unsqueeze(-1))
    else:
        assert torch.allclose(queried_n2, history[:, :, 2].unsqueeze(-1))


    # Case 4: Get all states (equivalent to history if no args)
    queried_all = rnn.query_neuron_states()
    # query_neuron_states returns (T, B, N) by default if batch_first=False,
    # and (B, T, N) if batch_first=True and no specific query reduces a dim
    if batch_first:
        assert queried_all.shape == (batch_size, num_steps, num_neurons)
        assert torch.allclose(queried_all, history)
    else:
        assert queried_all.shape == (num_steps, batch_size, num_neurons)
        assert torch.allclose(queried_all, history)


    if use_neuron_types:
        # Case 5: Get states for a specific neuron type
        queried_type0 = rnn.query_neuron_states(neuron_type_query=0)
        type0_indices = (neuron_type_tensor == 0).nonzero(as_tuple=True)[0]
        num_type0_neurons = len(type0_indices)
        expected_type0_shape = (num_steps, batch_size, num_type0_neurons) if not batch_first else (batch_size, num_steps, num_type0_neurons)
        assert queried_type0.shape == expected_type0_shape
        if batch_first:
             assert torch.allclose(queried_type0, history[:, :, type0_indices])
        else:
             assert torch.allclose(queried_type0, history[:, :, type0_indices])


        # Case 6: Specific time, batch, and type
        queried_t0_b0_type1 = rnn.query_neuron_states(time_step=0, batch_idx=0, neuron_type_query=1)
        type1_indices = (neuron_type_tensor == 1).nonzero(as_tuple=True)[0]
        num_type1_neurons = len(type1_indices)

        # Shape depends on how query handles multiple single-index queries.
        # Based on current understanding, it should reduce dimensions.
        # (1 time_step, 1 batch_idx, num_type1_neurons)
        expected_t0_b0_type1_shape = (1, 1, num_type1_neurons)
        assert queried_t0_b0_type1.shape[-1] == num_type1_neurons # Check neuron dim

        if batch_first:
            # history is (B, T, N)
            expected_data = history[0, 0, type1_indices].reshape(1,1,num_type1_neurons) # Reshape to match squeezed output
        else:
            # history is (T, B, N)
            expected_data = history[0, 0, type1_indices].reshape(1,1,num_type1_neurons)
        assert torch.allclose(queried_t0_b0_type1, expected_data)

    # Case 7: time_step list, neuron_idx list
    time_indices = [0, 2]
    neuron_indices_q = [1, 3]
    queried_multi = rnn.query_neuron_states(time_step=time_indices, neuron_idx_query=neuron_indices_q)
    if batch_first: # B, T_list, N_list
        assert queried_multi.shape == (batch_size, len(time_indices), len(neuron_indices_q))
        assert torch.allclose(queried_multi, history[:, time_indices, :][:, :, neuron_indices_q])
    else: # T_list, B, N_list
        assert queried_multi.shape == (len(time_indices), batch_size, len(neuron_indices_q))
        assert torch.allclose(queried_multi, history[time_indices, :, :][:, :, neuron_indices_q])


@pytest.mark.parametrize("train_tau", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clamp_tau(train_tau, dtype):
    torch.manual_seed(42)
    num_neurons = 4
    input_size = 2
    connectome = torch.eye(num_neurons, dtype=dtype)

    # Initialize taus with values below and above 1.0
    initial_taus = torch.tensor([0.5, 0.8, 1.0, 1.2], dtype=dtype)

    rnn = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        neuron_tau_init=initial_taus.clone(), # Pass a clone
        train_tau=train_tau,
        dt=0.1 # dt is needed for _clamp_tau internal logic for min_tau
    )
    rnn.to(dtype)

    # Check initial taus (before any forward pass or direct call)
    assert torch.allclose(rnn.tau, initial_taus if not train_tau else torch.nn.Parameter(initial_taus))

    # Call _clamp_tau directly to test its effect
    rnn._clamp_tau()

    expected_taus_after_clamp = torch.tensor([1.0, 1.0, 1.0, 1.2], dtype=dtype)

    if train_tau:
        # If train_tau is True, rnn.tau is a Parameter, so compare its .data
        assert torch.allclose(rnn.tau.data, expected_taus_after_clamp)
        assert rnn.tau.requires_grad # Ensure it's still a Parameter
    else:
        assert torch.allclose(rnn.tau, expected_taus_after_clamp)
        assert not rnn.tau.requires_grad

    # Test that calling it again doesn't change already clamped values
    rnn._clamp_tau()
    if train_tau:
        assert torch.allclose(rnn.tau.data, expected_taus_after_clamp)
    else:
        assert torch.allclose(rnn.tau, expected_taus_after_clamp)

    # Test with all taus initially >= 1.0
    initial_taus_above_one = torch.tensor([1.0, 1.5, 2.0, 3.0], dtype=dtype)
    rnn_above = ConnectomeRNN(
        input_size=input_size,
        num_neurons=num_neurons,
        connectome=connectome,
        neuron_tau_init=initial_taus_above_one.clone(),
        train_tau=train_tau,
        dt=0.1
    )
    rnn_above.to(dtype)
    rnn_above._clamp_tau()
    if train_tau:
        assert torch.allclose(rnn_above.tau.data, initial_taus_above_one)
    else:
        assert torch.allclose(rnn_above.tau, initial_taus_above_one)


# Remove the TODO as tests are now comprehensive
# TODO: More extensive testing needs to be done for the connectome RNN and ODERNN
