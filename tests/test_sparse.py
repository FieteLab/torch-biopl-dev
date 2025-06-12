import torch
import pytest
from torch_sparse import SparseTensor # For creating sparse COO tensors
import numpy as np
import os
import tempfile # For mocking file paths

# Models to test
from bioplnn.models.sparse import SparseLinear, SparseRNN, SparseODERNN

# For mocking torch.compile if needed, though direct checking is hard
from unittest.mock import patch

torch.manual_seed(0)

# --- Helper Functions ---

def create_sparse_coo_tensor(size, nnz, device="cpu", dtype=torch.float32, requires_grad_values=False):
    """Creates a random sparse COO tensor."""
    assert len(size) == 2, "Only 2D sparse tensors are supported by this helper."
    row = torch.randint(0, size[0], (nnz,), device=device)
    col = torch.randint(0, size[1], (nnz,), device=device)
    indices = torch.stack([row, col], dim=0)

    # Ensure no duplicate indices for a valid COO tensor, then sort
    # A simple way for testing is to generate unique pairs if nnz is not too large
    # For larger nnz, this might be slow or fail.
    # A more robust way for general use would involve `coalesce`.
    # For testing, let's assume nnz is small enough or accept potential duplicates for simplicity if not coalescing.
    # However, SparseTensor constructor expects sorted and coalesced indices.

    # Create unique indices
    if nnz <= size[0] * size[1]: # Only makes sense if nnz is not trying to be fully dense
        unique_indices_set = set()
        new_indices_list = []
        for i in range(nnz):
            r, c = row[i].item(), col[i].item()
            # Try to make them unique - this is a bit hacky for a general helper
            # but for tests, we can control nnz to be small relative to size.
            # A better approach for truly random sparse: generate more, then unique, then pick nnz.
            # For now, this will do, or rely on SparseTensor to handle/error on duplicates.
            # The `SparseTensor.from_edge_index` handles duplicates by summing values.
            pass # Let SparseTensor handle it, or assume test cases use valid COO.

    values = torch.randn(nnz, device=device, dtype=dtype, requires_grad=requires_grad_values)

    # Using SparseTensor.from_dense as a shortcut if direct COO creation is tricky with sorting/coalescing
    # For true COO creation for SparseLinear, it expects indices and values.
    # Let's try direct creation and ensure it's valid for SparseLinear.
    # SparseLinear currently uses matmul with a dense tensor derived from sparse, so input can be standard COO.

    # A simple way to get a valid COO for testing without deep SparseTensor internals:
    dense_for_coo = torch.zeros(size, device=device, dtype=dtype)
    chosen_indices = torch.stack([
        torch.randint(0, size[0], (nnz,)),
        torch.randint(0, size[1], (nnz,))
    ], dim=0)

    # Ensure indices are unique for this simple method
    # This is not efficient for large nnz but okay for tests
    unique_indices_map = {}
    final_indices_list_r = []
    final_indices_list_c = []
    for i in range(nnz):
        r, c = chosen_indices[0, i].item(), chosen_indices[1, i].item()
        if (r,c) not in unique_indices_map:
            unique_indices_map[(r,c)] = 1
            final_indices_list_r.append(r)
            final_indices_list_c.append(c)

    final_indices = torch.tensor([final_indices_list_r, final_indices_list_c], dtype=torch.long, device=device)
    final_nnz = final_indices.shape[1]
    final_values = torch.randn(final_nnz, device=device, dtype=dtype, requires_grad=requires_grad_values)

    return final_indices, final_values, size


def mock_load_connectivity(path, device="cpu", dtype=torch.float32):
    """Mocks loading a sparse tensor from a path by returning a predefined one."""
    # This function will be patched in for 'load_connectivity_from_path'
    # For the purpose of testing, we just need it to return a valid sparse tensor or its components

    # Example: return components for a 5x5 sparse tensor with 3 non-zero elements
    size = (5,5)
    nnz = 3
    indices = torch.tensor([[0,1,2],[2,3,4]], dtype=torch.long, device=device)
    values = torch.ones(nnz, device=device, dtype=dtype)
    return indices, values, size


# --- SparseLinear Tests ---

@pytest.mark.parametrize("in_features", [5, 10])
@pytest.mark.parametrize("out_features", [3, 8])
@pytest.mark.parametrize("feature_dim", [0, 1, -1])
@pytest.mark.parametrize("bias_flag", [True, False])
@pytest.mark.parametrize("requires_grad_values", [True, False])
@pytest.mark.parametrize("requires_grad_bias", [True, False])
def test_sparse_linear_initialization(in_features, out_features, feature_dim, bias_flag, requires_grad_values, requires_grad_bias):
    torch.manual_seed(1)

    # Connectivity: feature_dim determines which dimension of connectivity matches in_features
    # If feature_dim is 0 or -2 (for a 2D input), connectivity is (out_features, in_features)
    # If feature_dim is 1 or -1 (for a 2D input), connectivity is (in_features, out_features) - Transposed logic

    # For SparseLinear, the weight matrix is (out_features, in_features)
    # The provided connectivity should match this shape.
    connectivity_shape = (out_features, in_features)
    nnz = (in_features * out_features) // 2 # Example sparsity

    indices, values, size = create_sparse_coo_tensor(
        connectivity_shape, nnz, requires_grad_values=requires_grad_values
    )

    if not bias_flag and requires_grad_bias: # Bias grad requires bias to be True
        pytest.skip("Cannot require_grad_bias if bias_flag is False")

    sl = SparseLinear(
        in_features=in_features,
        out_features=out_features,
        connectivity_indices=indices,
        connectivity_values=values,
        connectivity_size=size,
        feature_dim=feature_dim, # This affects how input is multiplied, not connectivity shape itself for SparseLinear's weight
        bias=bias_flag,
        requires_grad_values=requires_grad_values,
        requires_grad_bias=requires_grad_bias if bias_flag else False # Only if bias is present
    )

    assert sl.in_features == in_features
    assert sl.out_features == out_features
    assert sl.feature_dim == feature_dim
    assert sl.weight_indices is indices
    assert sl.weight_values.requires_grad == requires_grad_values
    assert sl.weight_size == size

    if bias_flag:
        assert sl.bias is not None
        assert sl.bias.shape == (out_features,)
        assert sl.bias.requires_grad == requires_grad_bias
    else:
        assert sl.bias is None

    # Test with connectivity as a dense tensor (should raise error or be handled if SparseLinear supports it)
    # Current SparseLinear expects indices, values, size. So dense tensor directly is not supported.
    # Test with invalid connectivity shape (e.g., values shape mismatch with indices)
    with pytest.raises(RuntimeError): # PyTorch sparse COO checks this
         SparseLinear(in_features, out_features, indices, torch.randn(nnz+1), size)

    # Test if connectivity_size doesn't match out_features, in_features
    with pytest.raises(ValueError, match="Connectivity size mismatch with out_features, in_features"):
         SparseLinear(in_features, out_features, indices, values, (out_features+1, in_features))


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("feature_dim", [0, 1, -1]) # Assuming 2D input (batch, features) or (features, batch)
@pytest.mark.parametrize("other_dims", [(), (3,2)]) # For inputs like (B, other, Feat, other2)
def test_sparse_linear_forward(batch_size, feature_dim, other_dims):
    torch.manual_seed(2)
    in_features = 10
    out_features = 5

    # Weight shape (out_features, in_features)
    connectivity_shape = (out_features, in_features)
    indices, values, size = create_sparse_coo_tensor(connectivity_shape, 20)

    sl = SparseLinear(in_features, out_features, indices, values, size, feature_dim=feature_dim, bias=True)

    # Construct input tensor based on feature_dim
    # If feature_dim = 0 or -2 (assuming 2D input for simplicity: (in_feat, batch_size))
    # If feature_dim = 1 or -1 (assuming 2D input: (batch_size, in_feat))

    if feature_dim == 0: # Input: (in_features, batch_size, *other_dims_transposed_if_any)
        input_shape = (in_features, batch_size, *other_dims)
        # Output: (out_features, batch_size, *other_dims)
        expected_out_first_dim = out_features
        expected_out_second_dim = batch_size
    elif feature_dim == 1: # Input: (batch_size, in_features, *other_dims)
        input_shape = (batch_size, in_features, *other_dims)
        # Output: (batch_size, out_features, *other_dims)
        expected_out_first_dim = batch_size
        expected_out_second_dim = out_features
    elif feature_dim == -1: # Input: (*other_dims_first, batch_size, in_features)
        # Let's simplify for this test: (batch_size, *other_dims, in_features)
        input_shape = (batch_size, *other_dims, in_features)
        # Output: (batch_size, *other_dims, out_features)
        expected_out_first_dim = batch_size
        # The rest of the shape depends on how SparseLinear handles broadcasting with feature_dim=-1
        # For a typical matmul with feature_dim=-1, output is (*batch_dims, out_features)
        # So, (batch_size, *other_dims, out_features)
    else: # e.g. feature_dim = -2
        # (batch_size, in_features, *other_dims) -> like feature_dim=1 if other_dims is last
        # This case needs careful thought on how SparseLinear applies matmul.
        # For simplicity, assuming feature_dim is 0, 1, or -1 relative to a core (Batch, Features) or (Features, Batch)
        pytest.skip(f"feature_dim {feature_dim} with other_dims needs specific shape logic not fully covered here.")
        return

    input_tensor = torch.rand(input_shape)
    output = sl(input_tensor)

    # Determine expected output shape
    if feature_dim == 0:
        expected_output_shape = (out_features, batch_size, *other_dims)
    elif feature_dim == 1:
        expected_output_shape = (batch_size, out_features, *other_dims)
    elif feature_dim == -1:
        expected_output_shape = (*input_shape[:-1], out_features)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize("requires_grad_values", [True, False])
@pytest.mark.parametrize("bias_flag", [True, False])
@pytest.mark.parametrize("requires_grad_bias", [True, False])
def test_sparse_linear_backward(requires_grad_values, bias_flag, requires_grad_bias):
    torch.manual_seed(3)
    in_features = 6
    out_features = 3

    if not bias_flag and requires_grad_bias:
        pytest.skip("Cannot require_grad_bias if bias_flag is False")

    indices, values, size = create_sparse_coo_tensor(
        (out_features, in_features), 10, requires_grad_values=requires_grad_values
    )

    sl = SparseLinear(
        in_features, out_features, indices, values, size,
        bias=bias_flag, requires_grad_values=requires_grad_values,
        requires_grad_bias=requires_grad_bias if bias_flag else False
    )

    input_tensor = torch.rand(4, in_features) # (batch_size, in_features)
    output = sl(input_tensor)
    loss = output.sum()
    loss.backward()

    if requires_grad_values:
        assert sl.weight_values.grad is not None
        assert not torch.isnan(sl.weight_values.grad).any()
        assert not torch.isinf(sl.weight_values.grad).any()
        if sl.weight_values.grad.numel() > 0: # Check sum only if there are elements
             assert sl.weight_values.grad.abs().sum() > 1e-8
    else:
        assert sl.weight_values.grad is None

    if bias_flag and requires_grad_bias:
        assert sl.bias.grad is not None
        assert not torch.isnan(sl.bias.grad).any()
        assert not torch.isinf(sl.bias.grad).any()
        assert sl.bias.grad.abs().sum() > 1e-8
    elif sl.bias is not None: # Bias exists but not trained
        assert sl.bias.grad is None


# --- SparseRNN Tests ---

# Mocked paths for connectivity loading
MOCK_HH_PATH = "mock_hh_connectivity.npz"
MOCK_IH_PATH = "mock_ih_connectivity.npz"
MOCK_HO_PATH = "mock_ho_connectivity.npz"


@pytest.fixture(autouse=True)
def manage_mock_connectivity_files():
    # Create dummy files that mock_load_connectivity can "use"
    # This is more about having a valid path string than file content for these tests
    # as load_connectivity_from_path will be patched.

    # For SparseRNN, connectivity often implies (hidden_size, hidden_size) or similar
    # Let's assume hidden_size = 5, input_size = 4, output_size = 3 for these mocks

    # Mock HH (5x5)
    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((5,5), 10)
    # Mock IH (5x4) - hidden_size x input_size
    indices_ih, values_ih, size_ih = create_sparse_coo_tensor((5,4), 8)
    # Mock HO (3x5) - output_size x hidden_size
    indices_ho, values_ho, size_ho = create_sparse_coo_tensor((3,5), 6)

    # Save these if we weren't patching load_connectivity_from_path
    # For now, the paths are just placeholders.
    # np.savez(MOCK_HH_PATH, indices=indices_hh.numpy(), values=values_hh.numpy(), size=np.array(size_hh))
    # np.savez(MOCK_IH_PATH, indices=indices_ih.numpy(), values=values_ih.numpy(), size=np.array(size_ih))
    # np.savez(MOCK_HO_PATH, indices=indices_ho.numpy(), values=values_ho.numpy(), size=np.array(size_ho))

    yield # Test runs here

    # Cleanup (optional, as they are mock paths if load is patched)
    # for p in [MOCK_HH_PATH, MOCK_IH_PATH, MOCK_HO_PATH]:
    #     if os.path.exists(p):
    #         os.remove(p)


@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("use_connectivity_paths", [True, False])
@pytest.mark.parametrize("use_dense_ih", [True, False])
@pytest.mark.parametrize("use_dense_ho", [True, False])
@pytest.mark.parametrize("bias_hh", [True, False])
@pytest.mark.parametrize("nonlinearity", ["Sigmoid", "ReLU"])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("train_flags", [[True,True,True], [False,True,False]]) # train_hh, train_ih, train_ho
def test_sparse_rnn_initialization(
    mock_loader, use_connectivity_paths, use_dense_ih, use_dense_ho,
    bias_hh, nonlinearity, batch_first, train_flags
):
    torch.manual_seed(4)
    input_size = 4
    hidden_size = 5
    output_size = 3 # Relevant if connectivity_ho is used

    train_hh, train_ih, train_ho = train_flags

    # Prepare connectivity arguments
    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 10, requires_grad_values=train_hh)
    indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 8, requires_grad_values=train_ih)
    indices_ho, values_ho, size_ho = create_sparse_coo_tensor((output_size, hidden_size), 6, requires_grad_values=train_ho)

    connectivity_hh_arg = MOCK_HH_PATH if use_connectivity_paths else (indices_hh, values_hh, size_hh)
    connectivity_ih_arg = MOCK_IH_PATH if use_connectivity_paths else (indices_ih, values_ih, size_ih)
    connectivity_ho_arg = MOCK_HO_PATH if use_connectivity_paths else (indices_ho, values_ho, size_ho)

    if use_dense_ih:
        connectivity_ih_arg = None # Dense IH means no sparse connectivity_ih
    if use_dense_ho:
        connectivity_ho_arg = None

    # Skip invalid combinations
    if use_dense_ih and train_ih: # train_ih for sparse values, not dense layer
        pytest.skip("train_ih=True is for sparse values, not applicable if use_dense_ih=True")
    if use_dense_ho and train_ho:
        pytest.skip("train_ho=True is for sparse values, not applicable if use_dense_ho=True")


    rnn = SparseRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size, # output_size needed if HO is used
        connectivity_hh=connectivity_hh_arg,
        connectivity_ih=connectivity_ih_arg,
        connectivity_ho=connectivity_ho_arg,
        bias_hh=bias_hh,
        bias_ih=not use_dense_ih, # Sparse IH can have bias
        bias_ho=not use_dense_ho, # Sparse HO can have bias
        use_dense_ih=use_dense_ih,
        use_dense_ho=use_dense_ho,
        train_hh_values=train_hh,
        train_ih_values=train_ih if not use_dense_ih else False,
        train_ho_values=train_ho if not use_dense_ho else False,
        nonlinearity=nonlinearity,
        batch_first=batch_first
    )

    assert rnn.input_size == input_size
    assert rnn.hidden_size == hidden_size
    assert rnn.output_size == output_size # or hidden_size if HO is not used
    assert rnn.batch_first == batch_first
    assert isinstance(rnn.hh, SparseLinear)
    assert rnn.hh.weight_values.requires_grad == train_hh

    if use_dense_ih:
        assert isinstance(rnn.ih, torch.nn.Linear)
        assert rnn.ih.weight.requires_grad # Dense layers are trainable by default
    elif connectivity_ih_arg is not None:
        assert isinstance(rnn.ih, SparseLinear)
        assert rnn.ih.weight_values.requires_grad == train_ih
    else: # Neither dense nor sparse IH provided (e.g. input_size == hidden_size and no explicit IH)
        assert rnn.ih is None or isinstance(rnn.ih, torch.nn.Identity) # Or some other handling

    if use_dense_ho:
        assert isinstance(rnn.ho, torch.nn.Linear)
        assert rnn.ho.weight.requires_grad
    elif connectivity_ho_arg is not None:
        assert isinstance(rnn.ho, SparseLinear)
        assert rnn.ho.weight_values.requires_grad == train_ho
    else: # No HO layer
        assert rnn.ho is None
        assert rnn.output_size == hidden_size # Output is hidden state if no HO

    # Test error: input_size != hidden_size, connectivity_ih=None, use_dense_ih=False
    if not use_dense_ih and input_size != hidden_size : # And no connectivity_ih
        with pytest.raises(ValueError, match="input_size must equal hidden_size if connectivity_ih is None and use_dense_ih is False"):
            SparseRNN(input_size=input_size+1, hidden_size=hidden_size, connectivity_ih=None, use_dense_ih=False, connectivity_hh=connectivity_hh_arg)

    # Test error: connectivity_ih is dense but use_dense_ih is False
    # This is more of a type error if a dense tensor is passed to SparseLinear setup
    # The current structure assumes connectivity_X_arg is path or (indices, values, size) for sparse.
    # If connectivity_ih_arg was a dense tensor:
    # with pytest.raises(TypeError): # Or ValueError depending on internal checks
    #    SparseRNN(input_size, hidden_size, connectivity_ih=torch.rand(hidden_size, input_size), use_dense_ih=False, ...)


@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("provide_h0", [True, False])
@pytest.mark.parametrize("num_steps", [1, 5])
@pytest.mark.parametrize("use_ho_layer", [True, False]) # Test with and without output layer
def test_sparse_rnn_forward(mock_loader, batch_first, provide_h0, num_steps, use_ho_layer):
    torch.manual_seed(5)
    input_size = 4
    hidden_size = 5
    output_size_if_ho = 3
    batch_size_test = 2

    # Connectivity
    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 10)
    indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 8)

    connectivity_ho_arg = None
    final_output_size = hidden_size
    if use_ho_layer:
        indices_ho, values_ho, size_ho = create_sparse_coo_tensor((output_size_if_ho, hidden_size), 6)
        connectivity_ho_arg = (indices_ho, values_ho, size_ho)
        final_output_size = output_size_if_ho

    rnn = SparseRNN(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size_if_ho,
        connectivity_hh=(indices_hh, values_hh, size_hh),
        connectivity_ih=(indices_ih, values_ih, size_ih),
        connectivity_ho=connectivity_ho_arg,
        batch_first=batch_first
    )
    rnn.eval() # For consistent behavior

    if batch_first:
        x = torch.rand(batch_size_test, num_steps, input_size)
    else:
        x = torch.rand(num_steps, batch_size_test, input_size)

    h0_tensor = None
    if provide_h0:
        h0_tensor = torch.rand(batch_size_test, hidden_size) # Shape (batch, hidden)

    outs, hs = rnn.forward(x, h0=h0_tensor)

    # Check outs shape
    if batch_first:
        assert outs.shape == (batch_size_test, num_steps, final_output_size)
    else:
        assert outs.shape == (num_steps, batch_size_test, final_output_size)

    # Check hs shape (final hidden state)
    assert hs.shape == (batch_size_test, hidden_size)


@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("train_flags", [[True,True,True], [False,False,False]]) # hh, ih, ho
@pytest.mark.parametrize("use_dense_layers", [[False,False], [True,True]]) # ih, ho
def test_sparse_rnn_backward(mock_loader, batch_first, train_flags, use_dense_layers):
    torch.manual_seed(6)
    input_size = 3
    hidden_size = 4
    output_size = 2 # if HO is used
    batch_size_test = 1 # Simpler for backward
    num_steps = 2

    train_hh_v, train_ih_v, train_ho_v = train_flags
    use_dense_ih, use_dense_ho = use_dense_layers

    if use_dense_ih and train_ih_v: pytest.skip("train_ih for sparse values")
    if use_dense_ho and train_ho_v: pytest.skip("train_ho for sparse values")

    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 8, requires_grad_values=train_hh_v)

    conn_ih_arg, conn_ho_arg = None, None
    if not use_dense_ih:
        indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 5, requires_grad_values=train_ih_v)
        conn_ih_arg = (indices_ih, values_ih, size_ih)
    if not use_dense_ho:
        indices_ho, values_ho, size_ho = create_sparse_coo_tensor((output_size, hidden_size), 4, requires_grad_values=train_ho_v)
        conn_ho_arg = (indices_ho, values_ho, size_ho)

    rnn = SparseRNN(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        connectivity_hh=(indices_hh, values_hh, size_hh),
        connectivity_ih=conn_ih_arg, use_dense_ih=use_dense_ih,
        connectivity_ho=conn_ho_arg, use_dense_ho=use_dense_ho,
        train_hh_values=train_hh_v, train_ih_values=train_ih_v if not use_dense_ih else False,
        train_ho_values=train_ho_v if not use_dense_ho else False,
        bias_hh=True, bias_ih=True, bias_ho=True, # Ensure biases exist for grad check
        batch_first=batch_first
    )

    if batch_first:
        x = torch.rand(batch_size_test, num_steps, input_size)
    else:
        x = torch.rand(num_steps, batch_size_test, input_size)

    outs, _ = rnn.forward(x)
    loss = outs.sum()
    loss.backward()

    # Check HH layer
    assert rnn.hh.weight_values.requires_grad == train_hh_v
    if train_hh_v:
        assert rnn.hh.weight_values.grad is not None; assert rnn.hh.weight_values.grad.abs().sum() > 1e-8
    if rnn.hh.bias is not None and rnn.hh.bias.requires_grad: # Bias grad by default if bias=True
        assert rnn.hh.bias.grad is not None; assert rnn.hh.bias.grad.abs().sum() > 1e-8

    # Check IH layer
    if use_dense_ih:
        assert rnn.ih.weight.grad is not None; assert rnn.ih.weight.grad.abs().sum() > 1e-8
        if rnn.ih.bias is not None: assert rnn.ih.bias.grad is not None; assert rnn.ih.bias.grad.abs().sum() > 1e-8
    elif rnn.ih is not None and isinstance(rnn.ih, SparseLinear):
        assert rnn.ih.weight_values.requires_grad == train_ih_v
        if train_ih_v:
            assert rnn.ih.weight_values.grad is not None; assert rnn.ih.weight_values.grad.abs().sum() > 1e-8
        if rnn.ih.bias is not None and rnn.ih.bias.requires_grad:
             assert rnn.ih.bias.grad is not None; assert rnn.ih.bias.grad.abs().sum() > 1e-8

    # Check HO layer
    if use_dense_ho:
        assert rnn.ho.weight.grad is not None; assert rnn.ho.weight.grad.abs().sum() > 1e-8
        if rnn.ho.bias is not None: assert rnn.ho.bias.grad is not None; assert rnn.ho.bias.grad.abs().sum() > 1e-8
    elif rnn.ho is not None and isinstance(rnn.ho, SparseLinear):
        assert rnn.ho.weight_values.requires_grad == train_ho_v
        if train_ho_v:
            assert rnn.ho.weight_values.grad is not None; assert rnn.ho.weight_values.grad.abs().sum() > 1e-8
        if rnn.ho.bias is not None and rnn.ho.bias.requires_grad:
            assert rnn.ho.bias.grad is not None; assert rnn.ho.bias.grad.abs().sum() > 1e-8


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("use_init_fn", [True, False])
def test_sparse_rnn_init_hidden(batch_size, use_init_fn):
    torch.manual_seed(7)
    hidden_size = 5
    rnn = SparseRNN(input_size=3, hidden_size=hidden_size, connectivity_hh=create_sparse_coo_tensor((hidden_size,hidden_size),1)) # Minimal config

    init_fn_mock = None
    if use_init_fn:
        init_fn_mock = lambda bs, hs, **kwargs: torch.ones(bs, hs) * 0.5
        rnn.init_fn = init_fn_mock # Assign to instance for this test

    h = rnn.init_hidden(batch_size)
    assert h.shape == (batch_size, hidden_size)
    if use_init_fn:
        assert torch.allclose(h, torch.ones(batch_size, hidden_size) * 0.5)
    else: # Default is zeros
        assert torch.allclose(h, torch.zeros(batch_size, hidden_size))

def test_sparse_rnn_clamp_connectivity():
    torch.manual_seed(8)
    hidden_size = 4
    # Create hh_values with some negative values
    indices_hh, values_hh_orig, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 6)
    values_hh_orig.data[0] = -0.5
    values_hh_orig.data[2] = -1.0

    rnn = SparseRNN(
        input_size=3, hidden_size=hidden_size,
        connectivity_hh=(indices_hh, values_hh_orig.clone().detach().requires_grad_(False), size_hh), # Pass non-trainable for this specific test
        train_hh_values=False # Ensure clamping happens even if not training (it's for stability)
    )

    # Clamping usually happens in forward. Let's run a dummy forward.
    x_dummy = torch.rand(1, 3) # batch_size=1, input_size=3
    rnn.forward(x_dummy)

    # Check if values in rnn.hh.weight_values (which should be the clamped ones) are non-negative
    # Note: _clamp_connectivity modifies the values in-place if they are parameters,
    # or re-assigns self.hh.weight_values if they are not.
    # If train_hh_values is False, weight_values is a buffer, not a parameter.
    assert torch.all(rnn.hh.weight_values >= 0)
    # Ensure some values were actually changed if they were negative
    if torch.any(values_hh_orig < 0): # If there were negatives to clamp
         assert torch.any(rnn.hh.weight_values[values_hh_orig < 0] == 0) # Check specific previously negative values are now 0


# --- SparseODERNN Tests ---

@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("use_connectivity_paths", [True, False])
@pytest.mark.parametrize("solver", ["rk4", "dopri5"])
@pytest.mark.parametrize("use_dense_ih", [True, False])
@pytest.mark.parametrize("compile_kwargs_present", [True, False]) # Test with and without compile kwargs
def test_sparse_odernn_initialization(
    mock_loader, use_connectivity_paths, solver, use_dense_ih, compile_kwargs_present
):
    torch.manual_seed(9)
    input_size = 4
    hidden_size = 5

    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 10)
    connectivity_hh_arg = MOCK_HH_PATH if use_connectivity_paths else (indices_hh, values_hh, size_hh)

    connectivity_ih_arg = None
    if not use_dense_ih:
        indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 8)
        connectivity_ih_arg = MOCK_IH_PATH if use_connectivity_paths else (indices_ih, values_ih, size_ih)

    compile_solver_kwargs = {"mode": "reduce-overhead"} if compile_kwargs_present else None
    compile_update_fn_kwargs = {"fullgraph": True} if compile_kwargs_present else None

    odernn = SparseODERNN(
        input_size=input_size,
        hidden_size=hidden_size,
        connectivity_hh=connectivity_hh_arg,
        connectivity_ih=connectivity_ih_arg,
        use_dense_ih=use_dense_ih,
        solver=solver,
        solver_options={"step_size": 0.1} if solver == "rk4" else None,
        compile_solver_kwargs=compile_solver_kwargs,
        compile_update_fn_kwargs=compile_update_fn_kwargs
    )

    assert odernn.input_size == input_size
    assert odernn.hidden_size == hidden_size
    assert odernn.solver == solver
    assert isinstance(odernn.hh, SparseLinear)
    if use_dense_ih:
        assert isinstance(odernn.ih, torch.nn.Linear)
    elif connectivity_ih_arg:
        assert isinstance(odernn.ih, SparseLinear)

    # Check if compile kwargs are stored (direct check of torch.compile is hard)
    if compile_kwargs_present:
        assert odernn.compile_solver_kwargs == compile_solver_kwargs
        assert odernn.compile_update_fn_kwargs == compile_update_fn_kwargs
    else:
        assert odernn.compile_solver_kwargs is None
        assert odernn.compile_update_fn_kwargs is None

    # Check if functions are callable (mocking torch.compile to see if it was called is complex)
    # For now, if initialization doesn't fail and attributes are set, assume it's okay.
    # If torch.compile fails with bad args, it would error here.
    assert callable(odernn.ode_func)
    # `solve_ivp` is not directly an attribute of the class, but used by `ode_solve`
    # `ode_solve` method itself is what might be compiled if `compile_solver_kwargs` is used.
    # We can't easily check if `self.ode_solve_compiled` was created without specific torch versions.


@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("provide_h0", [True, False])
@pytest.mark.parametrize("num_evals", [None, 3, 5]) # Test different num_evals
@pytest.mark.parametrize("input_is_3d", [True, False]) # Test 2D (single sequence) and 3D (batch) input
def test_sparse_odernn_forward(mock_loader, batch_first, provide_h0, num_evals, input_is_3d):
    torch.manual_seed(10)
    input_size = 3
    hidden_size = 4
    batch_size_test = 2
    num_input_steps = 1 if not input_is_3d else 5 # Num input time points if x is 3D

    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 8)
    indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 6)

    odernn = SparseODERNN(
        input_size=input_size, hidden_size=hidden_size,
        connectivity_hh=(indices_hh, values_hh, size_hh),
        connectivity_ih=(indices_ih, values_ih, size_ih),
        batch_first=batch_first,
        solver="rk4", solver_options={"step_size": 0.1} # Fixed step for predictability
    )
    odernn.eval()

    if input_is_3d:
        if batch_first:
            x = torch.rand(batch_size_test, num_input_steps, input_size)
        else:
            x = torch.rand(num_input_steps, batch_size_test, input_size)
    else: # 2D input (batch_size, features)
        x = torch.rand(batch_size_test, input_size)
        # batch_first doesn't apply to 2D input in the same way for sequence dim
        # SparseODERNN's forward expects 3D or 2D. If 2D, it's (batch, features) -> treated as 1 time step.

    h0_tensor = None
    if provide_h0:
        h0_tensor = torch.rand(batch_size_test, hidden_size)

    # Determine eval_times or rely on num_evals
    # For simplicity, let num_evals determine output points relative to input duration

    outs, hs, ts = odernn.forward(x, h0=h0_tensor, num_evals=num_evals)

    # Determine expected output sequence length
    if num_evals is not None:
        expected_out_seq_len = num_evals
    elif input_is_3d:
        expected_out_seq_len = num_input_steps
    else: # 2D input, num_evals=None -> defaults to 2 (start and end of a nominal 0-1 interval)
        expected_out_seq_len = 2

    # Check outs shape (output is always hidden state for SparseODERNN base)
    if batch_first or not input_is_3d : # If input was 2D, output becomes (batch, seq_out, hidden)
        assert outs.shape == (batch_size_test, expected_out_seq_len, hidden_size)
        assert ts.shape == (batch_size_test, expected_out_seq_len) # Timestamps per batch
    else: # input_is_3D and not batch_first
        assert outs.shape == (expected_out_seq_len, batch_size_test, hidden_size)
        assert ts.shape == (expected_out_seq_len,) # Timestamps are global

    # Check hs shape (final hidden state)
    assert hs.shape == (batch_size_test, hidden_size)


@patch('bioplnn.models.sparse.load_connectivity_from_path', side_effect=mock_load_connectivity)
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("train_flags", [[True,True], [False,False]]) # hh_values, ih_values
@pytest.mark.parametrize("solver", ["rk4", "dopri5"]) # Test adjoint for dopri5
def test_sparse_odernn_backward(mock_loader, batch_first, train_flags, solver):
    torch.manual_seed(11)
    input_size = 3
    hidden_size = 4
    batch_size_test = 1
    num_evals_test = 2 # Keep small for backward

    train_hh_v, train_ih_v = train_flags

    indices_hh, values_hh, size_hh = create_sparse_coo_tensor((hidden_size, hidden_size), 8, requires_grad_values=train_hh_v)
    indices_ih, values_ih, size_ih = create_sparse_coo_tensor((hidden_size, input_size), 5, requires_grad_values=train_ih_v)

    odernn = SparseODERNN(
        input_size=input_size, hidden_size=hidden_size,
        connectivity_hh=(indices_hh, values_hh, size_hh),
        connectivity_ih=(indices_ih, values_ih, size_ih),
        train_hh_values=train_hh_v, train_ih_values=train_ih_v,
        bias_hh=True, bias_ih=True, # Ensure biases for grad check
        batch_first=batch_first,
        solver=solver,
        solver_options={"step_size": 0.2} if solver == "rk4" else {"atol":1e-2, "rtol":1e-2} # Looser tol for faster test
    )

    # Input (2D for simplicity, treated as 1 time step by ODE forward)
    x = torch.rand(batch_size_test, input_size)

    outs, _, _ = odernn.forward(x, num_evals=num_evals_test)
    loss = outs.sum()
    loss.backward()

    # Check HH layer
    assert odernn.hh.weight_values.requires_grad == train_hh_v
    if train_hh_v:
        assert odernn.hh.weight_values.grad is not None; assert odernn.hh.weight_values.grad.abs().sum() > 1e-9
    if odernn.hh.bias is not None and odernn.hh.bias.requires_grad:
        assert odernn.hh.bias.grad is not None; assert odernn.hh.bias.grad.abs().sum() > 1e-9

    # Check IH layer (assuming SparseLinear IH)
    if isinstance(odernn.ih, SparseLinear):
        assert odernn.ih.weight_values.requires_grad == train_ih_v
        if train_ih_v:
            assert odernn.ih.weight_values.grad is not None; assert odernn.ih.weight_values.grad.abs().sum() > 1e-9
        if odernn.ih.bias is not None and odernn.ih.bias.requires_grad:
             assert odernn.ih.bias.grad is not None; assert odernn.ih.bias.grad.abs().sum() > 1e-9


def test_sparse_odernn_index_from_time():
    torch.manual_seed(12)
    odernn = SparseODERNN(input_size=1, hidden_size=2, connectivity_hh=create_sparse_coo_tensor((2,2),1)) # Minimal

    # Case 1: Single query time, unique times
    query_time = torch.tensor(0.5)
    times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    idx = odernn._index_from_time(query_time, times)
    assert idx.item() == 2

    # Case 2: Multiple query times, unique times
    query_times = torch.tensor([0.1, 0.6, 0.9])
    idx = odernn._index_from_time(query_times, times)
    assert torch.equal(idx, torch.tensor([0, 2, 3])) # 0.1 rounds to 0, 0.6 to 0.5 (idx 2), 0.9 to 0.75 (idx 3)

    # Case 3: Query time outside range (should clamp or error - current behavior is clamping to nearest)
    query_time_low = torch.tensor(-0.5)
    idx_low = odernn._index_from_time(query_time_low, times)
    assert idx_low.item() == 0

    query_time_high = torch.tensor(1.5)
    idx_high = odernn._index_from_time(query_time_high, times)
    assert idx_high.item() == 4

    # Case 4: Non-unique times (e.g. from batched eval_times where padding might occur or times repeat)
    # _index_from_time expects `times` to be sorted.
    # If batch_first=True, `times` in forward can be (batch, seq_len). _index_from_time might get one row.
    times_batched_row = torch.tensor([0.0, 0.5, 0.5, 1.0, 1.0]) # Example row
    query_time_b = torch.tensor(0.5)
    idx_b = odernn._index_from_time(query_time_b, times_batched_row)
    assert idx_b.item() == 1 # Should pick the first occurrence of 0.5

    query_times_b = torch.tensor([0.0, 1.0])
    idx_b_multi = odernn._index_from_time(query_times_b, times_batched_row)
    assert torch.equal(idx_b_multi, torch.tensor([0, 3]))

    # Case 5: query_time has more dims (should be handled by broadcasting in torch.abs(times - query_time.unsqueeze(-1)))
    query_times_unsqueeze = torch.tensor([[0.1], [0.6]]) # (2,1)
    idx_unsqueeze = odernn._index_from_time(query_times_unsqueeze, times) # times (5,)
    # Expected: [[0],[2]] based on broadcasting rules of subtraction then argmin
    assert torch.equal(idx_unsqueeze, torch.tensor([[0],[2]]))
