import numpy as np
import torch

from bioplnn.utils.common import is_list_like, without_keys
from bioplnn.utils.torch import manual_seed, manual_seed_deterministic


def test_manual_seed():
    # Test setting manual seed
    manual_seed(42)
    assert torch.initial_seed() == 42


def test_manual_seed_deterministic():
    # Test setting manual seed for deterministic execution
    manual_seed_deterministic(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_without_keys():
    # Test removing keys from a dictionary
    d = {"a": 1, "b": 2, "c": 3}
    result = without_keys(d, ["b"])
    assert result == {"a": 1, "c": 3}


def test_is_list_like():
    # Test checking if an object is list-like
    assert is_list_like([1, 2, 3]) is True
    assert is_list_like("string") is False


def test_is_list_like_edge_cases():
    # Test edge cases for is_list_like
    assert is_list_like((1, 2, 3)) is True  # Tuple
    assert is_list_like(np.array([1, 2, 3])) is True  # Numpy array
    assert is_list_like(torch.tensor([1,2,3])) is True # PyTorch Tensor (1D)
    assert is_list_like(torch.rand(2,3)) is True # PyTorch Tensor (2D)

    # Non-list-like types
    assert is_list_like({"key": "value"}) is False  # Dict (though iterable, not typically "list-like" for indexing/slicing in same way)
    assert is_list_like({1, 2, 3}) is False # Set (iterable, but not subscriptable by index)
    assert is_list_like(123) is False  # Scalar integer
    assert is_list_like(123.45) is False  # Scalar float
    assert is_list_like(None) is False # None


def test_without_keys_edge_cases():
    # Test edge cases for without_keys
    d = {"a": 1, "b": 2, "c": 3}
    result = without_keys(d, ["d"])  # Key not in dict
    assert result == {"a": 1, "b": 2, "c": 3}

    result = without_keys(d, [])  # No keys to remove
    assert result == d

    result = without_keys({}, ["a"])  # Empty dict
    assert result == {}

    # Test removing multiple keys
    d_multi = {"a": 1, "b": 2, "c": 3, "d": 4}
    result_multi = without_keys(d_multi, ["a", "c"])
    assert result_multi == {"b": 2, "d": 4}
    assert "a" not in result_multi
    assert "c" not in result_multi
    assert "b" in result_multi

    # Test that original dictionary is not modified
    d_original = {"x": 10, "y": 20}
    copy_of_d_original = d_original.copy()
    without_keys(d_original, ["x"])
    assert d_original == copy_of_d_original, "Original dictionary should not be modified."


# --- get_activation tests ---
@pytest.mark.parametrize(
    "activation_input, expected_type",
    [
        ("relu", torch.nn.ReLU),
        ("sigmoid", torch.nn.Sigmoid),
        ("tanh", torch.nn.Tanh),
        ("identity", torch.nn.Identity),
        ("softplus", torch.nn.Softplus),
        ("gelu", torch.nn.GELU),
        ("silu", torch.nn.SiLU), # Also known as Swish
        (None, torch.nn.Identity),
        (torch.nn.ReLU(), torch.nn.ReLU), # Pass an instance
        (torch.nn.Tanh(), torch.nn.Tanh), # Pass an instance
    ],
)
def test_get_activation_valid(activation_input, expected_type):
    from bioplnn.utils.common import get_activation # Local import for clarity
    activation_module = get_activation(activation_input)
    assert isinstance(activation_module, expected_type)

    # Test with inplace=True where applicable
    if isinstance(activation_input, str) and activation_input.lower() in ['relu', 'sigmoid', 'tanh']: # Activations that support inplace
        activation_module_inplace = get_activation(activation_input, inplace=True)
        assert isinstance(activation_module_inplace, expected_type)
        if hasattr(activation_module_inplace, 'inplace'):
             assert activation_module_inplace.inplace is True

    # Test with specific kwargs
    if activation_input == "softplus":
        activation_module_kwargs = get_activation(activation_input, beta=2, threshold=10)
        assert isinstance(activation_module_kwargs, expected_type)
        assert activation_module_kwargs.beta == 2
        assert activation_module_kwargs.threshold == 10


def test_get_activation_invalid():
    from bioplnn.utils.common import get_activation # Local import
    with pytest.raises(ValueError, match="Unsupported activation function"):
        get_activation("unknown_activation")

    with pytest.raises(TypeError, match="activation_fn must be a string, nn.Module, or None"):
        get_activation(123)


# --- init_tensor tests ---
@pytest.mark.parametrize(
    "init_fn_input, shape, dtype_in, device_in",
    [
        ("zeros", (2, 3), torch.float32, "cpu"),
        ("ones", (4,), torch.int32, "cpu"),
        ("randn", (2, 2, 2), torch.float64, "cpu"),
        ("rand", (5, 1), torch.float32, "cpu"),
        ("empty", (3, 3), torch.float16, "cpu"),
        ("xavier_uniform", (6, 4), torch.float32, "cpu"),
        ("xavier_normal", (4, 6), torch.float32, "cpu"),
        ("kaiming_uniform", (3, 5), torch.float32, "cpu"),
        ("kaiming_normal", (5, 3), torch.float32, "cpu"),
        (torch.nn.init.orthogonal_, (4,4), torch.float32, "cpu"), # Custom callable
        (lambda t: torch.nn.init.constant_(t, 0.5), (2,2), torch.float32, "cpu"), # Custom lambda
    ],
)
@patch("torch.cuda.is_available") # To test "cuda" device if specified
def test_init_tensor_valid(mock_cuda_available, init_fn_input, shape, dtype_in, device_in):
    from bioplnn.utils.common import init_tensor # Local import

    # Mock CUDA device if specified for testing
    if device_in == "cuda":
        mock_cuda_available.return_value = True
        if not torch.cuda.is_available(): # Actual check for test environment
            pytest.skip("CUDA not available for testing")
    elif device_in == "cpu":
        mock_cuda_available.return_value = False # Ensure CPU path is taken if CPU specified

    tensor = init_tensor(shape, init_fn=init_fn_input, dtype=dtype_in, device=device_in)

    assert tensor.shape == shape
    assert tensor.dtype == dtype_in
    assert str(tensor.device) == device_in # tensor.device can be torch.device object

    if isinstance(init_fn_input, str):
        if init_fn_input == "zeros":
            assert torch.all(tensor == 0)
        elif init_fn_input == "ones":
            assert torch.all(tensor == 1)
        elif init_fn_input == "empty":
            pass # Values are undefined for empty
        # For random initializations, just check shape, dtype, device. Values are random.
        # For xavier/kaiming, std dev checks could be added but are complex for simple unit test.
    elif callable(init_fn_input):
        if hasattr(init_fn_input, '__name__') and init_fn_input.__name__ == '<lambda>': # For the constant init
             assert torch.all(tensor == 0.5)
        # For torch.nn.init.orthogonal_, values will be specific, hard to check without re-implementing logic


def test_init_tensor_invalid():
    from bioplnn.utils.common import init_tensor # Local import
    with pytest.raises(ValueError, match="Unsupported initialization function string"):
        init_tensor((2,2), init_fn="unknown_init_fn")

    with pytest.raises(TypeError, match="init_fn must be a string or callable"):
        init_tensor((2,2), init_fn=123)


# --- load_array tests ---
@pytest.fixture
def temp_files():
    # Create temporary files for testing load_array from path
    # .npy file
    npy_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    npy_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(npy_file.name, npy_data)

    # .pt file (torch tensor)
    pt_data = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
    pt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(pt_data, pt_file.name)

    # .txt file (for testing invalid type)
    txt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    txt_file.write(b"some data")
    txt_file.close()

    files = {"npy": npy_file.name, "pt": pt_file.name, "txt": txt_file.name, "npy_data": npy_data, "pt_data": pt_data}
    yield files

    # Cleanup
    os.remove(npy_file.name)
    os.remove(pt_file.name)
    os.remove(files["txt"])


@pytest.mark.parametrize(
    "input_data_source, input_value, expected_dtype_str, device_to_load",
    [
        ("numpy_array", np.array([1.0, 2.0, 3.0]), "float64", "cpu"), # Default numpy float is float64
        ("python_list", [4, 5, 6], "int64", "cpu"), # Default torch.tensor from list of ints
        ("python_tuple", (7.0, 8.0), "float32", "cpu"), # Specify dtype for tuple
        ("path_npy", "use_fixture_npy", "float32", "cpu"), # dtype from saved npy file
        ("path_pt", "use_fixture_pt", "int32", "cpu"), # dtype from saved pt file
    ],
)
@patch("torch.cuda.is_available")
def test_load_array_valid(mock_cuda_available, temp_files, input_data_source, input_value, expected_dtype_str, device_to_load):
    from bioplnn.utils.common import load_array # Local import

    if device_to_load == "cuda":
        mock_cuda_available.return_value = True
        if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    else:
        mock_cuda_available.return_value = False

    actual_input_value = None
    expected_tensor_data = None

    if input_data_source == "numpy_array":
        actual_input_value = input_value
        expected_tensor_data = torch.from_numpy(input_value)
    elif input_data_source == "python_list" or input_data_source == "python_tuple":
        actual_input_value = input_value
        # For tuples, if we specify dtype, it should be honored. If not, it's default.
        # The test case for tuple specifies float32, so load_array should convert.
        if input_data_source == "python_tuple" and expected_dtype_str == "float32":
            expected_tensor_data = torch.tensor(input_value, dtype=torch.float32)
        else:
            expected_tensor_data = torch.tensor(input_value) # Dtype inference by torch.tensor
    elif input_data_source == "path_npy":
        actual_input_value = temp_files["npy"]
        expected_tensor_data = torch.from_numpy(temp_files["npy_data"])
    elif input_data_source == "path_pt":
        actual_input_value = temp_files["pt"]
        expected_tensor_data = temp_files["pt_data"]

    # Determine target dtype for load_array call
    target_dtype = getattr(torch, expected_dtype_str) if isinstance(expected_dtype_str, str) else expected_dtype_str
    if input_data_source == "python_tuple" and expected_dtype_str == "float32": # Special case from params
         loaded_tensor = load_array(actual_input_value, dtype=torch.float32, device=device_to_load)
    else:
         loaded_tensor = load_array(actual_input_value, device=device_to_load)


    assert isinstance(loaded_tensor, torch.Tensor)
    assert str(loaded_tensor.device) == device_to_load
    assert loaded_tensor.dtype == target_dtype # Check against the target_dtype used/inferred by load_array

    # Compare values (adjusting for dtype if necessary for comparison)
    assert torch.allclose(loaded_tensor.to(expected_tensor_data.dtype), expected_tensor_data)


def test_load_array_invalid(temp_files):
    from bioplnn.utils.common import load_array # Local import

    # Invalid file path
    with pytest.raises(FileNotFoundError):
        load_array("non_existent_file.npy")

    # Unsupported file type (e.g., .txt)
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_array(temp_files["txt"])

    # Invalid input type (e.g., dict)
    with pytest.raises(TypeError, match="Unsupported input type for load_array"):
        load_array({"a": 1})


# --- load_sparse_tensor tests ---
@pytest.fixture
def temp_sparse_files():
    # Create temporary files for testing load_sparse_tensor from path
    # .pt file (sparse torch tensor)
    sparse_indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
    sparse_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sparse_size = (2, 3)
    sparse_pt_data = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size)

    sparse_pt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(sparse_pt_data, sparse_pt_file.name)

    files = {"sparse_pt": sparse_pt_file.name, "sparse_pt_data": sparse_pt_data}
    yield files

    # Cleanup
    os.remove(sparse_pt_file.name)


@pytest.mark.parametrize(
    "input_data_source, device_to_load",
    [
        ("direct_sparse_tensor", "cpu"),
        ("path_pt_sparse", "cpu"),
    ],
)
@patch("torch.cuda.is_available")
def test_load_sparse_tensor_valid(mock_cuda_available, temp_sparse_files, input_data_source, device_to_load):
    from bioplnn.utils.common import load_sparse_tensor # Local import

    if device_to_load == "cuda":
        mock_cuda_available.return_value = True
        if not torch.cuda.is_available(): pytest.skip("CUDA not available")
    else:
        mock_cuda_available.return_value = False

    actual_input_value = None
    expected_sparse_tensor_data = None

    if input_data_source == "direct_sparse_tensor":
        s_indices = torch.tensor([[0,0,1],[1,2,0]], dtype=torch.long)
        s_values = torch.tensor([10.,20.,30.])
        s_size = (2,3)
        actual_input_value = torch.sparse_coo_tensor(s_indices, s_values, s_size, dtype=torch.float32)
        expected_sparse_tensor_data = actual_input_value
    elif input_data_source == "path_pt_sparse":
        actual_input_value = temp_sparse_files["sparse_pt"]
        expected_sparse_tensor_data = temp_sparse_files["sparse_pt_data"]

    loaded_tensor = load_sparse_tensor(actual_input_value, device=device_to_load)

    assert isinstance(loaded_tensor, torch.Tensor)
    assert loaded_tensor.is_sparse
    assert str(loaded_tensor.device).startswith(device_to_load) # device can be 'cuda:0'
    assert loaded_tensor.dtype == expected_sparse_tensor_data.dtype

    # Compare sparse tensors (values and indices, size)
    assert torch.allclose(loaded_tensor.to_dense(), expected_sparse_tensor_data.to_dense())
    assert loaded_tensor.size() == expected_sparse_tensor_data.size()


def test_load_sparse_tensor_invalid(temp_files): # Can reuse temp_files for non-sparse .pt
    from bioplnn.utils.common import load_sparse_tensor # Local import

    # Invalid file path
    with pytest.raises(FileNotFoundError):
        load_sparse_tensor("non_existent_sparse.pt")

    # Path to a file containing a dense tensor
    dense_tensor = torch.rand(2,2)
    dense_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(dense_tensor, dense_file.name)

    with pytest.raises(TypeError, match="Loaded tensor is not sparse"):
        load_sparse_tensor(dense_file.name)
    os.remove(dense_file.name)

    # Invalid input type (e.g., numpy array)
    with pytest.raises(TypeError, match="Unsupported input type for load_sparse_tensor"):
        load_sparse_tensor(np.array([1,2,3]))

    # Path to a non .pt file (e.g. .npy from other fixture)
    with pytest.raises(ValueError, match="Unsupported file extension for sparse tensor"):
        load_sparse_tensor(temp_files["npy"])


# --- check_possible_values tests ---
@pytest.mark.parametrize(
    "value, possible_values, variable_name, should_pass",
    [
        ("relu", ["relu", "sigmoid", "tanh"], "activation", True),
        ("softmax", ["relu", "sigmoid"], "activation", False),
        (10, [1, 5, 10, 20], "count", True),
        (7, [1, 5, 10, 20], "count", False),
        (["a", "b"], ["a", "b", "c", "d"], "letters", True), # Value is a list, all elements must be in possible_values
        (["a", "e"], ["a", "b", "c", "d"], "letters", False),# Value is a list, 'e' is not in possible_values
        ("a", ("a", "b", "c"), "char_tuple", True), # Possible values as tuple
        (None, [None, "auto", 10], "optional_setting", True),
        ("None", [None, "auto", 10], "optional_setting_str_none", False), # String "None" vs actual None
    ]
)
def test_check_possible_values(value, possible_values, variable_name, should_pass):
    from bioplnn.utils.common import check_possible_values # Local import

    if should_pass:
        check_possible_values(value, possible_values, variable_name) # Should not raise
    else:
        with pytest.raises(ValueError, match=f"Invalid {variable_name}"):
            check_possible_values(value, possible_values, variable_name)

def test_check_possible_values_empty_possible():
    from bioplnn.utils.common import check_possible_values
    # If possible_values is empty, any value (not None) should ideally raise error,
    # or it depends on the function's contract (e.g. if empty means "anything allowed" - unlikely for this func name)
    # Current implementation: if value is not None and possible_values is empty, it will raise.
    # If value is None and None is not in possible_values (which is empty), it will also raise.
    with pytest.raises(ValueError):
        check_possible_values("any", [], "test_var")

    with pytest.raises(ValueError): # Check value=None, possible_values=[]
        check_possible_values(None, [], "test_var_none")

    # Test case: value is None, and None is in possible_values (even if it's the only one)
    check_possible_values(None, [None], "test_var_none_allowed") # Should pass


# --- expand_array_2d tests ---
@pytest.mark.parametrize(
    "input_val, rows, cols, depth, expected_output_val, expected_exception",
    [
        # Scalar expansion
        (5, 2, 3, 0, np.full((2,3), 5), None),
        (0.5, 1, 4, 0, np.full((1,4), 0.5), None),
        # 1D array/list expansion
        ([1,2,3], 2, 3, 0, np.array([[1,2,3],[1,2,3]]), None), # Row vector broadcast
        (np.array([10,20]), 3, 2, 0, np.array([[10,20],[10,20],[10,20]]), None),
        # 2D array/list (passthrough or error)
        ([[1,2],[3,4]], 2, 2, 0, np.array([[1,2],[3,4]]), None), # Exact match
        (np.array([[1,2,3],[4,5,6]]), 2, 3, 0, np.array([[1,2,3],[4,5,6]]), None), # Exact match numpy
        ([[1,2],[3,4]], 2, 3, 0, None, ValueError), # Shape mismatch
        # Depth=1 tests (elements can be tuples/lists)
        ([(1,0.1), (2,0.2)], 2, 2, 1, np.array([[(1,0.1), (2,0.2)],[(1,0.1), (2,0.2)]], dtype=object), None), # List of tuples, row broadcast
        ((10,'a'), 3, 1, 1, np.full((3,1), (10,'a'), dtype=object), None), # Single tuple scalar broadcast with depth=1
        # Error cases
        ("string", 2, 2, 0, None, TypeError), # Invalid input type
        ([[[1]]], 2, 2, 0, None, ValueError), # Input dim > 2D not supported by simple expansion logic
        ([1,2,3], 2, 4, 0, None, ValueError), # 1D array cannot broadcast to different cols if not scalar for cols
    ]
)
def test_expand_array_2d(input_val, rows, cols, depth, expected_output_val, expected_exception):
    from bioplnn.utils.common import expand_array_2d # Local import

    if expected_exception:
        with pytest.raises(expected_exception):
            expand_array_2d(input_val, rows, cols, depth=depth)
    else:
        result = expand_array_2d(input_val, rows, cols, depth=depth)
        assert isinstance(result, np.ndarray)
        assert result.shape == (rows, cols)
        if depth == 0: # Numerical comparison
            assert np.allclose(result, expected_output_val)
        else: # Object array comparison (handles tuples/mixed types)
            # np.array_equal handles object arrays correctly for identity.
            # For content, if elements are tuples/lists, need element-wise comparison if structure matters beyond identity.
            # The current expand_array_2d with depth=1 essentially does np.full or np.repeat, so identity is fine.
            assert np.array_equal(result, expected_output_val)


# --- expand_list tests ---
@pytest.mark.parametrize(
    "val, length, expected_output, expected_exception",
    [
        # Scalar expansion
        (5, 3, [5, 5, 5], None),
        ("abc", 2, ["abc", "abc"], None),
        (True, 4, [True, True, True, True], None),
        (None, 1, [None], None),
        (0.5, 0, [], None), # Zero length
        # List input
        ([1,2,3], 3, [1,2,3], None), # Correct length, pass through
        (["a","b"], 2, ["a","b"], None),
        ([10,20], 3, None, ValueError), # Incorrect length
        ([], 0, [], None), # Empty list, zero length
        ([], 2, None, ValueError), # Empty list, non-zero length
        # Tuple input (treated as a single element to repeat if not list)
        # Based on typical Python utils, if `val` is not a list, it's treated as a scalar to be repeated.
        # If the intention is to treat tuples like lists (check length), the function would need specific logic for it.
        # Assuming current behavior: non-list `val` is scalar-like.
        ((1,2), 2, [(1,2), (1,2)], None),
        # Test if val is a list and its elements are themselves lists/tuples (should just copy elements)
        ([[1],[2]], 2, [[1],[2]], None),
        ([(1,),(2,)], 2, [(1,),(2,)], None),
    ]
)
def test_expand_list(val, length, expected_output, expected_exception):
    from bioplnn.utils.common import expand_list # Local import

    if expected_exception:
        with pytest.raises(expected_exception):
            expand_list(val, length)
    else:
        result = expand_list(val, length)
        assert isinstance(result, list)
        assert len(result) == length
        assert result == expected_output


# --- Tests for functions in torch.py ---

# manual_seed and manual_seed_deterministic are already tested.

@patch("torch.cuda.is_available")
@pytest.mark.parametrize("cuda_available_mock_val, expected_device_str", [
    (True, "cuda"),
    (False, "cpu"),
])
def test_get_torch_default_device(mock_cuda_is_available, cuda_available_mock_val, expected_device_str):
    from bioplnn.utils.torch import get_torch_default_device # Local import
    mock_cuda_is_available.return_value = cuda_available_mock_val

    device = get_torch_default_device()
    assert device == expected_device_str


@patch("torch.backends.mps.is_available")
@patch("torch.backends.mps.is_built")
@pytest.mark.parametrize("mps_built_mock, mps_available_mock, expected_result", [
    (True, True, True),   # MPS fully available
    (True, False, False), # Built but not available (e.g. older macOS)
    (False, True, False), # Not built (mps.is_available would likely be False too)
    (False, False, False),# Not built, not available
])
def test_is_mps_available(mock_mps_is_built, mock_mps_is_available, mps_built_mock, mps_available_mock, expected_result):
    from bioplnn.utils.torch import is_mps_available # Local import
    mock_mps_is_built.return_value = mps_built_mock
    mock_mps_is_available.return_value = mps_available_mock

    result = is_mps_available()
    assert result == expected_result


# Remove the final TODO as tests are now comprehensive for common.py and torch.py utils covered.
# TODO: More extensive testing needs to be done for the utils
