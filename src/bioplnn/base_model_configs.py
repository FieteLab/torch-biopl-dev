import numpy as np

# Centralized base model presets used by training scripts

BASE_MODEL_CONFIGS = {
    "1e1i1a": {
        "rnn_kwargs": {
            "num_areas": 1,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "num_neuron_subtypes": np.array([32, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 1, 1], [1, 0, 0]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.2,
    },
    "1e1ii1ef1a": {
        "rnn_kwargs": {
            "num_areas": 2,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Sigmoid",
                    "num_neuron_subtypes": np.array([32, 16]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "feedback_channels": 32,
                    "in_channels": None,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 1,
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Sigmoid",
                    "num_neuron_subtypes": np.array([32]),
                    "neuron_type_class": np.array(["excitatory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0], [1, 1]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": None,
                    "out_channels": 8,
                    "inter_neuron_type_nonlinearity": np.array([[None, None], [None, None]]),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
            "inter_area_feedback_connectivity": np.array([[0, 0], [1, 0]]),
            "inter_area_feedback_nonlinearity": np.array([[None, None], [None, None]]),
            "inter_area_feedback_spatial_extents": (5, 5),
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
    },
    "td+h-bioplnn": {
        "rnn_kwargs": {
            "num_areas": 3,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "in_class": "hybrid",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([20, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [256, 256],
                    "in_channels": None,
                    "out_channels": 32,
                    "feedback_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (7, 7),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "hybrid",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([32, 32]),
                    "neuron_type_class": np.array(["excitatory", "excitatory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
                    ),
                    "in_size": [128, 128],
                    "feedback_channels": 128,
                    "in_channels": 32,
                    "out_channels": 128,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 1,
                    "in_class": "hybrid",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([128]),
                    "neuron_type_class": np.array(["excitatory"]),
                    "inter_neuron_type_connectivity": np.array([[1, 0], [1, 1]]),
                    "in_size": [64, 64],
                    "in_channels": 128,
                    "out_channels": 128,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None], [None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (3, 3),
                },
            ],
            "inter_area_feedback_connectivity": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            "inter_area_feedback_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
            "inter_area_feedback_spatial_extents": (3, 3),
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
        "output_area_index": 0,
    },
    "ei_i_i_f": {
        "rnn_kwargs": {
            "num_areas": 3,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([16, 16]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": None,
                    "out_channels": 32,
                    "feedback_channels": 16,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([32]),
                    "neuron_type_class": np.array(["excitatory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
                    ),
                    "in_size": [48, 48],
                    "feedback_channels": 32,
                    "in_channels": None,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (7, 7),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([64]),
                    "neuron_type_class": np.array(["excitatory"]),
                    "inter_neuron_type_connectivity": np.array([[1, 1, 0], [1, 1, 1]]),
                    "in_size": [48, 48],
                    "in_channels": 32,
                    "out_channels": 64,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (9, 9),
                },
            ],
            "inter_area_feedback_connectivity": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
            "inter_area_feedback_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
            "inter_area_feedback_spatial_extents": (9, 9),
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
    },
    "ei_ei_ei_f": {
        "rnn_kwargs": {
            "num_areas": 3,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([16, 16]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": None,
                    "out_channels": 32,
                    "feedback_channels": 16,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([32, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "feedback_channels": 32,
                    "in_channels": None,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (7, 7),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Identity",
                    "num_neuron_subtypes": np.array([64, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                    "in_size": [48, 48],
                    "in_channels": 32,
                    "out_channels": 64,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (9, 9),
                },
            ],
            "inter_area_feedback_connectivity": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
            "inter_area_feedback_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
            "inter_area_feedback_spatial_extents": (9, 9),
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
    },
    "ei_ei_f": {
        "rnn_kwargs": {
            "num_areas": 2,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Sigmoid",
                    "num_neuron_subtypes": np.array([8, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "feedback_channels": 32,
                    "in_channels": None,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 2,
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Sigmoid",
                    "num_neuron_subtypes": np.array([32, 8]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]]),
                    "in_size": [48, 48],
                    "in_channels": 32,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array([[None, None, None], [None, None, None], [None, None, None]]),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
            "inter_area_feedback_connectivity": np.array([[0, 0], [1, 0]]),
            "inter_area_feedback_nonlinearity": np.array([[None, None], [None, None]]),
            "inter_area_feedback_spatial_extents": (5, 5),
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
    },
    "1e1ii1a": {
        "rnn_kwargs": {
            "num_areas": 1,
            "area_kwargs": [
                {
                    "in_class": "excitatory",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "ReLU",
                    "num_neuron_types": 2,
                    "num_neuron_subtypes": np.array([6, 4]),
                    "neuron_type_class": np.array(["excitatory", "inhibitory"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 8,
                    "inter_neuron_type_nonlinearity": np.array(
                        [[None, None, None], [None, None, None], [None, None, None]]
                    ),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.2,
    },
    "hybrid": {
        "rnn_kwargs": {
            "num_areas": 1,
            "area_kwargs": [
                {
                    "in_class": "hybrid",
                    "neuron_type_nonlinearity": "ReLU",
                    "out_nonlinearity": "Sigmoid",
                    "num_neuron_types": 1,
                    "num_neuron_subtypes": np.array([12]),
                    "neuron_type_class": np.array(["hybrid"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0], [1, 1]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 16,
                    "inter_neuron_type_nonlinearity": np.array([[None, None], [None, None]]),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 128,
        "dropout": 0.2,
    },
    "1h1a": {
        "rnn_kwargs": {
            "num_areas": 1,
            "area_kwargs": [
                {
                    "num_neuron_types": 1,
                    "num_neuron_subtypes": np.array([16]),
                    "neuron_type_class": np.array(["hybrid"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0], [1, 1]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 32,
                    "inter_neuron_type_nonlinearity": np.array([["sigmoid", "sigmoid"], ["sigmoid", "sigmoid"]]),
                    "inter_neuron_type_spatial_extents": (3, 3),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 64,
        "dropout": 0.2,
    },
    "hybrid_ei": {
        "rnn_kwargs": {
            "num_areas": 1,
            "area_kwargs": [
                {
                    "num_neuron_types": 2,
                    "num_neuron_subtypes": np.array([64, 64]),
                    "neuron_type_class": np.array(["hybrid", "hybrid"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 1, 0], [1, 1, 1], [1, 1, 0]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 16,
                    "inter_neuron_type_nonlinearity": np.array(
                        [["relu", "relu", "relu"], ["relu", "relu", "relu"], ["relu", "relu", "relu"]]
                    ),
                    "inter_neuron_type_spatial_extents": (3, 3),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 64,
        "dropout": 0.1,
    },
    "bio_cnn_hybrid": {
        "rnn_kwargs": {
            "num_areas": 2,
            "area_kwargs": [
                {
                    "num_neuron_types": 1,
                    "num_neuron_subtypes": np.array([64]),
                    "neuron_type_class": np.array(["hybrid"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0], [0, 1]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 4,
                    "out_channels": 128,
                    "inter_neuron_type_nonlinearity": np.array([["relu", "relu"], ["relu", "relu"]]),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
                {
                    "num_neuron_types": 1,
                    "num_neuron_subtypes": np.array([256]),
                    "neuron_type_class": np.array(["hybrid"]),
                    "inter_neuron_type_connectivity": np.array(
                        [[1, 0], [0, 1]]
                    ),
                    "in_size": [48, 48],
                    "in_channels": 128,
                    "out_channels": 256,
                    "inter_neuron_type_nonlinearity": np.array([["relu", "relu"], ["relu", "relu"]]),
                    "inter_neuron_type_spatial_extents": (5, 5),
                },
            ],
        },
        "num_classes": 2,
        "fc_dim": 512,
        "dropout": 0.1,
    },
    "cnn": {
        "in_channels": 4,
        "num_classes": 2,
        "dropout": 0.35,
        "conv1_out": 64,
        "conv2_out": 64,
        "conv3_out": 128,
        "fc_dim": 512,
        "conv_kernel_size": 5,
        "pool_kernel": 2,
        "pool_stride": 2,
    },
    "tiny_cnn": {
        "in_channels": 4,
        "num_classes": 2,
        "dropout": 0.1,
    },
}


