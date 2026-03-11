import copy
import pickle

import torch

from bioplnn.base_model_configs import BASE_MODEL_CONFIGS
from bioplnn.models.classifiers import SpatiallyEmbeddedClassifier
from bioplnn.models.spatially_embedded import SpatiallyEmbeddedAreaConfig


def test_td_h_bioplnn_config_roundtrip(tmp_path):
    # Start from preset and ensure required input channels are set
    base_cfg = copy.deepcopy(BASE_MODEL_CONFIGS["td+h-bioplnn"])
    rnn_kwargs = copy.deepcopy(base_cfg["rnn_kwargs"])
    area_cfgs = rnn_kwargs["area_kwargs"]
    if area_cfgs[0].get("in_channels") is None:
        area_cfgs[0]["in_channels"] = 3
    base_cfg["rnn_kwargs"] = rnn_kwargs

    model1 = SpatiallyEmbeddedClassifier(**base_cfg)

    # Save weights and config
    state_path = tmp_path / "model.pt"
    cfg_path = tmp_path / "config.pkl"
    torch.save(model1.state_dict(), state_path)
    with open(cfg_path, "wb") as f:
        pickle.dump(model1.get_config(), f)

    # Reload config and rebuild model
    with open(cfg_path, "rb") as f:
        loaded_cfg = pickle.load(f)

    loaded_rnn_kwargs = loaded_cfg["rnn_kwargs"]
    loaded_area_cfgs = [
        SpatiallyEmbeddedAreaConfig(**area_cfg)
        for area_cfg in loaded_rnn_kwargs.pop("area_configs")
    ]
    loaded_rnn_kwargs["area_configs"] = loaded_area_cfgs

    model2 = SpatiallyEmbeddedClassifier(
        rnn_kwargs=loaded_rnn_kwargs,
        num_classes=loaded_cfg["num_classes"],
        pool_size_classifier=tuple(loaded_cfg["pool_size_classifier"]),
        pool_mode_classifier=loaded_cfg["pool_mode_classifier"],
        fc_dim=loaded_cfg["fc_dim"],
        dropout=loaded_cfg["dropout"],
        output_area_index=loaded_cfg["output_area_index"],
    )
    model2.load_state_dict(torch.load(state_path, map_location="cpu"))

    # Weights should match exactly after roundtrip
    for name, param in model1.state_dict().items():
        torch.testing.assert_close(param, model2.state_dict()[name])