from torch import nn
from transformers import PreTrainedModel


def get_transformer_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the `nn.ModuleList` containing the transformer layers in a model."""
    assert not model.config.is_encoder_decoder, "Encoder-decoder models not supported"

    lists = [mod for mod in model.modules() if isinstance(mod, nn.ModuleList)]
    if not lists:
        raise ValueError("Could not find transformer layers")

    # Return the module list with the most parameters
    return max(lists, key=lambda mod: sum(p.numel() for p in mod.parameters()))
