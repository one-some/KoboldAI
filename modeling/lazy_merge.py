import gc
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch

from modeling.virtual_model import PretendModel


class BadMergeRequestError(Exception):
    """Raised when the user requests a model merge configuration the backend
    thinks is lame"""

    pass


class MergeStrategy(Enum):
    LINEAR = 0


@dataclass
class ModelProportion:
    name: str
    weight: float
    virtual_model: Optional[PretendModel] = None


class parameters:
    enable: bool = True

    # I don't like this way of representing model proportions because it leaves
    # the primary model's weight ambigious for people who struggle with 1st
    # grade math (me)
    secondary_models: List[ModelProportion] = [
        ModelProportion("models/EleutherAI_gpt-j-6B", weight=0.5),
    ]
    main_model_weight: Optional[float] = None
    strategy: MergeStrategy = MergeStrategy.LINEAR

def parse_arg(args: Namespace) -> None:
    pass


def start_load() -> None:
    total_weights_excluding_main_model = sum(
        [m.weight for m in parameters.secondary_models]
    )

    if total_weights_excluding_main_model <= 0.0:
        raise BadMergeRequestError(
            "Primary model has all the weight. That's hardly a 'merge', eh?"
        )
    elif total_weights_excluding_main_model >= 1.0:
        raise BadMergeRequestError(
            "Secondary models cumulative weight equals or exceeds 1.0! Give the primary model some!"
        )

    for model in parameters.secondary_models:
        if model.weight < 0.0:
            raise BadMergeRequestError(f"Model {model.name} has negative weight.")
        elif model.weight == 0.0:
            raise BadMergeRequestError(f"Model {model.name} has zero weight.")
        elif model.weight >= 1.0:
            raise BadMergeRequestError(
                f"Model {model.name} has weight exceeding or equaling 1."
            )

        # Actually load the virtual models; slowest part besides materializing,
        # but still isn't too slow.
        print(f"[merge] Loading virtual model {model.name}")
        model.virtual_model = PretendModel(model.name)

    parameters.main_model_weight = 1.0 - total_weights_excluding_main_model

    # Dump info
    print(f"[main]\t{parameters.main_model_weight}")
    for model in parameters.secondary_models:
        print(f"[{model.name}]\t{model.weight}")


def post_load_cleanup() -> None:
    parameters.secondary_models.clear()
    gc.collect()


def merge_with_secondary_models(
    tensor_name: str, primary_tensor: torch.Tensor
) -> torch.Tensor:
    if parameters.strategy == MergeStrategy.LINEAR:
        ret = primary_tensor * parameters.main_model_weight
        for model in parameters.secondary_models:
            ret += (
                model.virtual_model.lazy_tensors[tensor_name].materialize(
                    map_location="cpu"
                )
                * model.weight
            )
        return ret
    raise NotImplementedError(f"No implementation for strategy '{parameters.strategy}'")
