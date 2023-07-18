from __future__ import annotations
import gc

import os
from typing import Dict

import transformers
from transformers import AutoModelForCausalLM

from modeling import lazy_loader


class pretend_load:
    def __init__(self, pretend_model: PretendModel) -> None:
        self.pretend_model = pretend_model

        # We need to grab this at calltime, as it could have been altered by
        # LazyloadPatches (patches.py)
        self._unpatched = transformers.modeling_utils._load_state_dict_into_meta_model

    def __enter__(self) -> None:
        def _cache_lazy_tensors(model, state_dict, *args, **kwargs):
            # BEWARE: This will run multiple times in some cases (such as sharded models)
            for tensor_name, lazy_tensor in state_dict.items():
                self.pretend_model.lazy_tensors[tensor_name] = lazy_tensor
            return [], None, None

        # `_load_state_dict_into_meta_model` chosen for easy access to state_dict,
        # and I'm already familiar with it. There may be a higher-level function
        # called earlier in the callstack which could potentially reduce memory overhead.
        transformers.modeling_utils._load_state_dict_into_meta_model = (
            _cache_lazy_tensors
        )

        # Don't confuse the user with progress bars for fake models!
        transformers.utils.logging.disable_progress_bar()

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        transformers.modeling_utils._load_state_dict_into_meta_model = self._unpatched
        transformers.utils.logging.enable_progress_bar()


class PretendModel:
    """A virtual model that does not load any weights into memory, but instead
    populates `lazy_tensors` as a dictionary of tensor keys to `LazyTensor`s."""

    # Memory usage doesn't fluctuate terribly when indexing multiple models,
    # but there is an initial spike of around 200mb. Probably caching or
    # something on the Torch/HF end. Actually storing the lazytensors takes
    # like 3mb, so no concerns there

    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.basename(path)
        self.lazy_tensors: Dict[str, lazy_loader.LazyTensor] = {}

        # Dematerializing modules actually uses like 100mb more ram (6b, gpt-j) for some reason
        with lazy_loader.use_lazy_load(
            dematerialized_modules=False, caching_virtual_model=True
        ), pretend_load(self):
            # Use meta device map to trigger _load_state_dict_into_meta_model
            # because I'm too lazy to find the non-offload function to patch
            try:
                AutoModelForCausalLM.from_pretrained(
                    self.path,
                    device_map="meta",
                    cache_dir="cache",
                )
            except ValueError as e:
                # Error happens because weights have no value but whatever because we don't
                # want them to
                if "is on the meta device, we need a `value`" not in str(e):
                    raise e
        gc.collect()
