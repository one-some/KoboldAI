from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

model_backend_name = "mlc"
model_backend_type = "mlc"

class model_backend(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.model_name = "MLC"

        # self.capabilties = ModelCapabilities(api_host=False)

    def is_valid(self, model_name, model_path, menu_path):
        print(model_name)
        return True

    def get_requested_parameters(
        self, model_name, model_path, menu_path, parameters={}
    ):
        self.model_path = model_name
        return []

    def set_input_parameters(self, parameters):
        pass

    def _load(self, save_model: bool, initial_load: bool) -> None:
        from mlc_chat import ChatModule
        from mlc_chat import chat_module

        def _path(model, model_path, *args, **kwargs):
            return _path.unpatched(model, "data/mlc-lib", *args, **kwargs)
        _path.unpatched = chat_module._get_lib_module
        chat_module._get_lib_module = _path

        self.model = ChatModule(self.model_path, device="vulkan")

        self.tokenizer = self._get_tokenizer("EleutherAI/gpt-neo-2.7B")

    def _save_settings(self):
        pass

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        from mlc_chat.chat_module import ChatConfig

        if seed is not None:
            logger.warning(
                "Seed is unsupported on MLC backend. Seed will be ignored."
            )

        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        self.model.reset_chat()
        self.model.chat_config.max_gen_len = max_new
        self.model._prefill(decoded_prompt, decode_next_token=True)

        decoded_output = ""
        while not self.model._stopped():
            self.model._decode()
            decoded_output = self.model._get_message()
            print(decoded_output)

        genout = [decoded_output]

        return GenerationResult(
            model=self,
            out_batches=np.array([self.tokenizer.encode(x) for x in genout]),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )
