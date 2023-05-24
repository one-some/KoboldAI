from typing import Any, Union, List

import torch
import itertools
import numpy as np

import llama_cpp

from modeling.tokenizer import GenericTokenizer


class LlamaCppTokenizer(GenericTokenizer):
    def __init__(self, llama_ctx, llama_params) -> None:
        self.llama_ctx = llama_ctx
        self.max_tokens = llama_params.n_ctx
    
    def _base_encode(self, text: str):
        print("e")
        tokens = (llama_cpp.llama_token * int(self.max_tokens))()
        print("e1")
        n_tokens = llama_cpp.llama_tokenize(
            self.llama_ctx,
            bytes(text),
            tokens,
            self.max_tokens,
            add_bos=llama_cpp.c_bool(True),
        )
        print("e2")
        return tokens

    def encode(self, text: str) -> list:
        buf = np.ctypeslib.as_array(self._base_encode(text))
        print(buf)
        out = list(reversed(itertools.dropwhile(lambda x: x == 0, reversed(buf))))
        print(out)
        return out
    
    def convert_tokens_for_llama_cpp(self, tokens):
        # HACK: MEGA HACK!!! MEGA MEGA MEGA MEGA MEGA HACK!!!!
        text = self.decode(tokens)
        return self._base_encode(text)

    def decode(self, tokens: Union[int, List[int], torch.Tensor]) -> str:
        print("self")
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        if isinstance(tokens, int):
            tokens = [tokens]

        return "".join(
            [
                llama_cpp.llama_token_to_str(self.llama_ctx, token).decode(
                    "utf-8", errors="ignore"
                )
                for token in tokens
            ]
        )
