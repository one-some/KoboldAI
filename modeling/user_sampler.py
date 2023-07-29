from __future__ import annotations

from queue import Queue
from typing import TYPE_CHECKING

import torch

import utils

if TYPE_CHECKING:
    from modeling.inference_model import InferenceModel

user_sampler_queue = Queue()

OPTION_COUNT = 15


class UserSampler:
    def __call__(
        self,
        model: InferenceModel,
        scores: torch.FloatTensor,
        input_ids: torch.longLongTensor,
    ) -> torch.FloatTensor:
        # print(scores)

        if utils.koboldai_vars.numseqs != 1:
            return scores

        out = {}
        top_k = torch.topk(scores[0], k=OPTION_COUNT)
        for k, v in zip(top_k.indices, top_k.values):
            k = k.item()
            v = str(round(v.item(), 2))
            if v == "-inf":
                continue

            out[k] = {
                "dec": model.tokenizer.decode(k),
                "score": v,
            }
        utils.emit("user_sampler_options", out, broadcast=True, room="UI_2")

        scores[:] = float("-inf")
        target_token_id = int(user_sampler_queue.get())
        scores[0][target_token_id] = 999

        return scores
