from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

import torch
from modeling.llama_cpp_tokenizer import LlamaCppTokenizer

import utils
from logger import logger

from modeling import warpers
from modeling.warpers import Warper
from modeling.stoppers import Stoppers
from modeling.post_token_hooks import PostTokenHooks
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    ModelCapabilities,
)

import llama_cpp

# TODO: _post_token_gen, stopper_hooks, logits_processors


class LlamaCppInferenceModel(InferenceModel):
    def __init__(
        self,
    ) -> None:
        self.post_token_hooks = [
            PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            Stoppers.core_stopper,
            Stoppers.dynamic_wi_scanner,
            Stoppers.singleline_stopper,
            Stoppers.chat_mode_stopper,
            Stoppers.stop_sequence_stopper,
        ]

        self.capabilties = ModelCapabilities(
            embedding_manipulation=False,
            post_token_hooks=True,
            stopper_hooks=False,
            post_token_probs=False,
        )

        self.llama_ctx = None
        self.llama_params = llama_cpp.llama_context_default_params()

        self.n_threads = 16

    def _load(self, save_model: bool, initial_load: bool) -> None:
        print("LOAD")
        self.tokenizer = LlamaCppTokenizer(self.llama_ctx, self.llama_params)
        self.llama_ctx = llama_cpp.llama_init_from_file(
            b"/media/somebody/Elements/AI/llamamodels/13B/ggml-model-q5_1.bin",
            self.llama_params,
        )
        print("HI!")

    def _apply_warpers(
        self, scores: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        print("applyWARPERS")
        warpers.update_settings()

        for sid in utils.koboldai_vars.sampler_order:
            warper = Warper.from_id(sid)

            if not warper.value_is_valid():
                continue

            if warper == warpers.RepetitionPenalty:
                # Rep pen needs more data than other samplers
                scores = warper.torch(scores, input_ids=input_ids)
            else:
                scores = warper.torch(scores)

            assert scores is not None, f"Scores are None; warper '{warper}' is to blame"
        return scores

    def _post_load(m_self) -> None:
        print("postLOAd")
        if not utils.koboldai_vars.model_type:
            utils.koboldai_vars.model_type = "llama-cpp"
        print("ok")

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        print("Noooo")
        if isinstance(prompt_tokens, torch.Tensor):
            in_tokens = prompt_tokens.cpu().tolist()
        else:
            in_tokens = prompt_tokens

        in_tokens = self.tokenizer.convert_tokens_for_llama_cpp(in_tokens)
        print(in_tokens)

        out = []

        start_time = time.time()

        n_past = 0
        embd = []
        last_n_size = 64
        last_n_tokens_data = [0] * last_n_size
        n_batch = 24
        last_n_repeat = 64
        repeat_penalty = 1
        frequency_penalty = 0.0
        presence_penalty = 0.0

        for _ in range(min(max_new, utils.koboldai_vars.max_length)):
            logits = llama_cpp.llama_get_logits(self.llama_ctx)

            if len(embd) > 0:
                llama_cpp.llama_eval(
                    self.llama_ctx,
                    (llama_cpp.c_int * len(embd))(*embd),
                    len(embd),
                    n_past,
                    self.n_threads,
                )

            n_past += len(embd)
            embd = []

            if len(in_tokens) <= input_consumed:
                logits = llama_cpp.llama_get_logits(self.llama_ctx)
                n_vocab = llama_cpp.llama_n_vocab(self.llama_ctx)

                _arr = (llama_cpp.llama_token_data * n_vocab)(
                    *[
                        llama_cpp.llama_token_data(token_id, logits[token_id], 0.0)
                        for token_id in range(n_vocab)
                    ]
                )
                candidates_p = llama_cpp.ctypes.pointer(
                    llama_cpp.llama_token_data_array(_arr, len(_arr), False)
                )

                _arr = (llama_cpp.c_int * len(last_n_tokens_data))(*last_n_tokens_data)

                # TODO: Fix with actual samplers
                llama_cpp.llama_sample_repetition_penalty(
                    self.llama_ctx, candidates_p, _arr, last_n_repeat, repeat_penalty
                )
                llama_cpp.llama_sample_frequency_and_presence_penalties(
                    self.llama_ctx,
                    candidates_p,
                    _arr,
                    last_n_repeat,
                    frequency_penalty,
                    presence_penalty,
                )
                llama_cpp.llama_sample_top_k(self.llama_ctx, candidates_p, 40)
                llama_cpp.llama_sample_top_p(self.llama_ctx, candidates_p, 0.8)
                llama_cpp.llama_sample_temperature(self.llama_ctx, candidates_p, 0.2)
                id = llama_cpp.llama_sample_token(self.llama_ctx, candidates_p)

                last_n_tokens_data = last_n_tokens_data[1:] + [id]
                embd.append(id)
                out.append(id)
                remaining_tokens -= 1
            else:
                while len(in_tokens) > input_consumed:
                    out.append(in_tokens[input_consumed])
                    embd.append(in_tokens[input_consumed])
                    last_n_tokens_data = last_n_tokens_data[1:] + [
                        in_tokens[input_consumed]
                    ]
                    input_consumed += 1
                    if len(embd) >= n_batch:
                        break

            for id in embd:
                print(
                    llama_cpp.llama_token_to_str(self.llama_ctx, id).decode(
                        "utf-8", errors="ignore"
                    ),
                    end="",
                    flush=True,
                )

            if len(embd) > 0 and embd[-1] == llama_cpp.llama_token_eos():
                break

        llama_cpp.llama_print_timings(self.llama_ctx)
        # llama_cpp.llama_free(self.llama_ctx)

        logger.debug(
            "torch_raw_generate: run generator {}s".format(time.time() - start_time)
        )

        return GenerationResult(
            self,
            out_batches=out,
            prompt=prompt_tokens,
            is_whole_generation=False,
            output_includes_prompt=True,
        )
