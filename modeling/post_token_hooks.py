import torch

import utils
from modeling.inference_model import InferenceModel

# This will be added and removed by models for temporary callbacks (API streaming)
ephemeral_callback_hook = None

class PostTokenHooks:
    @staticmethod
    def stream_tokens(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> None:
        if not model.gen_state.get("do_streaming"):
            return

        data = [
            utils.applyoutputformatting(
                utils.decodenewlines(model.tokenizer.decode(x[-1])),
                no_sentence_trimming=True,
                no_single_line=True,
            )
            for x in input_ids
        ]

        # If callback exists (and do_streaming is enabled), generate the data...
        if ephemeral_callback_hook:
            ephemeral_callback_hook(data)

        # ...but only show the client if they want to see it.
        if not utils.koboldai_vars.output_streaming:
            return

        utils.koboldai_vars.actions.stream_tokens(data)

