from __future__ import annotations

import os
import json
import torch
import shutil

from transformers import BitsAndBytesConfig
try:
    from hf_bleeding_edge import AutoModelForCausalLM
except ImportError:
    from transformers import AutoModelForCausalLM

from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF_WEIGHTS_NAME, FLAX_WEIGHTS_NAME, FLAX_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME

import utils
import importlib
from logger import logger


from modeling import downloader
from modeling import lazy_loader
from modeling.inference_models.hf_torch import HFTorchInferenceModel

model_backend_name = "Huggingface"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class model_backend(HFTorchInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.quantization = False

    def is_valid(self, model_name, model_path, menu_path):
        base_is_valid = super().is_valid(model_name, model_path, menu_path)
        path = False
        gen_path = "models/{}".format(model_name.replace('/', '_'))
        if model_path is not None and os.path.exists(model_path):
            path = model_path
        elif os.path.exists(gen_path):
            path = gen_path

        fnames = [WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF_WEIGHTS_NAME, FLAX_WEIGHTS_NAME, FLAX_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME]

        return base_is_valid and any(os.path.exists(os.path.join(path, fname)) for fname in fnames)

    def _initialize_model(self):
        return

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        requested_parameters = super().get_requested_parameters(model_name, model_path, menu_path, parameters)
        dependency_exists = importlib.util.find_spec("bitsandbytes")
        if dependency_exists:
            if model_name != 'customhuggingface' or "custom_model_name" in parameters:
                if os.path.exists("settings/{}.generic_hf_torch.model_backend.settings".format(model_name.replace("/", "_"))) and 'base_url' not in vars(self):
                    with open("settings/{}.generic_hf_torch.model_backend.settings".format(model_name.replace("/", "_")), "r") as f:
                        temp = json.load(f)
                else:
                    temp = {}
                requested_parameters.append({
                                            "uitype": "dropdown",
                                            "unit": "text",
                                            "label": "Quantization",
                                            "id": "quantization",
                                            "default": temp['quantization'] if 'quantization' in temp else '4bit' if dependency_exists else '16-bit',
                                            "tooltip": "Whether or not to use BnB's 4-bit or 8-bit mode",
                                            "menu_path": "Layers",
                                            "children": [{'text': '4-bit', 'value': '4bit'}, {'text': '8-bit', 'value': '8bit'}, {'text': '16-bit', 'value':'16-bit'}],
                                            "extra_classes": "",
                                            "refresh_model_inputs": False
                                        })
        else:
            logger.warning("Bitsandbytes is not installed, you can not use Quantization for Huggingface models")
        return requested_parameters

    def set_input_parameters(self, parameters):
        super().set_input_parameters(parameters)
        self.quantization = parameters['quantization'] if 'quantization' in parameters else False

    def _load(self, save_model: bool, initial_load: bool) -> None:
        utils.koboldai_vars.allowsp = True

        # Make model path the same as the model name to make this consistent
        # with the other loading method if it isn't a known model type. This
        # code is not just a workaround for below, it is also used to make the
        # behavior consistent with other loading methods - Henk717
        # if utils.koboldai_vars.model not in ["NeoCustom", "GPT2Custom"]:
        #     utils.koboldai_vars.custmodpth = utils.koboldai_vars.model

        if self.model_name == "NeoCustom":
            self.model_name = os.path.basename(os.path.normpath(self.path))
        utils.koboldai_vars.model = self.model_name

        # If we specify a model and it's in the root directory, we need to move
        # it to the models directory (legacy folder structure to new)
        if self.get_local_model_path(legacy=True):
            shutil.move(
                self.get_local_model_path(legacy=True, ignore_existance=True),
                self.get_local_model_path(ignore_existance=True),
            )

        self.init_model_config()

        tf_kwargs = {
            "low_cpu_mem_usage": True,
        }

        if self.quantization == "8bit":
            tf_kwargs.update({
                "quantization_config":BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                ),
            })

        if self.quantization == "4bit" or utils.koboldai_vars.colab_arg:
            tf_kwargs.update({
                "quantization_config":BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    llm_int8_enable_fp32_cpu_offload=True
                ),
            })

        if self.model_type == "gpt2":
            # We must disable low_cpu_mem_usage and if using a GPT-2 model
            # because GPT-2 is not compatible with this feature yet.
            tf_kwargs.pop("low_cpu_mem_usage", None)
            tf_kwargs.pop("quantization_config", None)

            # Also, lazy loader doesn't support GPT-2 models
            self.lazy_load = False

        if self.model_type == "llama":
            tf_kwargs.update({
                "pretraining_tp": 1 # Workaround recommended by HF to fix their mistake on the config.json tuners adopted
            })

        logger.debug(
            "lazy_load: {} hascuda: {} breakmodel: {} nobreakmode: {}".format(
                self.lazy_load,
                utils.koboldai_vars.hascuda,
                self.breakmodel,
                self.nobreakmodel,
            )
        )

        # If we're using torch_lazy_loader, we need to get breakmodel config
        # early so that it knows where to load the individual model tensors
        if (
            self.lazy_load
            and utils.koboldai_vars.hascuda
            and utils.koboldai_vars.breakmodel
            and not utils.koboldai_vars.nobreakmodel
        ):
            self.breakmodel_device_config(self.model_config)

        if self.lazy_load:
            # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
            tf_kwargs.pop("low_cpu_mem_usage", None)

            # If we're using lazy loader, we need to figure out what the model's hidden layers are called
            with lazy_loader.use_lazy_load(dematerialized_modules=True):
                try:
                    metamodel = AutoModelForCausalLM.from_config(self.model_config)
                    utils.layers_module_names = utils.get_layers_module_names(metamodel)
                    utils.module_names = list(metamodel.state_dict().keys())
                    utils.named_buffers = list(metamodel.named_buffers(recurse=True))
                except Exception as e:
                    if utils.args.panic:
                        raise e
                    logger.warning(f"Gave up on lazy loading due to {e}")
                    self.lazy_load = False

        # Download model from Huggingface if it does not exist, otherwise load locally
        if self.get_local_model_path():
            # Model is stored locally, load it.
            self.model = self._get_model(self.get_local_model_path(), tf_kwargs)
            self.tokenizer = self._get_tokenizer(self.get_local_model_path())
        else:
            # Model not stored locally, we need to download it.

            with downloader.detect_fp32():
                self.model = self._get_model(self.model_name, tf_kwargs)
                self.tokenizer = self._get_tokenizer(self.model_name)

            if save_model:
                downloader.save_transformers_model(self)

        self.patch_embedding()

        self.model.kai_model = self
        utils.koboldai_vars.modeldim = self.get_hidden_size()

    def _save_settings(self):
        with open(
            "settings/{}.generic_hf_torch.model_backend.settings".format(
                self.model_name.replace("/", "_")
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "layers": self.layers if "layers" in vars(self) else [],
                    "disk_layers": self.disk_layers
                    if "disk_layers" in vars(self)
                    else 0,
                    "quantization": self.quantization,
                },
                f,
                indent="",
            )
