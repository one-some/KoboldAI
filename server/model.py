import gc
import os
import torch
import warnings
import traceback
from collections import OrderedDict

import koboldai_settings
from logger import logger
from modeling.inference_model import InferenceModel

from server.kaivars import koboldai_vars
from server.settings import load_model_settings
from server.socket import SLEEP_HACK, command, ui1_command
from server.state import set_aibusy


model = None
should_save_model: bool

# Load all of the model importers
import importlib

model_backend_code = {}
model_backends = {}
for module in os.listdir("./modeling/inference_models"):
    if (
        not os.path.isfile(os.path.join("./modeling/inference_models", module))
        and module != "__pycache__"
    ):
        try:
            model_backend_code[module] = importlib.import_module(
                "modeling.inference_models.{}.class".format(module)
            )
            model_backends[
                model_backend_code[module].model_backend_name
            ] = model_backend_code[module].model_backend()
            if "disable" in vars(
                model_backends[model_backend_code[module].model_backend_name]
            ):
                if model_backends[
                    model_backend_code[module].model_backend_name
                ].disable:
                    del model_backends[model_backend_code[module].model_backend_name]
        except Exception:
            logger.error("Model Backend {} failed to load".format(module))
            logger.error(traceback.format_exc())

logger.info(
    "We loaded the following model backends: \n{}".format(
        "\n".join([x for x in model_backends])
    )
)


def set_should_save_model(do_save_model: bool) -> None:
    # Yes this is a bit weird, better solutions welcome.
    global should_save_model
    should_save_model = do_save_model


def unload_model():
    global model

    # We need to wipe out the existing model and refresh the cuda cache
    model = None
    koboldai_vars.online_model = ""

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="torch.distributed.reduce_op is deprecated"
            )
            for tensor in gc.get_objects():
                try:
                    if torch.is_tensor(tensor):
                        tensor.set_(
                            torch.tensor((), device=tensor.device, dtype=tensor.dtype)
                        )
                except:
                    pass

    gc.collect()

    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Unable to empty cache due to {e}")
        logger.debug(traceback.format_exc())

    # Reload our badwords
    koboldai_vars.badwordsids = koboldai_settings.badwordsids_default


def load_model(model_backend: str, initial_load: bool = False) -> InferenceModel:
    global model

    koboldai_vars.aibusy = True
    koboldai_vars.horde_share = False

    koboldai_vars.reset_model()

    koboldai_vars.noai = False
    set_aibusy(True)

    if koboldai_vars.model != "ReadOnly":
        command(
            "model_load_status",
            "Loading {}".format(
                model_backends[model_backend].model_name
                if "model_name" in vars(model_backends[model_backend])
                else model_backends[model_backend].id
            ),
        )
        SLEEP_HACK()

    if model:
        model.unload()

    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if not koboldai_vars.use_colab_tpu and koboldai_vars.model not in [
        "InferKit",
        "Colab",
        "API",
        "CLUSTER",
        "OAI",
        "GooseAI",
        "ReadOnly",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ]:
        # loadmodelsettings()
        # loadsettings()
        logger.init("GPU support", status="Searching")
        koboldai_vars.bmsupported = (
            (koboldai_vars.model_type != "gpt2")
            or koboldai_vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")
        ) and not koboldai_vars.nobreakmodel
        if koboldai_vars.hascuda:
            logger.init_ok("GPU support", status="Found")
        else:
            logger.init_warn("GPU support", status="Not Found")

        # if koboldai_vars.hascuda:
        #    if(koboldai_vars.bmsupported):
        #        koboldai_vars.usegpu = False
        #        koboldai_vars.breakmodel = True
        #    else:
        #        koboldai_vars.breakmodel = False
        #        koboldai_vars.usegpu = use_gpu
    else:
        koboldai_vars.default_preset = koboldai_settings.default_preset

    model = model_backends[model_backend]
    model.load(
        initial_load=initial_load,
        save_model=should_save_model,
    )
    koboldai_vars.model = (
        model.model_name if "model_name" in vars(model) else model.id
    )  # Should have model_name, but it could be set to id depending on how it's setup
    logger.debug("Model Type: {}".format(koboldai_vars.model_type))

    # TODO: Convert everywhere to use model.tokenizer
    if model:
        tokenizer = model.tokenizer

    load_model_settings(model)
    loadsettings()

    lua_startup()
    # Load scripts
    load_lua_scripts()

    final_startup()
    # if not initial_load:
    set_aibusy(False)

    ui1_command("hide_model_name")
    SLEEP_HACK()

    if not koboldai_vars.gamestarted:
        setStartState()
        sendsettings()
        refresh_settings()

    # Saving the tokenizer to the KoboldStoryRegister class so we can do token counting on the story data
    if "tokenizer" in [x for x in globals()]:
        koboldai_vars.tokenizer = tokenizer

    # Let's load the presets
    preset_same_model = {}
    preset_same_class_size = {}
    preset_same_class = {}
    preset_others = {}
    model_info_data = model_info()

    for file in os.listdir("./presets"):
        if file[-8:] == ".presets":
            with open("./presets/{}".format(file)) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            for preset in data:
                if preset["Model Name"] == koboldai_vars.model:
                    preset_same_model[preset["preset"]] = preset
                    preset_same_model[preset["preset"]]["Match"] = "Recommended"
                elif (
                    not (
                        preset["preset"] in preset_same_model
                        and preset_same_model[preset["preset"]]["Match"]
                        == "Recommended"
                    )
                    and model_info_data["Model Type"] == preset["Model Type"]
                    and model_info_data["Model Size"] == preset["Model Size"]
                ):
                    preset_same_class_size[preset["preset"]] = preset
                    preset_same_class_size[preset["preset"]]["Match"] = "Recommended"
                elif (
                    not (
                        preset["preset"] in preset_same_model
                        and preset_same_model[preset["preset"]]["Match"]
                        == "Recommended"
                    )
                    and not (
                        (
                            preset["preset"] in preset_same_class_size
                            and preset_same_class_size[preset["preset"]]["Match"]
                            == "Recommended"
                        )
                    )
                    and model_info_data["Model Type"] == preset["Model Type"]
                ):
                    preset_same_class[preset["preset"]] = preset
                    preset_same_class[preset["preset"]]["Match"] = "Same Class"
                elif (
                    preset["preset"] not in preset_same_model
                    and preset["preset"] not in preset_same_class_size
                    and preset["preset"] not in preset_same_class
                ):
                    preset_others[preset["preset"]] = preset
                    preset_others[preset["preset"]]["Match"] = "Other"

    # Combine it all
    presets = preset_same_model
    for item in preset_same_class_size:
        if item not in presets:
            presets[item] = preset_same_class_size[item]
    for item in preset_same_class:
        if item not in presets:
            presets[item] = preset_same_class[item]
    for item in preset_others:
        if item not in presets:
            presets[item] = preset_others[item]

    presets["Default"] = koboldai_vars.default_preset

    koboldai_vars.uid_presets = presets
    # We want our data to be a 2 deep dict. Top level is "Recommended", "Same Class", "Model 1", "Model 2", etc
    # Next layer is "Official", "Custom"
    # Then the preset name

    to_use = OrderedDict()

    to_use["Recommended"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Recommended"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Recommended"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    to_use["Same Class"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Same Class"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Same Class"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    to_use["Other"] = {
        "Official": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Other"
            and presets[x]["Preset Category"] == "Official"
        ],
        "Custom": [
            presets[x]
            for x in presets
            if presets[x]["Match"] == "Other"
            and presets[x]["Preset Category"] == "Custom"
        ],
    }
    koboldai_vars.presets = to_use

    koboldai_vars.aibusy = False
    if not os.path.exists("./softprompts"):
        os.mkdir("./softprompts")
    koboldai_vars.splist = [
        [f, get_softprompt_desc(os.path.join("./softprompts", f), None, True)]
        for f in os.listdir("./softprompts")
        if os.path.isfile(os.path.join("./softprompts", f))
        and valid_softprompt(os.path.join("./softprompts", f))
    ]
    if initial_load and koboldai_vars.cloudflare_link != "":
        print(
            format(colors.GREEN)
            + "KoboldAI has finished loading and is available at the following link for UI 1: "
            + koboldai_vars.cloudflare_link
            + format(colors.END)
        )
        print(
            format(colors.GREEN)
            + "KoboldAI has finished loading and is available at the following link for UI 2: "
            + koboldai_vars.cloudflare_link
            + "/new_ui"
            + format(colors.END)
        )

    return model
