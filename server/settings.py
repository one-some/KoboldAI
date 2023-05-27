import os
import json
from typing import Optional
from dataclasses import dataclass

import koboldai_settings
from server.formatting import kml
from server.kaivars import koboldai_vars
from modeling.inference_model import InferenceModel


@dataclass
class Setting:
    name: str
    kai_vars_name: Optional[str] = None
    alter_default_preset: Optional[bool] = False


def load_settings(model_name: str) -> None:
    """Read settings from client file JSON and send to koboldai_vars"""

    path = "settings/" + model_name.replace("/", "_") + ".v2_settings"
    if not os.path.exists(path):
        return

    with open(path, "r") as file:
        koboldai_vars._model_settings.from_json(file.read())


def load_model_settings(model: InferenceModel):
    """Allow the models to override some settings"""

    model.model_config

    config = {}

    try:
        config = model.config
    except AttributeError:
        for model_dir in [
            koboldai_vars.custmodpth,
            koboldai_vars.custmodpth.replace("/", "_"),
        ]:
            try:
                with open(os.path.join(model_dir, "config.json"), "r") as file:
                    config = json.load(file)
            except FileNotFoundError:
                pass

    koboldai_vars.default_preset = koboldai_settings.default_preset

    if koboldai_vars.model_type == "xglm" or config.get("compat", "j") == "fairseq_lm":
        koboldai_vars.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if koboldai_vars.model_type == "opt" or koboldai_vars.model_type == "bloom":
        koboldai_vars.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them

    koboldai_vars.modelconfig = config

    # Simple settings
    for setting in [
        Setting("badwordsids"),
        Setting("nobreakmodel"),
        Setting("temp", alter_default_preset=True),
        Setting("top_p", alter_default_preset=True),
        Setting("top_k", alter_default_preset=True),
        Setting("tfs", alter_default_preset=True),
        Setting("typical", alter_default_preset=True),
        Setting("top_a", alter_default_preset=True),
        Setting("rep_pen", alter_default_preset=True),
        Setting("rep_pen_slope", alter_default_preset=True),
        Setting("rep_pen_range", alter_default_preset=True),
        Setting("adventure"),
        Setting("chatmode"),
        Setting("dynamicscan"),
        Setting("newlinemode"),
    ]:
        setattr(
            koboldai_vars, setting.kai_vars_name or setting.name, config[setting.name]
        )
        if setting.alter_default_preset:
            koboldai_vars.default_preset[setting.name] = config[setting.name]

    # More complicated settings have their own logic

    if "sampler_order" in config:
        sampler_order = config["sampler_order"]
        if len(sampler_order) < 7:
            sampler_order = [6] + sampler_order
        koboldai_vars.sampler_order = sampler_order

    if "formatoptns" in config:
        for setting in [
            "frmttriminc",
            "frmtrmblln",
            "frmtrmspch",
            "frmtadsnsp",
            "singleline",
        ]:
            if setting in config["formatoptns"]:
                setattr(koboldai_vars, setting, config["formatoptns"][setting])

    if "welcome" in config:
        koboldai_vars.welcome = (
            kml(config["welcome"])
            if config["welcome"] != False
            else koboldai_vars.welcome_default
        )

    if "newlinemode" in config:
        koboldai_vars.newlinemode = config["newlinemode"]

    if "antemplate" in config:
        koboldai_vars.setauthornotetemplate = config["antemplate"]
        if not koboldai_vars.gamestarted:
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate
