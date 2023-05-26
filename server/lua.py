import os
import sys
import lupa
from eventlet import tpool
from typing import Callable, TypeVar
from server.chunks import inline_delete, inline_edit

import utils
import fileops
from logger import logger, Colors
from modeling.inference_model import InferenceModel
from server.kaivars import koboldai_vars
from server.socket import ui1_command, ui1_error, ui2_error
from server.state import set_aibusy, set_gamesaved, ui1_send_debug


_bridged = {}
F = TypeVar("F", bound=Callable)

model: InferenceModel


def lua_startup(_model: InferenceModel) -> None:
    """Lua runtime startup"""
    global _bridged
    global F
    global bridged
    global model

    model = _model

    print("", end="", flush=True)
    logger.init("LUA bridge", status="Starting")

    # Set up Lua state
    koboldai_vars.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

    # Load bridge.lua
    bridged = {
        "corescript_path": "cores",
        "userscript_path": "userscripts",
        "config_path": "userscripts",
        "lib_paths": koboldai_vars.lua_state.table(
            "lualibs", os.path.join("extern", "lualibs")
        ),
        "koboldai_vars": koboldai_vars,
    }
    for kwarg in _bridged:
        bridged[kwarg] = _bridged[kwarg]
    try:
        (
            koboldai_vars.lua_kobold,
            koboldai_vars.lua_koboldcore,
            koboldai_vars.lua_koboldbridge,
        ) = koboldai_vars.lua_state.globals().dofile("bridge.lua")(
            koboldai_vars.lua_state.globals().python,
            bridged,
        )
    except lupa.LuaError as e:
        print(Colors.RED + "ERROR!" + Colors.END)
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        logger.error("LUA ERROR: " + str(e).replace("\033", ""))
        logger.warning(
            "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts."
        )
        ui2_error("error", str(e), broadcast=True, room="UI_2")
        exit(1)
    logger.init_ok("LUA bridge", status="OK")


def lua_log_format_name(name):
    return f"[{name}]" if type(name) is str else "CORE"


def bridged_kwarg(name=None):
    def _bridged_kwarg(f: F):
        _bridged[
            name
            if name is not None
            else f.__name__[4:]
            if f.__name__[:4] == "lua_"
            else f.__name__
        ] = f
        return f

    return _bridged_kwarg


# ==================================================================#
#  Event triggered when a userscript is loaded
# ==================================================================#
@bridged_kwarg()
def load_callback(filename, modulename):
    print(Colors.GREEN + f"Loading Userscript [{modulename}] <{filename}>" + Colors.END)


# ==================================================================#
#  Load all Lua scripts
# ==================================================================#
def load_lua_scripts():
    logger.init("LUA Scripts", status="Starting")

    filenames = []
    modulenames = []
    descriptions = []

    lst = fileops.getusfiles(long_desc=True)
    filenames_dict = {ob["filename"]: i for i, ob in enumerate(lst)}

    for filename in koboldai_vars.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    koboldai_vars.has_genmod = False

    try:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(
            koboldai_vars.lua_koboldbridge.load_corescript, koboldai_vars.corescript
        )
        koboldai_vars.has_genmod = tpool.execute(
            koboldai_vars.lua_koboldbridge.load_userscripts,
            filenames,
            modulenames,
            descriptions,
        )
        koboldai_vars.lua_running = True
    except lupa.LuaError as e:
        try:
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        except:
            pass

        koboldai_vars.lua_running = False

        if koboldai_vars.serverstarted:
            ui1_error("Lua script error; please check console.")
            sendUSStatItems()

        logger.error("LUA ERROR: " + str(e).replace("\033", ""))
        logger.warning(
            "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts."
        )
        ui2_error(e)

        if koboldai_vars.serverstarted:
            set_aibusy(False)
    logger.init_ok("LUA Scripts", status="OK")


# ==================================================================#
#  Print message that originates from the userscript with the given name
# ==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name:
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(
            Colors.BLUE
            + lua_log_format_name(koboldai_vars.lua_logname)
            + ":"
            + Colors.END,
            file=sys.stderr,
        )
    print(Colors.PURPLE + msg.replace("\033", "") + Colors.END)


# ==================================================================#
#  Print warning that originates from the userscript with the given name
# ==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name:
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(
            Colors.BLUE
            + lua_log_format_name(koboldai_vars.lua_logname)
            + ":"
            + Colors.END,
            file=sys.stderr,
        )
    print(Colors.YELLOW + msg.replace("\033", "") + Colors.END)


# ==================================================================#
#  Decode tokens into a string using current tokenizer
# ==================================================================#
@bridged_kwarg()
def lua_decode(tokens):
    tokens = list(tokens.values())
    assert type(tokens) is list
    if "tokenizer" not in globals():
        from transformers import GPT2Tokenizer

        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", revision=koboldai_vars.revision, cache_dir="cache"
        )
    return utils.decodenewlines(tokenizer.decode(tokens))


# ==================================================================#
#  Encode string into list of token IDs using current tokenizer
# ==================================================================#
@bridged_kwarg()
def lua_encode(string):
    assert type(string) is str
    if "tokenizer" not in globals():
        from transformers import GPT2Tokenizer

        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", revision=koboldai_vars.revision, cache_dir="cache"
        )
    return tokenizer.encode(utils.encodenewlines(string))


# ==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
# ==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if kwargs is None:
        kwargs = koboldai_vars.lua_state.table()
    actions = koboldai_vars.actions
    allowed_entries = None
    allowed_folders = None
    if entries is not None:
        allowed_entries = set()
        i = 1
        while entries[i] is not None:
            allowed_entries.add(int(entries[i]))
            i += 1
    if folders is not None:
        allowed_folders = set()
        i = 1
        while folders[i] is not None:
            allowed_folders.add(int(folders[i]))
            i += 1
    txt, _, _, found_entries = koboldai_vars.calc_ai_text(
        submitted_text=submission,
        allowed_wi_entries=allowed_entries,
        allowed_wi_folders=allowed_folders,
    )
    return utils.decodenewlines(tokenizer.decode(txt))


# ==================================================================#
#  Get property of a world info entry given its UID and property name
# ==================================================================#
@bridged_kwarg()
def lua_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "folder",
        "num",
        "selective",
        "constant",
        "uid",
    ):
        return koboldai_vars.worldinfo_u[uid][k]


# ==================================================================#
#  Set property of a world info entry given its UID, property name and new value
# ==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if type(koboldai_vars.worldinfo_u[uid][k]) is int and type(v) is float:
        v = int(v)
    assert type(koboldai_vars.worldinfo_u[uid][k]) is type(v)
    koboldai_vars.worldinfo_u[uid][k] = v
    print(
        Colors.GREEN
        + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}"
        + Colors.END
    )
    koboldai_vars.sync_worldinfo_v1_to_v2()
    sendwi()


# ==================================================================#
#  Get property of a world info folder given its UID and property name
# ==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if uid in koboldai_vars.wifolders_d and k in ("name",):
        return koboldai_vars.wifolders_d[uid][k]


# ==================================================================#
#  Set property of a world info folder given its UID, property name and new value
# ==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.wifolders_d and k in ("name",)
    if type(koboldai_vars.wifolders_d[uid][k]) is int and type(v) is float:
        v = int(v)
    assert type(koboldai_vars.wifolders_d[uid][k]) is type(v)
    koboldai_vars.wifolders_d[uid][k] = v
    print(
        Colors.GREEN
        + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}"
        + Colors.END
    )
    koboldai_vars.sync_worldinfo_v1_to_v2()
    sendwi()


# ==================================================================#
#  Get the "Amount to Generate"
# ==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return koboldai_vars.genamt


# ==================================================================#
#  Set the "Amount to Generate"
# ==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert (
        koboldai_vars.lua_koboldbridge.userstate != "genmod"
        and type(genamt) in (int, float)
        and genamt >= 0
    )
    print(
        Colors.GREEN
        + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}"
        + Colors.END
    )
    koboldai_vars.genamt = int(genamt)


# ==================================================================#
#  Get the "Gens Per Action"
# ==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return koboldai_vars.numseqs


# ==================================================================#
#  Set the "Gens Per Action"
# ==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(
        Colors.GREEN
        + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}"
        + Colors.END
    )
    koboldai_vars.numseqs = int(numseqs)


# ==================================================================#
#  Check if a setting exists with the given name
# ==================================================================#
@bridged_kwarg()
def lua_has_setting(setting):
    return setting in (
        "anotedepth",
        "settemp",
        "settopp",
        "settopk",
        "settfs",
        "settypical",
        "settopa",
        "setreppen",
        "setreppenslope",
        "setreppenrange",
        "settknmax",
        "setwidepth",
        "setuseprompt",
        "setadventure",
        "setchatmode",
        "setdynamicscan",
        "setnopromptgen",
        "autosave",
        "setrngpersist",
        "temp",
        "topp",
        "top_p",
        "topk",
        "top_k",
        "tfs",
        "typical",
        "topa",
        "reppen",
        "reppenslope",
        "reppenrange",
        "tknmax",
        "widepth",
        "useprompt",
        "chatmode",
        "chatname",
        "botname",
        "adventure",
        "dynamicscan",
        "nopromptgen",
        "rngpersist",
        "frmttriminc",
        "frmtrmblln",
        "frmtrmspch",
        "frmtadsnsp",
        "frmtsingleline",
        "triminc",
        "rmblln",
        "rmspch",
        "adsnsp",
        "singleline",
        "output_streaming",
        "show_probs",
    )


# ==================================================================#
#  Return the setting with the given name if it exists
# ==================================================================#
@bridged_kwarg()
def lua_get_setting(setting):
    if setting in ("settemp", "temp"):
        return koboldai_vars.temp
    if setting in ("settopp", "topp", "top_p"):
        return koboldai_vars.top_p
    if setting in ("settopk", "topk", "top_k"):
        return koboldai_vars.top_k
    if setting in ("settfs", "tfs"):
        return koboldai_vars.tfs
    if setting in ("settypical", "typical"):
        return koboldai_vars.typical
    if setting in ("settopa", "topa"):
        return koboldai_vars.top_a
    if setting in ("setreppen", "reppen"):
        return koboldai_vars.rep_pen
    if setting in ("setreppenslope", "reppenslope"):
        return koboldai_vars.rep_pen_slope
    if setting in ("setreppenrange", "reppenrange"):
        return koboldai_vars.rep_pen_range
    if setting in ("settknmax", "tknmax"):
        return koboldai_vars.max_length
    if setting == "anotedepth":
        return koboldai_vars.andepth
    if setting in ("setwidepth", "widepth"):
        return koboldai_vars.widepth
    if setting in ("setuseprompt", "useprompt"):
        return koboldai_vars.useprompt
    if setting in ("setadventure", "adventure"):
        return koboldai_vars.adventure
    if setting in ("setchatmode", "chatmode"):
        return koboldai_vars.chatmode
    if setting in ("setdynamicscan", "dynamicscan"):
        return koboldai_vars.dynamicscan
    if setting in ("setnopromptgen", "nopromptgen"):
        return koboldai_vars.nopromptgen
    if setting in ("autosave", "autosave"):
        return koboldai_vars.autosave
    if setting in ("setrngpersist", "rngpersist"):
        return koboldai_vars.rngpersist
    if setting in ("frmttriminc", "triminc"):
        return koboldai_vars.frmttriminc
    if setting in ("frmtrmblln", "rmblln"):
        return koboldai_vars.frmttrmblln
    if setting in ("frmtrmspch", "rmspch"):
        return koboldai_vars.frmttrmspch
    if setting in ("frmtadsnsp", "adsnsp"):
        return koboldai_vars.frmtadsnsp
    if setting in ("frmtsingleline", "singleline"):
        return koboldai_vars.singleline
    if setting == "output_streaming":
        return koboldai_vars.output_streaming
    if setting == "show_probs":
        return koboldai_vars.show_probs


# ==================================================================#
#  Set the setting with the given name if it exists
# ==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (
        actual_type is type(v) or (actual_type is int and type(v) is float)
    )
    v = actual_type(v)
    print(
        Colors.GREEN
        + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {setting} to {v}"
        + Colors.END
    )
    if setting in ("setadventure", "adventure") and v:
        koboldai_vars.actionmode = 1
    if setting in ("settemp", "temp"):
        koboldai_vars.temp = v
    if setting in ("settopp", "topp"):
        koboldai_vars.top_p = v
    if setting in ("settopk", "topk"):
        koboldai_vars.top_k = v
    if setting in ("settfs", "tfs"):
        koboldai_vars.tfs = v
    if setting in ("settypical", "typical"):
        koboldai_vars.typical = v
    if setting in ("settopa", "topa"):
        koboldai_vars.top_a = v
    if setting in ("setreppen", "reppen"):
        koboldai_vars.rep_pen = v
    if setting in ("setreppenslope", "reppenslope"):
        koboldai_vars.rep_pen_slope = v
    if setting in ("setreppenrange", "reppenrange"):
        koboldai_vars.rep_pen_range = v
    if setting in ("settknmax", "tknmax"):
        koboldai_vars.max_length = v
        return True
    if setting == "anotedepth":
        koboldai_vars.andepth = v
        return True
    if setting in ("setwidepth", "widepth"):
        koboldai_vars.widepth = v
        return True
    if setting in ("setuseprompt", "useprompt"):
        koboldai_vars.useprompt = v
        return True
    if setting in ("setadventure", "adventure"):
        koboldai_vars.adventure = v
    if setting in ("setdynamicscan", "dynamicscan"):
        koboldai_vars.dynamicscan = v
    if setting in ("setnopromptgen", "nopromptgen"):
        koboldai_vars.nopromptgen = v
    if setting in ("autosave", "noautosave"):
        koboldai_vars.autosave = v
    if setting in ("setrngpersist", "rngpersist"):
        koboldai_vars.rngpersist = v
    if setting in ("setchatmode", "chatmode"):
        koboldai_vars.chatmode = v
    if setting in ("frmttriminc", "triminc"):
        koboldai_vars.frmttriminc = v
    if setting in ("frmtrmblln", "rmblln"):
        koboldai_vars.frmttrmblln = v
    if setting in ("frmtrmspch", "rmspch"):
        koboldai_vars.frmttrmspch = v
    if setting in ("frmtadsnsp", "adsnsp"):
        koboldai_vars.frmtadsnsp = v
    if setting in ("frmtsingleline", "singleline"):
        koboldai_vars.singleline = v
    if setting == "output_streaming":
        koboldai_vars.output_streaming = v
    if setting == "show_probs":
        koboldai_vars.show_probs = v


# ==================================================================#
#  Get contents of memory
# ==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return koboldai_vars.memory


# ==================================================================#
#  Set contents of memory
# ==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    koboldai_vars.memory = m


# ==================================================================#
#  Get contents of author's note
# ==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return koboldai_vars.authornote


# ==================================================================#
#  Set contents of author's note
# ==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    koboldai_vars.authornote = m


# ==================================================================#
#  Get contents of author's note template
# ==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return koboldai_vars.authornotetemplate


# ==================================================================#
#  Set contents of author's note template
# ==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    koboldai_vars.authornotetemplate = m


# ==================================================================#
#  Save settings and send them to client
# ==================================================================#
@bridged_kwarg()
def lua_resend_settings():
    print("lua_resend_settings")
    settingschanged()
    refresh_settings()


# ==================================================================#
#  Set story chunk text and delete the chunk if the new chunk is empty
# ==================================================================#
@bridged_kwarg()
def lua_set_chunk(k, v):
    assert type(k) in (int, None) and type(v) is str
    assert k >= 0
    assert k != 0 or len(v) != 0
    if len(v) == 0:
        print(
            Colors.GREEN
            + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} deleted story chunk {k}"
            + Colors.END
        )
        chunk = int(k)
        koboldai_vars.actions.delete_action(chunk - 1)
        koboldai_vars.lua_deleted.add(chunk)
        ui1_send_debug()
    else:
        if k == 0:
            print(
                Colors.GREEN
                + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited prompt chunk"
                + Colors.END
            )
        else:
            print(
                Colors.GREEN
                + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited story chunk {k}"
                + Colors.END
            )
        chunk = int(k)
        if chunk == 0:
            if koboldai_vars.lua_koboldbridge.userstate == "genmod":
                koboldai_vars._prompt = v
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.prompt = v
        else:
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.actions[chunk - 1] = v
            ui1_send_debug()


# ==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
# ==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    if koboldai_vars.noai:
        return "readonly"
    if koboldai_vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit"):
        return "api"
    if (
        not koboldai_vars.use_colab_tpu
        and koboldai_vars.model
        not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")
        and (
            koboldai_vars.model in ("GPT2Custom", "NeoCustom")
            or koboldai_vars.model_type in ("gpt2", "gpt_neo", "gptj")
        )
    ):
        hidden_size = get_hidden_size_from_model(model)
    if koboldai_vars.model in ("gpt2",) or (
        koboldai_vars.model_type == "gpt2" and hidden_size == 768
    ):
        return "gpt2"
    if koboldai_vars.model in ("gpt2-medium",) or (
        koboldai_vars.model_type == "gpt2" and hidden_size == 1024
    ):
        return "gpt2-medium"
    if koboldai_vars.model in ("gpt2-large",) or (
        koboldai_vars.model_type == "gpt2" and hidden_size == 1280
    ):
        return "gpt2-large"
    if koboldai_vars.model in ("gpt2-xl",) or (
        koboldai_vars.model_type == "gpt2" and hidden_size == 1600
    ):
        return "gpt2-xl"
    if koboldai_vars.model_type == "gpt_neo" and hidden_size == 768:
        return "gpt-neo-125M"
    if koboldai_vars.model in ("EleutherAI/gpt-neo-1.3B",) or (
        koboldai_vars.model_type == "gpt_neo" and hidden_size == 2048
    ):
        return "gpt-neo-1.3B"
    if koboldai_vars.model in ("EleutherAI/gpt-neo-2.7B",) or (
        koboldai_vars.model_type == "gpt_neo" and hidden_size == 2560
    ):
        return "gpt-neo-2.7B"
    if (
        koboldai_vars.model in ("EleutherAI/gpt-j-6B",)
        or (
            (
                koboldai_vars.use_colab_tpu
                or koboldai_vars.model == "TPUMeshTransformerGPTJ"
            )
            and tpu_mtj_backend.params["d_model"] == 4096
        )
        or (koboldai_vars.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)
    ):
        return "gpt-j-6B"
    return "unknown"


# ==================================================================#
#  Get model backend as "transformers" or "mtj"
# ==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if koboldai_vars.noai:
        return "readonly"
    if koboldai_vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit"):
        return "api"
    if koboldai_vars.use_colab_tpu or koboldai_vars.model in (
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ):
        return "mtj"
    return "transformers"


# ==================================================================#
#  Check whether model is loaded from a custom path
# ==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return koboldai_vars.model in (
        "GPT2Custom",
        "NeoCustom",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    )


# ==================================================================#
#  Return the filename (as a string) of the current soft prompt, or
#  None if no soft prompt is loaded
# ==================================================================#
@bridged_kwarg()
def lua_get_spfilename():
    return koboldai_vars.spfilename.strip() or None


# ==================================================================#
#  When called with a string as argument, sets the current soft prompt;
#  when called with None as argument, uses no soft prompt.
#  Returns True if soft prompt changed, False otherwise.
# ==================================================================#
@bridged_kwarg()
def lua_set_spfilename(filename: Union[str, None]):
    if filename is None:
        filename = ""
    filename = str(filename).strip()
    changed = lua_get_spfilename() != filename
    assert all(q not in filename for q in ("/", "\\"))
    load_softprompt(filename)
    return changed


# ==================================================================#
#
# ==================================================================#
def execute_inmod():
    set_gamesaved(False)
    koboldai_vars.lua_logname = ...
    koboldai_vars.lua_edited = set()
    koboldai_vars.lua_deleted = set()
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        ui1_error("Lua script error; please check console.")
        ui2_error(e)
        sendUSStatItems()
        logger.error("LUA ERROR: " + str(e).replace("\033", ""))
        logger.warning(
            "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts."
        )
        set_aibusy(False)


def execute_genmod():
    koboldai_vars.lua_koboldbridge.execute_genmod()


def execute_outmod():
    set_gamesaved(False)
    ui1_command("hidemsg")
    emit("from_server", {"cmd": "hidemsg", "data": ""}, broadcast=True, room="UI_1")
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        ui1_error("Lua script error; please check console.")
        ui2_error(e)
        sendUSStatItems()
        logger.error("LUA ERROR: " + str(e).replace("\033", ""))
        logger.warning(
            "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts."
        )
        set_aibusy(False)

    if koboldai_vars.lua_koboldbridge.resend_settings_required:
        koboldai_vars.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()

    for k in koboldai_vars.lua_edited:
        inline_edit(k, koboldai_vars.actions[k])

    for k in koboldai_vars.lua_deleted:
        inline_delete(k)
