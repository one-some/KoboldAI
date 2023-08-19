from threading import Timer
import re
import json
import subprocess
import tempfile
import requests
import requests.adapters
import time
from transformers import PreTrainedModel
from tqdm.auto import tqdm
import os
import itertools
import huggingface_hub
import packaging.version
from typing import List, Optional

koboldai_vars = None
args = None
num_shards: Optional[int] = None
current_shard = 0
from_pretrained_model_name = ""
from_pretrained_index_filename: Optional[str] = None
from_pretrained_kwargs = {}
bar = None

layers_module_names: Optional[List[str]] = None
module_names: Optional[List[str]] = None
named_buffers: Optional[List[tuple]] = None

default_sampler_order = [6, 0, 1, 2, 3, 4, 5]

emit = None

# Hack for socket stuff that needs app context
flask_app = None

#==================================================================#
# Decorator to prevent a function's actions from being run until
# at least x seconds have passed without the function being called
#==================================================================#
def debounce(wait): 
    def decorator(fun):
        def debounced(*args, **kwargs):
            def call_it():
                fun(*args, **kwargs)
 
            try:
                debounced.t.cancel()
            except AttributeError:
                pass
 
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
 
        return debounced
 
    return decorator

#==================================================================#
# Replace fancy quotes and apostrope's with standard ones
#==================================================================#
def fixquotes(txt):
    txt = txt.replace("“", '"')
    txt = txt.replace("”", '"')
    txt = txt.replace("’", "'")
    txt = txt.replace("`", "'")
    return txt

#==================================================================#
# 
#==================================================================#
def trimincompletesentence(txt):
    # Cache length of text
    ln = len(txt)
    # Find last instance of punctuation (Borrowed from Clover-Edition by cloveranon)
    lastpunc = max(txt.rfind("."), txt.rfind("!"), txt.rfind("?"))
    # Is this the end of a quote?
    if(lastpunc < ln-1):
        if(txt[lastpunc+1] == '"'):
            lastpunc = lastpunc + 1
    if(lastpunc >= 0):
        txt = txt[:lastpunc+1]
    return txt

#==================================================================#
# 
#==================================================================#
def replaceblanklines(txt):
    return txt.replace("\n\n", "\n")

#==================================================================#
# 
#==================================================================#
def removespecialchars(txt, koboldai_vars=None):
    if koboldai_vars is None or koboldai_vars.actionmode == 0:
        txt = re.sub(r"[#/@%<>{}+=~|\^]", "", txt)
    else:
        txt = re.sub(r"[#/@%{}+=~|\^]", "", txt)
    return txt

#==================================================================#
# If the next action follows a sentence closure, add a space
#==================================================================#
def addsentencespacing(txt, koboldai_vars):
    # Don't add sentence spacing if submission is empty or starts with whitespace
    if(len(txt) == 0 or len(txt) != len(txt.lstrip())):
        return txt
    # Get last character of last action
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[koboldai_vars.actions.get_last_key()]) > 0):
            action = koboldai_vars.actions[koboldai_vars.actions.get_last_key()]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar != " "):
        txt = " " + txt
    return txt
	
def singlelineprocessing(txt, koboldai_vars):
    txt = koboldai_vars.regex_sl.sub('', txt)
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[-1]) > 0):
            action = koboldai_vars.actions[-1]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar != "\n"):
        txt = txt + "\n"
    return txt

def chatmodeprocessing(txt, koboldai_vars):
    chatregex = re.compile(r'\s+%s:[.|\n|\W|\w]*'%koboldai_vars.chatname)
    txt = chatregex.sub('', txt)
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[-1]) > 0):
            action = koboldai_vars.actions[-1]
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
    return txt

#==================================================================#
#  Cleans string for use in file name
#==================================================================#
def cleanfilename(filename):
    filteredcharacters = ('/','\\')
    filename = "".join(c for c in filename if c not in filteredcharacters).rstrip()
    return filename
    
#==================================================================#
#  Newline substitution for fairseq models
#==================================================================#
def encodenewlines(txt):
    if(koboldai_vars.newlinemode == "s"):
        return txt.replace('\n', "</s>")
    return txt

def decodenewlines(txt):
    if(koboldai_vars.newlinemode == "s"):
        return txt.replace("</s>", '\n')
    if(koboldai_vars.newlinemode == "ns"):
        return txt.replace("</s>", '')
    return txt

#==================================================================#
#  Returns number of layers given an HF model config
#==================================================================#
def num_layers(config):
    return config["n_layer"] if isinstance(config, dict) else config.num_layers if hasattr(config, "num_layers") else config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layers if hasattr(config, "n_layers") else None

#==================================================================#
#  Given the path to a pytorch_model.bin.index.json, returns how many
#  shards there are in the model
#==================================================================#
def get_num_shards(filename):
    with open(filename) as f:
        map_data = json.load(f)
    return len(set(map_data["weight_map"].values()))

#==================================================================#
#  Given the name/path of a sharded model and the path to a
#  pytorch_model.bin.index.json, returns a list of weight names in the
#  sharded model.  Requires lazy loader to be enabled to work properly
#==================================================================#
def get_sharded_checkpoint_num_tensors(pretrained_model_name_or_path, filename, is_safetensors=False, cache_dir=None, force_download=False, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
    import transformers.modeling_utils
    _revision = koboldai_vars.revision if koboldai_vars.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
    shard_paths, _ = transformers.modeling_utils.get_checkpoint_shard_files(pretrained_model_name_or_path, filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token, user_agent=user_agent, revision=_revision)

    if is_safetensors:
        from safetensors import safe_open
        return list(itertools.chain(*(safe_open(p, framework="pt", device="cpu").keys() for p in shard_paths)))

    # Torch
    import torch
    return list(itertools.chain(*(torch.load(p, map_location="cpu").keys() for p in shard_paths)))

#==================================================================#
#  Given a PreTrainedModel, returns the list of module names that correspond
#  to the model's hidden layers.
#==================================================================#
def get_layers_module_names(model: PreTrainedModel) -> List[str]:
    names: List[str] = []
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if c[0].isnumeric() and any(c[1].__class__.__name__.endswith(suffix) for suffix in ("Block", "Layer")):
                names.append(name)
            else:
                recurse(c[1], head=name + ".")
    recurse(model)
    return names

#==================================================================#
#  Given a PreTrainedModel, returns the module name that corresponds
#  to the model's input embeddings.
#==================================================================#
def get_input_embeddings_module_name(model: PreTrainedModel) -> str:
    embeddings = model.get_input_embeddings()
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if c[1] is embeddings:
                return name
            else:
                return recurse(c[1], head=name + ".")
    return recurse(model)

#==================================================================#
#  Given a PreTrainedModel and a list of module names, returns a list
#  of module names such that the union of the set of modules given as input
#  and the set of modules returned as output contains all modules in the model.
#==================================================================#
def get_missing_module_names(model: PreTrainedModel, names: List[str]) -> List[str]:
    missing_names: List[str] = []
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if any(name.startswith(n) for n in names):
                continue
            if next(c[1].named_children(), None) is None:
                missing_names.append(name)
            else:
                recurse(c[1], head=name + ".")
    recurse(model)
    return missing_names

class UIProgressBarFile(object):
    """Write TQDM progress to the UI."""
    def __init__(self, emit_func=emit) -> None:
        self.emit_func = emit_func

    def write(self, bar):
        bar = bar.replace("\r", "").replace("\n", "").replace(chr(0), "")
        if bar != "" and [ord(num) for num in bar] != [27, 91, 65]: #No idea why we're getting the 27, 1, 65 character set, just killing to so we can move on
            #logger.info(bar)
            print('\r' + bar, end='')
            time.sleep(0.01)
            try:
                with flask_app.app_context():
                    self.emit_func('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1", namespace="/")
            except Exception as e:
                pass
        
    def flush(self):
        pass

#==================================================================#
# Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt, tokenizer):
    # If the submitted context was blank, then everything is new
    if(koboldai_vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(encodenewlines(koboldai_vars.lastctx), max_length=int(2e9), truncation=True)
    txttokens = tokenizer.encode(encodenewlines(txt), max_length=int(2e9), truncation=True)
    dif       = (len(txttokens) - len(ctxtokens)) * -1
    
    # Remove the context from the returned text
    newtokens = txttokens[dif:]
    
    return decodenewlines(tokenizer.decode(newtokens))

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt, no_sentence_trimming=False, no_single_line=False):
    #remove null ascii character (used to kill chat mode text in multi-generation)
    txt = txt.replace(chr(0), "")
    if len(txt) == 0:
        return txt
    
    # Handle <|endoftext|> for models that want this
    # In the future it would be nice if we could extend this to all EOS models.
    # However, since EOS detection may have unforseen consequences for now we hardcode <|endoftext|> until more can be tested
    # - Henk
    eotregex = re.compile(r'<\|endoftext\|>[.|\n|\W|\w]*')
    txt = eotregex.sub('', txt)

    # Cleanup stray </s>
    txt = txt.replace("</s>", "")

    # Use standard quotes and apostrophes
    txt = fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(koboldai_vars.adventure):
        txt = koboldai_vars.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(koboldai_vars.frmttriminc and not koboldai_vars.chatmode and not no_sentence_trimming):
        txt = trimincompletesentence(txt)

    # Replace blank lines
    if(koboldai_vars.frmtrmblln or koboldai_vars.chatmode):
        txt = replaceblanklines(txt)

    # trim off starting new lines in replies if we're in chat mode
    if koboldai_vars.chatmode and txt and txt[0] == "\n":
        txt = txt[1:]

    # Remove special characters
    if(koboldai_vars.frmtrmspch):
        txt = removespecialchars(txt, koboldai_vars)

	# Single Line Mode
    if(koboldai_vars.singleline and not no_single_line):
        txt = singlelineprocessing(txt, koboldai_vars)

 	# Chat Mode Trimming
    if(koboldai_vars.chatmode):
        txt = chatmodeprocessing(txt, koboldai_vars)   

    for sub in koboldai_vars.substitutions:
        if not sub["enabled"]:
            continue
        i = 0
        while sub["trueTarget"] in txt or sub["target"] in txt:
            i += 1
            if i > 1000:
                print("[substitutions] Infinite recursion :^(")
                break
            txt = txt.replace(sub["trueTarget"], sub["substitution"])
            txt = txt.replace(sub["target"], sub["substitution"])
    
    return txt
