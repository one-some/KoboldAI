import json
import zipfile
import numpy as np
import torch

import fileops
from server.kaivars import koboldai_vars


def load_softprompt(filename: str) -> None:
    """Sets the current softprompt to that located in the given filename."""

    if not koboldai_vars.allowsp:
        raise RuntimeError(
            "Soft prompts are not supported by your current model/backend"
        )

    old_filename = koboldai_vars.spfilename

    koboldai_vars.spfilename = ""

    if len(filename) == 0:
        koboldai_vars.sp = None
        koboldai_vars.sp_length = 0
        if old_filename != filename:
            koboldai_vars.sp_changed = True
        return

    z, version, shape, fortran_order, dtype = fileops.checksp(
        "./softprompts/" + filename, koboldai_vars.modeldim
    )

    if not isinstance(z, zipfile.ZipFile):
        raise RuntimeError(f"{repr(filename)} is not a valid soft prompt file")

    with z.open("meta.json") as f:
        koboldai_vars.spmeta = json.load(f)
        koboldai_vars.spname = koboldai_vars.spmeta["name"]

    z.close()

    with np.load(fileops.sppath(filename), allow_pickle=False) as f:
        tensor = f["tensor.npy"]

    # If the tensor is in bfloat16 format, convert it to float32
    if tensor.dtype == "V2":
        tensor.dtype = np.uint16
        tensor = np.uint32(tensor) << 16
        tensor.dtype = np.float32

    if tensor.dtype != np.float16:
        tensor = np.float32(tensor)
    assert not np.isinf(tensor).any() and not np.isnan(tensor).any()

    koboldai_vars.sp_length = tensor.shape[-2]
    koboldai_vars.spmeta["n_tokens"] = koboldai_vars.sp_length

    if koboldai_vars.use_colab_tpu or koboldai_vars.model in (
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ):
        # NOTE: Only import if TPU is used.
        import tpu_mtj_backend

        rows = tensor.shape[0]
        padding_amount = (
            tpu_mtj_backend.params["seq"]
            - (
                tpu_mtj_backend.params["seq"]
                % -tpu_mtj_backend.params["cores_per_replica"]
            )
            - rows
        )
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        koboldai_vars.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        koboldai_vars.sp = torch.from_numpy(tensor)

    koboldai_vars.spfilename = filename

    if old_filename != filename:
        koboldai_vars.sp_changed = True


def is_softprompt_valid(path: str) -> bool:
    z, version, shape, fortran_order, dtype = fileops.checksp(
        path, koboldai_vars.modeldim
    )
    if z in [1, 2, 3, 4]:
        return False
    elif not isinstance(z, zipfile.ZipFile):
        print("not zip")
        return False
    else:
        return True


def get_softprompt_desc(path: str, valid_selection: bool) -> list:
    if not valid_selection:
        return [None, None]
    z = zipfile.ZipFile(path)
    with z.open("meta.json") as f:
        ob = json.load(f)
        return [ob["name"], ob["description"]]
