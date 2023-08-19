import copy
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Union
from urllib.error import HTTPError

import huggingface_hub
import requests
import torch
import transformers.configuration_utils
import transformers.file_utils
import transformers.modeling_utils
from tqdm.auto import tqdm

import utils
from logger import logger
from modeling.inference_model import InferenceModel
from modeling.lazy_loader import LazyTensor


@contextmanager
def detect_fp32() -> None:
    old_rebuild_tensor = torch._utils._rebuild_tensor

    def new_rebuild_tensor(
        storage: Union[LazyTensor, torch.Storage],
        storage_offset,
        shape,
        stride,
    ):
        if not isinstance(storage, LazyTensor):
            dtype = storage.dtype
        else:
            dtype = storage.storage_type.dtype
            if not isinstance(dtype, torch.dtype):
                dtype = storage.storage_type(0).dtype
        if dtype is torch.float32 and len(shape) >= 2:
            utils.koboldai_vars.fp32_model = True
        return old_rebuild_tensor(storage, storage_offset, shape, stride)

    try:
        torch._utils._rebuild_tensor = new_rebuild_tensor
        yield
    finally:
        torch._utils._rebuild_tensor = old_rebuild_tensor


def copy_fp16_transformers_files(model: InferenceModel) -> None:
    # For fp16 models, we can just copy the model files directly

    # Save the config.json
    shutil.move(
        os.path.realpath(
            huggingface_hub.hf_hub_download(
                model.model_name,
                transformers.configuration_utils.CONFIG_NAME,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
                local_files_only=True,
                legacy_cache_layout=False,
            )
        ),
        os.path.join(
            model.get_local_model_path(ignore_existance=True),
            transformers.configuration_utils.CONFIG_NAME,
        ),
    )

    if utils.num_shards is None:
        # Save the pytorch_model.bin or model.safetensors of an unsharded model
        any_success = False
        possible_checkpoint_names = [
            transformers.modeling_utils.WEIGHTS_NAME,
            transformers.modeling_utils.SAFE_WEIGHTS_NAME,
        ]

        for possible_checkpoint_name in possible_checkpoint_names:
            try:
                shutil.move(
                    os.path.realpath(
                        huggingface_hub.hf_hub_download(
                            model.model_name,
                            possible_checkpoint_name,
                            revision=utils.koboldai_vars.revision,
                            cache_dir="cache",
                            local_files_only=True,
                            legacy_cache_layout=False,
                        )
                    ),
                    os.path.join(
                        model.get_local_model_path(ignore_existance=True),
                        possible_checkpoint_name,
                    ),
                )
                any_success = True
            except Exception:
                pass

        if not any_success:
            raise RuntimeError(
                f"Couldn't find any of {possible_checkpoint_names} in cache for {model.model_name} @ '{utils.koboldai_vars.revisison}'"
            )
    else:
        # Handle saving sharded models
        with open(utils.from_pretrained_index_filename) as f:
            map_data = json.load(f)

        filenames = set(map_data["weight_map"].values())
        # Save the pytorch_model.bin.index.json of a sharded model
        shutil.move(
            os.path.realpath(utils.from_pretrained_index_filename),
            os.path.join(
                model.get_local_model_path(ignore_existance=True),
                transformers.modeling_utils.WEIGHTS_INDEX_NAME,
            ),
        )
        # Then save the pytorch_model-#####-of-#####.bin files
        for filename in filenames:
            shutil.move(
                os.path.realpath(
                    huggingface_hub.hf_hub_download(
                        model.model_name,
                        filename,
                        revision=utils.koboldai_vars.revision,
                        cache_dir="cache",
                        local_files_only=True,
                        legacy_cache_layout=False,
                    )
                ),
                os.path.join(
                    model.get_local_model_path(ignore_existance=True),
                    filename,
                ),
            )


def save_transformers_model(model: InferenceModel) -> None:
    model.tokenizer.save_pretrained(model.get_local_model_path(ignore_existance=True))

    if utils.koboldai_vars.fp32_model:
        # Use save_pretrained to convert fp32 models to fp16,
        # unless we are using disk cache because save_pretrained
        # is not supported in that case
        model.model = model.model.half()
        model.model.save_pretrained(
            model.get_local_model_path(ignore_existance=True),
            max_shard_size="500MiB",
        )
    else:
        copy_fp16_transformers_files(model)

    shutil.rmtree("cache/")


def patch_hf_hub_downloader():
    def http_get(
        url: str,
        temp_file,
        proxies=None,
        resume_size=0,
        headers=None,
        file_name=None,
        # We ignore integrity check for now
        expected_size=None,
    ):
        """
        Download remote file. Do not gobble up errors.
        """
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        r = requests.get(url, stream=True, proxies=proxies, headers=headers)
        transformers.utils.hub.hf_raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = (
            resume_size + int(content_length) if content_length is not None else None
        )

        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        if url.endswith("config.json"):
            progress = tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                initial=resume_size,
                desc=f"Downloading {file_name}"
                if file_name is not None
                else "Downloading",
                file=utils.UIProgressBarFile(),
            )
            utils.koboldai_vars.status_message = "Download Model"
            utils.koboldai_vars.total_download_chunks = total

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if url.endswith("config.json"):
                    progress.update(len(chunk))
                    utils.koboldai_vars.downloaded_chunks += len(chunk)
                temp_file.write(chunk)

        if url.endswith("config.json"):
            progress.close()

        utils.koboldai_vars.status_message = ""

    transformers.utils.hub.http_get = http_get
    huggingface_hub.file_download.http_get = http_get


def aria2_hook(
    pretrained_model_name_or_path: str,
    force_download=False,
    cache_dir=None,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    **kwargs,
):
    import transformers
    import transformers.modeling_utils
    from huggingface_hub import HfFolder

    _revision = utils.koboldai_vars.revision or huggingface_hub.constants.DEFAULT_REVISION

    if not shutil.which("aria2c"):
        # Don't do anything if aria2 is not installed
        return

    if local_files_only:
        # If local_files_only is true, we obviously don't need to download anything
        return

    if (
        os.path.exists(pretrained_model_name_or_path)
        or os.path.isfile(pretrained_model_name_or_path + ".index")
        or transformers.modeling_utils.is_remote_url(pretrained_model_name_or_path)
    ):
        return

    if proxies:
        logger.warning(
            "KoboldAI does not support using aria2 to download models from huggingface.co through a proxy.  Disabling aria2 download mode."
        )
        return

    if use_auth_token:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You specified use_auth_token=True, but a huggingface token was not found."
                )

    _cache_dir = cache_dir or transformers.TRANSFORMERS_CACHE
    _revision = (
        utils.koboldai_vars.revision or huggingface_hub.constants.DEFAULT_REVISION
    )

    headers = {"user-agent": transformers.file_utils.http_user_agent(user_agent)}
    if use_auth_token:
        headers["authorization"] = f"Bearer {use_auth_token}"

    storage_folder = os.path.join(
        _cache_dir,
        huggingface_hub.file_download.repo_folder_name(
            repo_id=pretrained_model_name_or_path, repo_type="model"
        ),
    )
    os.makedirs(storage_folder, exist_ok=True)

    def is_cached(filename):
        try:
            huggingface_hub.hf_hub_download(
                pretrained_model_name_or_path,
                filename,
                cache_dir=cache_dir,
                local_files_only=True,
                revision=revision,
            )
        except ValueError:
            return False
        return True

    filename = None

    # NOTE: For now sharded Safetensors models are not supported. Haven't seen
    # one of these out in the wild yet, probably due to how Safetensors has a
    # lot of benifits of sharding built in
    for possible_filename in [
        transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.SAFE_WEIGHTS_NAME,
        transformers.modeling_utils.WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.WEIGHTS_NAME,
    ]:
        # Try to get the huggingface.co URL of the model's weights file(s)
        url = huggingface_hub.hf_hub_url(
            pretrained_model_name_or_path, possible_filename, revision=_revision
        )

        if is_cached(possible_filename) or requests.head(
            url, allow_redirects=True, proxies=proxies, headers=headers
        ):
            filename = possible_filename
            break

    if not filename:
        return

    if filename not in [
        transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.WEIGHTS_INDEX_NAME,
    ]:
        # If the model isn't sharded, theres only one file to download
        filenames = [filename]
    else:
        # Otherwise download the pytorch_model.bin.index.json and then let aria2 download all the pytorch_model-#####-of-#####.bin files mentioned inside it
        map_filename = huggingface_hub.hf_hub_download(
            pretrained_model_name_or_path,
            filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            revision=revision,
        )
        with open(map_filename) as f:
            map_data = json.load(f)
        filenames = set(map_data["weight_map"].values())

    urls = [
        huggingface_hub.hf_hub_url(pretrained_model_name_or_path, n, revision=_revision)
        for n in filenames
    ]
    if not force_download:
        urls = [u for u, n in zip(urls, filenames) if not is_cached(n)]
        if not urls:
            return

    blob_paths = []

    # This section is a modified version of hf_hub_download from huggingface_hub
    # See https://github.com/huggingface/huggingface_hub/blob/main/LICENSE for license
    for u, n in zip(urls, filenames):
        relative_filename = os.path.join(*n.split("/"))
        if not local_files_only:
            try:
                r = huggingface_hub.file_download._request_wrapper(
                    method="HEAD",
                    url=u,
                    headers=headers,
                    allow_redirects=False,
                    follow_relative_redirects=True,
                    proxies=proxies,
                    timeout=10,
                )
                try:
                    r.raise_for_status()
                except HTTPError as e:
                    error_code = r.headers.get("X-Error-Code")
                    if error_code != "EntryNotFound":
                        raise RuntimeError(
                            f"HEAD {u} failed with error code {r.status_code}"
                        )
                    commit_hash = r.headers.get(
                        huggingface_hub.file_download.HUGGINGFACE_HEADER_X_REPO_COMMIT
                    )
                    if commit_hash is not None:
                        no_exist_file_path = (
                            Path(storage_folder)
                            / ".no_exist"
                            / commit_hash
                            / relative_filename
                        )
                        no_exist_file_path.parent.mkdir(parents=True, exist_ok=True)
                        no_exist_file_path.touch()
                        huggingface_hub.file_download._cache_commit_hash_for_specific_revision(
                            storage_folder, _revision, commit_hash
                        )
                    raise
                commit_hash = r.headers[
                    huggingface_hub.file_download.HUGGINGFACE_HEADER_X_REPO_COMMIT
                ]
                if commit_hash is None:
                    raise OSError(
                        "Distant resource does not seem to be on huggingface.co (missing"
                        " commit header)."
                    )
                etag = r.headers.get(
                    huggingface_hub.file_download.HUGGINGFACE_HEADER_X_LINKED_ETAG
                ) or r.headers.get("ETag")
                # We favor a custom header indicating the etag of the linked resource, and
                # we fallback to the regular etag header.
                # If we don't have any of those, raise an error.
                if etag is None:
                    raise OSError(
                        "Distant resource does not have an ETag, we won't be able to"
                        " reliably ensure reproducibility."
                    )
                etag = huggingface_hub.file_download._normalize_etag(etag)
                # In case of a redirect, save an extra redirect on the request.get call,
                # and ensure we download the exact atomic version even if it changed
                # between the HEAD and the GET (unlikely, but hey).
                # Useful for lfs blobs that are stored on a CDN.
                if 300 <= r.status_code <= 399:
                    url_to_download = r.headers["Location"]
                    if (
                        "lfs.huggingface.co" in url_to_download
                        or "lfs-staging.huggingface.co" in url_to_download
                    ):
                        # Remove authorization header when downloading a LFS blob
                        headers.pop("authorization", None)
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
                # Actually raise for those subclasses of ConnectionError
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                huggingface_hub.file_download.OfflineModeIsEnabled,
            ):
                # Otherwise, our Internet connection is down.
                # etag is None
                pass
        if etag is None:
            # In those cases, we cannot force download.
            if force_download:
                raise ValueError(
                    "We have no connection or you passed local_files_only, so"
                    " force_download is not an accepted option."
                )
            if huggingface_hub.file_download.REGEX_COMMIT_HASH.match(_revision):
                commit_hash = _revision
            else:
                ref_path = os.path.join(storage_folder, "refs", _revision)
                with open(ref_path) as f:
                    commit_hash = f.read()
            pointer_path = os.path.join(
                storage_folder, "snapshots", commit_hash, relative_filename
            )
            if os.path.exists(pointer_path):
                return pointer_path
            # If we couldn't find an appropriate file on disk,
            # raise an error.
            # If files cannot be found and local_files_only=True,
            # the models might've been found if local_files_only=False
            # Notify the user about that
            if local_files_only:
                raise huggingface_hub.file_download.LocalEntryNotFoundError(
                    "Cannot find the requested files in the disk cache and"
                    " outgoing traffic has been disabled. To enable hf.co look-ups"
                    " and downloads online, set 'local_files_only' to False."
                )
            else:
                raise huggingface_hub.file_download.LocalEntryNotFoundError(
                    "Connection error, and we cannot find the requested files in"
                    " the disk cache. Please try again or make sure your Internet"
                    " connection is on."
                )
        # From now on, etag and commit_hash are not None.
        blob_path = os.path.join(storage_folder, "blobs", etag)
        pointer_path = os.path.join(
            storage_folder, "snapshots", commit_hash, relative_filename
        )
        os.makedirs(os.path.dirname(blob_path), exist_ok=True)
        os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
        # if passed revision is not identical to commit_hash
        # then revision has to be a branch name or tag name.
        # In that case store a ref.
        huggingface_hub.file_download._cache_commit_hash_for_specific_revision(
            storage_folder, _revision, commit_hash
        )
        if os.path.exists(pointer_path) and not force_download:
            return pointer_path
        if os.path.exists(blob_path) and not force_download:
            # we have the blob already, but not the pointer
            huggingface_hub.file_download.logger.info(
                "creating pointer to %s from %s", blob_path, pointer_path
            )
            huggingface_hub.file_download._create_relative_symlink(
                blob_path, pointer_path
            )
            return pointer_path
        # Some Windows versions do not allow for paths longer than 255 characters.
        # In this case, we must specify it is an extended path by using the "\\?\" prefix.
        if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
            blob_path = "\\\\?\\" + os.path.abspath(blob_path)
        blob_paths.append(blob_path)

    filenames = blob_paths
    headers = [
        requests.head(
            u, headers=headers, allow_redirects=True, proxies=proxies, timeout=10
        ).headers
        for u in urls
    ]

    for n in filenames:
        prefix, suffix = n.rsplit(os.sep, 1)
        path = os.path.join(prefix, "kai-tempfile." + suffix + ".aria2")
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join(prefix, "kai-tempfile." + suffix)
        if os.path.exists(path):
            os.remove(path)
    total_length = sum(int(h["Content-Length"]) for h in headers)
    aria2_config = "\n".join(
        f"{u}\n  out={os.path.join(prefix, 'kai-tempfile.' + suffix)}"
        for u, n in zip(urls, filenames)
        for prefix, suffix in [n.rsplit(os.sep, 1)]
    ).encode()
    _download_with_aria2(
        aria2_config,
        total_length,
        use_auth_token=token if use_auth_token else None,
        user_agent=user_agent,
        force_download=force_download,
    )
    for u, n in zip(urls, filenames):
        prefix, suffix = n.rsplit(os.sep, 1)
        os.rename(
            os.path.join(prefix, "kai-tempfile." + suffix), os.path.join(prefix, suffix)
        )


def _download_with_aria2(
    aria2_config: str,
    total_length: int,
    directory: str = ".",
    user_agent=None,
    force_download=False,
    use_auth_token=None,
):
    aria2_port = utils.koboldai_vars.aria2_port or 6799
    lengths = {}
    path = None
    s = requests.Session()
    s.mount(
        "http://",
        requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(total=120, backoff_factor=1)
        ),
    )
    bar = None
    done = False
    secret = os.urandom(17).hex()

    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as f:
            f.write(aria2_config)
            f.flush()
            p = subprocess.Popen(
                [
                    "aria2c",
                    "-x",
                    "10",
                    "-s",
                    "10",
                    "-j",
                    "10",
                    "--enable-rpc=true",
                    f"--rpc-secret={secret}",
                    "--rpc-listen-port",
                    str(aria2_port),
                    "--disable-ipv6",
                    "--file-allocation=trunc",
                    "--allow-overwrite",
                    "--auto-file-renaming=false",
                    "-d",
                    directory,
                    "-i",
                    f.name,
                    "-U",
                    transformers.file_utils.http_user_agent(user_agent),
                ]
                + (["-c"] if not force_download else [])
                + (
                    [f"--header='Authorization: Bearer {use_auth_token}'"]
                    if use_auth_token
                    else []
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while p.poll() is None:
                r = s.post(
                    f"http://localhost:{aria2_port}/jsonrpc",
                    json={
                        "jsonrpc": "2.0",
                        "id": "kai",
                        "method": "aria2.tellActive",
                        "params": [f"token:{secret}"],
                    },
                ).json()["result"]

                if not r:
                    s.close()
                    if bar is not None:
                        bar.n = bar.total
                        bar.close()
                        utils.koboldai_vars.downloaded_chunks = bar.total
                    p.terminate()
                    done = True
                    break

                visited = set()
                for x in r:
                    filename = x["files"][0]["path"]
                    lengths[filename] = (
                        int(x["completedLength"]),
                        int(x["totalLength"]),
                    )
                    visited.add(filename)

                for k, v in lengths.items():
                    if k not in visited:
                        lengths[k] = (v[1], v[1])

                if bar is None:
                    bar = tqdm(
                        total=total_length,
                        desc=f"[aria2] Downloading model",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1000,
                        file=utils.UIProgressBarFile(),
                    )
                    utils.koboldai_vars.status_message = "Download Model"
                    utils.koboldai_vars.total_download_chunks = sum(
                        v[1] for v in lengths.values()
                    )
                utils.koboldai_vars.downloaded_chunks = sum(
                    v[0] for v in lengths.values()
                )
                bar.n = utils.koboldai_vars.downloaded_chunks
                bar.update()
                time.sleep(0.1)
            utils.koboldai_vars.status_message = ""
            path = f.name
    except Exception as e:
        p.terminate()
        raise e
    finally:
        try:
            if path is not None:
                if os.path.exists(path):
                    os.remove(path)
        except OSError:
            pass

    code = p.wait()
    if not done and code:
        raise OSError(f"aria2 exited with exit code {code}")
