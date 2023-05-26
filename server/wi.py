# UI1 WI

import os
from server.kaivars import koboldai_vars
from server.socket import ui1_command
from server.state import set_gamesaved


def ui1_toggle_wi_mode():
    """Toggles the game mode for WI editing and sends UI commands"""
    if koboldai_vars.mode == "play":
        koboldai_vars.mode = "wi"
        ui1_command("wimode", "true")
    elif koboldai_vars.mode == "wi":
        # Commit WI fields first
        ui1_request_wi()
        # Then set UI state back to Play
        koboldai_vars.mode = "play"
        ui1_command("wimode", "false")
    ui1_send_wi()


def ui1_add_wi_item(folder_uid=None):
    assert folder_uid is None or folder_uid in koboldai_vars.wifolders_d
    ob = {
        "key": "",
        "keysecondary": "",
        "content": "",
        "comment": "",
        "folder": folder_uid,
        "num": len(koboldai_vars.worldinfo),
        "init": False,
        "selective": False,
        "constant": False,
    }
    koboldai_vars.worldinfo.append(ob)
    while True:
        uid = str(int.from_bytes(os.urandom(4), "little", signed=True))
        if uid not in koboldai_vars.worldinfo_u:
            break
    koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
    koboldai_vars.worldinfo[-1]["uid"] = uid
    if folder_uid is not None:
        koboldai_vars.wifolders_u[folder_uid].append(koboldai_vars.worldinfo[-1])
    ui1_command("addwiitem", ob)


def ui1_add_wi_folder() -> None:
    """Creates a new WI folder with an unused cryptographically secure random UID"""
    while True:
        uid = str(int.from_bytes(os.urandom(4), "little", signed=True))
        if uid not in koboldai_vars.wifolders_d:
            break
    ob = {"name": "", "collapsed": False}
    koboldai_vars.wifolders_d[uid] = ob
    koboldai_vars.wifolders_l.append(uid)
    koboldai_vars.wifolders_u[uid] = []
    ui1_command("addwifolder", data=ob, extra_data={"uid", uid})
    ui1_add_wi_item(folder_uid=uid)


# ==================================================================#
# ==================================================================#
def ui1_move_wi_item(dst: str, src: str):
    """Move the WI entry with UID src so that it immediately precedes the WI
    entry with UID dst"""
    set_gamesaved(False)

    if koboldai_vars.worldinfo_u[src]["folder"] is not None:
        for i, e in enumerate(
            koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]]
        ):
            if e is koboldai_vars.worldinfo_u[src]:
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]].pop(
                    i
                )
                break
    if koboldai_vars.worldinfo_u[dst]["folder"] is not None:
        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[dst]["folder"]].append(
            koboldai_vars.worldinfo_u[src]
        )
    koboldai_vars.worldinfo_u[src]["folder"] = koboldai_vars.worldinfo_u[dst]["folder"]
    for i, e in enumerate(koboldai_vars.worldinfo):
        if e is koboldai_vars.worldinfo_u[src]:
            _src = i
        elif e is koboldai_vars.worldinfo_u[dst]:
            _dst = i
    koboldai_vars.worldinfo.insert(
        _dst - (_dst >= _src), koboldai_vars.worldinfo.pop(_src)
    )
    ui1_send_wi()


def movewifolder(dst: str, src: str):
    """Move the WI folder with UID src so that it immediately precedes the WI
    folder with UID dst"""
    set_gamesaved(False)
    koboldai_vars.wifolders_l.remove(src)
    if dst is None:
        # If dst is None, that means we should move src to be the last folder
        koboldai_vars.wifolders_l.append(src)
    else:
        koboldai_vars.wifolders_l.insert(koboldai_vars.wifolders_l.index(dst), src)
    ui1_send_wi()


def ui1_send_wi():
    """Send WI to client"""
    # Cache len of WI
    ln = len(koboldai_vars.worldinfo)

    # Clear contents of WI container
    ui1_command(
        "wistart",
        data="",
        extra_data={
            "wifolders_d": koboldai_vars.wifolders_d,
            "wifolders_l": koboldai_vars.wifolders_l,
        },
    )

    # Stable-sort WI entries in order of folder
    ui1_stable_sort_wi()

    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if ln == 0:
        ui1_add_wi_item()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in koboldai_vars.worldinfo:
            if wi["folder"] != last_folder:
                ui1_command(
                    "addwifolder",
                    data=koboldai_vars.wifolders_d[str(wi["folder"])]
                    if wi["folder"] is not None
                    else None,
                    extra_data={
                        "uid": wi["folder"],
                    },
                )
                last_folder = wi["folder"]
            ob = wi
            ui1_command("addwiitem", ob)

    ui1_command("wifinish")


def ui1_request_wi():
    """Request current contents of all WI HTML elements"""
    wi_list = []
    for wi in koboldai_vars.worldinfo:
        wi_list.append(wi["num"])

    ui1_command("requestwiitem", wi_list)


# ==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
# ==================================================================#
def ui1_stable_sort_wi():
    mapping = {uid: index for index, uid in enumerate(koboldai_vars.wifolders_l)}
    koboldai_vars.worldinfo.sort(
        key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf")
    )
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(koboldai_vars.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if wi["folder"] != last_folder:
            if last_wi is not None and last_folder is not ...:
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if last_wi is not None:
        last_wi["init"] = False
    for folder in koboldai_vars.wifolders_u:
        koboldai_vars.wifolders_u[folder].sort(key=lambda x: x["num"])


# ==================================================================#
#  Extract object from server and send it to WI objects
# ==================================================================#
def commitwi(ar):
    for ob in ar:
        ob["uid"] = str(ob["uid"])
        koboldai_vars.worldinfo_u[ob["uid"]]["key"] = ob["key"]
        koboldai_vars.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        koboldai_vars.worldinfo_u[ob["uid"]]["content"] = ob["content"]
        koboldai_vars.worldinfo_u[ob["uid"]]["comment"] = ob.get("comment", "")
        koboldai_vars.worldinfo_u[ob["uid"]]["folder"] = ob.get("folder", None)
        koboldai_vars.worldinfo_u[ob["uid"]]["selective"] = ob["selective"]
        koboldai_vars.worldinfo_u[ob["uid"]]["constant"] = ob.get("constant", False)
    ui1_stable_sort_wi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    koboldai_vars.sync_worldinfo_v1_to_v2()
    ui1_send_wi()


def ui1_delete_wi(uid: str):
    if uid not in koboldai_vars.worldinfo_u:
        return

    set_gamesaved(False)

    # Store UID of deletion request
    koboldai_vars.deletewi = uid
    if koboldai_vars.deletewi is None:
        return

    if koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"] is not None:
        for i, e in enumerate(
            koboldai_vars.wifolders_u[
                koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]
            ]
        ):
            if e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]:
                koboldai_vars.wifolders_u[
                    koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]
                ].pop(i)

    for i, e in enumerate(koboldai_vars.worldinfo):
        if e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]:
            del koboldai_vars.worldinfo[i]
            break

    del koboldai_vars.worldinfo_u[koboldai_vars.deletewi]
    # Send the new WI array structure
    ui1_send_wi()
    # And reset deletewi
    koboldai_vars.deletewi = None


def ui1_delete_wi_folder(uid: str) -> None:
    uid = str(uid)
    del koboldai_vars.wifolders_u[uid]
    del koboldai_vars.wifolders_d[uid]
    del koboldai_vars.wifolders_l[koboldai_vars.wifolders_l.index(uid)]
    set_gamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    koboldai_vars.worldinfo = [
        wi for wi in koboldai_vars.worldinfo if wi["folder"] != uid or wi["init"]
    ]
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in koboldai_vars.worldinfo:
        if wi["folder"] == uid:
            wi["folder"] = None

    ui1_send_wi()


def check_world_info(
    txt,
    allowed_entries=None,
    allowed_folders=None,
    force_use_txt=False,
    scan_story=True,
    actions=None,
):
    """Look for WI keys in text to generator"""
    original_txt = txt

    if actions is None:
        actions = koboldai_vars.actions

    # Dont go any further if WI is empty
    if len(koboldai_vars.worldinfo) == 0:
        return "", set()

    # Cache actions length
    ln = len(actions)

    # Don't bother calculating action history if widepth is 0
    if koboldai_vars.widepth > 0 and scan_story:
        depth = koboldai_vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if not force_use_txt and (txt != "" and koboldai_vars.prompt != txt):
            txt = ""
            depth += 1

        if ln > 0:
            chunks = actions[-depth:]
            # i = 0
            # for key in reversed(actions):
            #    chunk = actions[key]
            #    chunks.appendleft(chunk)
            #    i += 1
            #    if(i == depth):
            #        break
        if ln >= depth:
            txt = "".join(chunks)
        elif ln > 0:
            txt = koboldai_vars.comregex_ai.sub("", koboldai_vars.prompt) + "".join(
                chunks
            )
        elif ln == 0:
            txt = koboldai_vars.comregex_ai.sub("", koboldai_vars.prompt)

    if force_use_txt:
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in koboldai_vars.worldinfo:
        if allowed_entries is not None and wi["uid"] not in allowed_entries:
            continue
        if allowed_folders is not None and wi["folder"] not in allowed_folders:
            continue

        if wi.get("constant", False):
            wimem = wimem + wi["content"] + "\n"
            found_entries.add(id(wi))
            continue

        if len(wi["key"].strip()) > 0 and (
            not wi.get("selective", False)
            or len(wi.get("keysecondary", "").strip()) > 0
        ):
            # Split comma-separated keys
            keys = wi["key"].split(",")
            keys_secondary = wi.get("keysecondary", "").split(",")

            for k in keys:
                ky = k
                # Remove leading/trailing spaces if the option is enabled
                if koboldai_vars.wirmvwhtsp:
                    ky = k.strip()
                if ky.lower() in txt.lower():
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if koboldai_vars.wirmvwhtsp:
                                ksy = ks.strip()
                            if ksy.lower() in txt.lower():
                                wimem = wimem + wi["content"] + "\n"
                                found_entries.add(id(wi))
                                found = True
                                break
                        if found:
                            break
                    else:
                        wimem = wimem + wi["content"] + "\n"
                        found_entries.add(id(wi))
                        break

    return wimem, found_entries
