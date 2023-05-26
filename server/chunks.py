from typing import Union
from logger import logger

from server.kaivars import koboldai_vars
from server.socket import ui1_command, ui1_error
from server.state import set_gamesaved, ui1_send_debug


def inline_edit(chunk: int, data: str) -> None:
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    if chunk == 0:
        if len(data.strip()) == 0:
            return
        koboldai_vars.prompt = data
    else:
        if chunk - 1 in koboldai_vars.actions:
            koboldai_vars.actions[chunk - 1] = data
        else:
            logger.warning(f"Attempted to edit non-existent chunk {chunk}")

    set_gamesaved(False)
    update_story_chunk(chunk)
    ui1_command("texteffect", chunk)
    ui1_command("editmode", "false")
    ui1_send_debug()


def inline_delete(chunk: int) -> None:
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if chunk == 0:
        # Send error message
        update_story_chunk(chunk)

        ui1_error("Cannot delete the prompt.")
        ui1_command("editmode", "false")
    else:
        if chunk - 1 in koboldai_vars.actions:
            koboldai_vars.actions.delete_action(chunk - 1)
        else:
            logger.warning(f"Attempted to delete non-existent chunk {chunk}")
        set_gamesaved(False)
        remove_story_chunk(chunk)
        ui1_command("editmode", "false")
    ui1_send_debug()


def update_story_chunk(idx: Union[int, str]):
    """Signals the Game Screen to update one of the chunks"""
    if idx == "last":
        if len(koboldai_vars.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            set_gamesaved(False)
            return

        idx = (
            koboldai_vars.actions.get_last_key() if len(koboldai_vars.actions) else 0
        ) + 1

    if idx == 0:
        text = koboldai_vars.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if idx - 1 not in koboldai_vars.actions:
            return
        text = koboldai_vars.actions[idx - 1]

    item = html.escape(text)
    item = koboldai_vars.comregex_ui.sub(
        lambda m: "\n".join(
            "<comment>" + l + "</comment>" for l in m.group().split("\n")
        ),
        item,
    )  # Add special formatting to comments
    item = koboldai_vars.acregex_ui.sub(
        "<action>\\1</action>", item
    )  # Add special formatting to adventure actions

    chunk_text = (
        f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    )
    emit(
        "from_server",
        {"cmd": "updatechunk", "data": {"index": idx, "html": chunk_text}},
        broadcast=True,
        room="UI_1",
    )

    set_gamesaved(False)


# ==================================================================#
# Signals the Game Screen to remove one of the chunks
# ==================================================================#
def remove_story_chunk(idx: int):
    emit(
        "from_server", {"cmd": "removechunk", "data": idx}, broadcast=True, room="UI_1"
    )
    set_gamesaved(False)


# ==================================================================#
# Sends the current generator settings to the Game Menu
# ==================================================================#
def refresh_settings():
    # Suppress toggle change events while loading state
    socketio.emit(
        "from_server",
        {"cmd": "allowtoggle", "data": False},
        broadcast=True,
        room="UI_1",
    )

    if koboldai_vars.model != "InferKit":
        socketio.emit(
            "from_server",
            {"cmd": "updatetemp", "data": koboldai_vars.temp},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetopp", "data": koboldai_vars.top_p},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetopk", "data": koboldai_vars.top_k},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetfs", "data": koboldai_vars.tfs},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetypical", "data": koboldai_vars.typical},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetopa", "data": koboldai_vars.top_a},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatereppen", "data": koboldai_vars.rep_pen},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatereppenslope", "data": koboldai_vars.rep_pen_slope},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatereppenrange", "data": koboldai_vars.rep_pen_range},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updateoutlen", "data": koboldai_vars.genamt},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetknmax", "data": koboldai_vars.max_length},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatenumseq", "data": koboldai_vars.numseqs},
            broadcast=True,
            room="UI_1",
        )
    else:
        socketio.emit(
            "from_server",
            {"cmd": "updatetemp", "data": koboldai_vars.temp},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updatetopp", "data": koboldai_vars.top_p},
            broadcast=True,
            room="UI_1",
        )
        socketio.emit(
            "from_server",
            {"cmd": "updateikgen", "data": koboldai_vars.ikgen},
            broadcast=True,
            room="UI_1",
        )

    socketio.emit(
        "from_server",
        {"cmd": "updateanotedepth", "data": koboldai_vars.andepth},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatewidepth", "data": koboldai_vars.widepth},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateuseprompt", "data": koboldai_vars.useprompt},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateadventure", "data": koboldai_vars.adventure},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatechatmode", "data": koboldai_vars.chatmode},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatedynamicscan", "data": koboldai_vars.dynamicscan},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateautosave", "data": koboldai_vars.autosave},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatenopromptgen", "data": koboldai_vars.nopromptgen},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updaterngpersist", "data": koboldai_vars.rngpersist},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatenogenmod", "data": koboldai_vars.nogenmod},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatefulldeterminism", "data": koboldai_vars.full_determinism},
        broadcast=True,
        room="UI_1",
    )

    socketio.emit(
        "from_server",
        {"cmd": "updatefrmttriminc", "data": koboldai_vars.frmttriminc},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatefrmtrmblln", "data": koboldai_vars.frmtrmblln},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatefrmtrmspch", "data": koboldai_vars.frmtrmspch},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatefrmtadsnsp", "data": koboldai_vars.frmtadsnsp},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatesingleline", "data": koboldai_vars.singleline},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateoutputstreaming", "data": koboldai_vars.output_streaming},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateshowbudget", "data": koboldai_vars.show_budget},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updateshowprobs", "data": koboldai_vars.show_probs},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatealt_text_gen", "data": koboldai_vars.alt_gen},
        broadcast=True,
        room="UI_1",
    )
    socketio.emit(
        "from_server",
        {"cmd": "updatealt_multi_gen", "data": koboldai_vars.alt_multi_gen},
        broadcast=True,
        room="UI_1",
    )

    # Allow toggle events again
    socketio.emit(
        "from_server", {"cmd": "allowtoggle", "data": True}, broadcast=True, room="UI_1"
    )


# ==================================================================#
#
# ==================================================================#
def editrequest(n):
    if n == 0:
        txt = koboldai_vars.prompt
    else:
        txt = koboldai_vars.actions[n - 1]

    koboldai_vars.editln = n
    emit(
        "from_server", {"cmd": "setinputtext", "data": txt}, broadcast=True, room="UI_1"
    )
    emit(
        "from_server", {"cmd": "enablesubmit", "data": ""}, broadcast=True, room="UI_1"
    )


# ==================================================================#
#
# ==================================================================#
def editsubmit(data):
    koboldai_vars.recentedit = True
    if koboldai_vars.editln == 0:
        koboldai_vars.prompt = data
    else:
        koboldai_vars.actions[koboldai_vars.editln - 1] = data

    koboldai_vars.mode = "play"
    update_story_chunk(koboldai_vars.editln)
    emit(
        "from_server",
        {"cmd": "texteffect", "data": koboldai_vars.editln},
        broadcast=True,
        room="UI_1",
    )
    emit("from_server", {"cmd": "editmode", "data": "false"}, room="UI_1")
    ui1_send_debug()


# ==================================================================#
#
# ==================================================================#
def deleterequest():
    koboldai_vars.recentedit = True
    # Don't delete prompt
    if koboldai_vars.editln == 0:
        # Send error message
        pass
    else:
        koboldai_vars.actions.delete_action(koboldai_vars.editln - 1)
        koboldai_vars.mode = "play"
        remove_story_chunk(koboldai_vars.editln)
        emit("from_server", {"cmd": "editmode", "data": "false"}, room="UI_1")
    ui1_send_debug()
