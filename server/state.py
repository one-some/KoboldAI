from server.kaivars import koboldai_vars
from server.socket import emit, ui1_command


def set_aibusy(busy: bool) -> None:
    """Sets the logical and display states for the AI Busy condition"""

    assert isinstance(busy, bool), "aibusy state must be boolean"

    if koboldai_vars.disable_set_aibusy:
        return

    koboldai_vars.aibusy = busy
    emit(
        "from_server",
        {"cmd": "setgamestate", "data": "wait" if busy else "ready"},
        broadcast=True,
        room="UI_1",
    )


def ui1_toggle_memory_mode() -> None:
    """Toggles the game mode for memory editing and sends UI commands"""

    if koboldai_vars.mode == "play":
        koboldai_vars.mode = "memory"
        ui1_command("memmode", "true")
        ui1_command("setinputtext", koboldai_vars.memory)
        ui1_command("setanote", koboldai_vars.authornote)
        ui1_command("setanotetemplate", koboldai_vars.authornotetemplate)
    elif koboldai_vars.mode == "memory":
        koboldai_vars.mode = "play"
        ui1_command("memmode", "false")