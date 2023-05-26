from server.kaivars import koboldai_vars
from server.socket import emit


def set_aibusy(busy: bool) -> None:
    """Sets the logical and display states for the AI Busy condition"""

    assert isinstance(busy , bool), "aibusy state must be boolean"

    if koboldai_vars.disable_set_aibusy:
        return

    koboldai_vars.aibusy = busy
    emit(
        "from_server",
        {"cmd": "setgamestate", "data": "wait" if busy else "ready"},
        broadcast=True,
        room="UI_1",
    )
