import time
from typing import Any, Optional


def emit(*args, **kwargs) -> None:
    # TODO
    raise NotImplementedError

def command(command: str, data: Any, *, room: Optional[str] = None) -> None:
    """
    Shortcut for the common command pattern:

    emit(
        "from_server",
        {"cmd": <command>, "data": <data>},
        broadcast=True,
        room="UI_1"
    )
    """
    assert room in [None, "UI_1"]

    kwargs = {}

    if room:
        kwargs["room"] = room

    emit("from_server", {"cmd": command, "data": data}, broadcast=True, **kwargs)

def ui1_command(command: str, data: Any) -> None:
    """Same as `command` but with `UI_1` as the room."""
    command(command=command, data=data, room="UI_1")

def SLEEP_HACK():
    """In some areas short sleeps are needed to make emits send. This is
    obviously very weird to do, so we make it a function to ease getting rid of
    it when we are able to."""
    time.sleep(0.1)