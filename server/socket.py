import time
from typing import Any, Dict, Optional


def emit(*args, **kwargs) -> None:
    # TODO
    raise NotImplementedError


def command(
    command: str,
    data: Any,
    *,
    room: Optional[str] = None,
    extra_data: Optional[Dict] = None
) -> None:
    """
    Shortcut for the common command pattern:

    emit(
        "from_server",
        {"cmd": <command>, "data": <data>},
        broadcast=True,
        room=<room>
    )
    """
    assert room in [None, "UI_1"]

    kwargs = {}

    if room:
        kwargs["room"] = room

    emit(
        "from_server",
        {"cmd": command, "data": data, **(extra_data or {})},
        broadcast=True,
        **kwargs
    )


def ui1_command(
    command: str, data: Optional[Any] = None, extra_data: Optional[Dict] = None
) -> None:
    """Same as `command` but with `UI_1` as the room."""
    command(command=command, data=data or "", room="UI_1", extra_data=extra_data)


def ui1_error(error_text: str) -> None:
    ui1_command("errmsg", str(error_text))


def ui2_error(error_text: str) -> None:
    """UI2 error notification"""
    emit("error", str(error_text), broadcast=True, room="UI_2")


def SLEEP_HACK():
    """In some areas short sleeps are needed to make emits send. This is
    obviously very weird to do, so we make it a function to ease getting rid of
    it when we are able to."""
    time.sleep(0.1)
