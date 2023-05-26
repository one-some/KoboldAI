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



def ui1_send_debug() -> None:
    if not koboldai_vars.debug:
        return

    debug_lines = []

    for debug_getter in [
        lambda: "Seed: {} ({})".format(
            repr(
                __import__("tpu_mtj_backend").get_rng_seed()
                if koboldai_vars.use_colab_tpu
                else __import__("torch").initial_seed()
            ),
            "specified by user in settings file"
            if koboldai_vars.seed_specified
            else "randomly generated",
        ),
        lambda: "Newline Mode: {}".format(koboldai_vars.newlinemode),
        lambda: "Action Length: {}".format(koboldai_vars.actions.get_last_key()),
        lambda: "Actions Metadata Length: {}".format(
            max(koboldai_vars.actions_metadata)
            if len(koboldai_vars.actions_metadata) > 0
            else 0
        ),
        lambda: "Actions: {}".format([k for k in koboldai_vars.actions]),
        lambda: "Actions Metadata: {}".format(
            [k for k in koboldai_vars.actions_metadata]
        ),
        lambda: "Last Action: {}".format(
            koboldai_vars.actions[koboldai_vars.actions.get_last_key()]
        ),
        lambda: "Last Metadata: {}".format(
            koboldai_vars.actions_metadata[max(koboldai_vars.actions_metadata)]
        ),
    ]:
        try:
            debug_lines.append(debug_getter())
        except Exception as e:
            debug_lines.append(f"(Error: {e})")

    ui1_command("debug_info", "\n".join(debug_lines))


def set_gamesaved(gamesaved: bool):
    """Set value of gamesaved"""
    assert type(gamesaved) is bool

    if gamesaved != koboldai_vars.gamesaved:
        ui1_command("gamesaved", gamesaved)
    koboldai_vars.gamesaved = gamesaved
