from server.socket import ui1_command
from server.kaivars import koboldai_vars


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