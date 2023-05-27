import html
from typing import Union
from logger import logger
from server.formatting import format_for_html

from server.kaivars import koboldai_vars
from server.socket import ui1_command, ui1_error
from server.state import set_gamesaved, ui1_send_debug


def refresh_story() -> str:
    """Sends the current story content to the Game Screen"""
    text_parts = [
        '<chunk n="0" id="n0" tabindex="-1">',
        koboldai_vars.comregex_ui.sub(
            lambda m: "\n".join(
                "<comment>" + l + "</comment>" for l in m.group().split("\n")
            ),
            html.escape(koboldai_vars.prompt),
        ),
        "</chunk>",
    ]
    for idx in koboldai_vars.actions:
        item = koboldai_vars.actions[idx]
        idx += 1
        item = html.escape(item)
        item = koboldai_vars.comregex_ui.sub(
            lambda m: "\n".join(
                "<comment>" + l + "</comment>" for l in m.group().split("\n")
            ),
            item,
        )  # Add special formatting to comments
        item = koboldai_vars.acregex_ui.sub(
            "<action>\\1</action>", item
        )  # Add special formatting to adventure actions
        text_parts.extend(
            (
                '<chunk n="',
                str(idx),
                '" id="n',
                str(idx),
                '" tabindex="-1">',
                item,
                "</chunk>",
            )
        )

    ui1_command(
        "updatescreen",
        data=format_for_html("".join(text_parts)),
        extra_data={
            "gamestarted": koboldai_vars.gamestarted,
        },
    )


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
        ui1_remove_story_chunk(chunk)
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
        f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{format_for_html(item)}</chunk>'
    )
    ui1_command("updatechunk", {"index": idx, "html": chunk_text})
    set_gamesaved(False)


def ui1_remove_story_chunk(idx: int):
    """Signals the Game Screen to remove one of the chunks"""
    ui1_command("removechunk", idx)
    set_gamesaved(False)


def ui1_edit_request(n: int):
    if n == 0:
        txt = koboldai_vars.prompt
    else:
        txt = koboldai_vars.actions[n - 1]

    koboldai_vars.editln = n
    ui1_command("setinputtext", txt)
    ui1_command("enablesubmit")


def ui1_edit_submit(data: str):
    koboldai_vars.recentedit = True
    if koboldai_vars.editln == 0:
        koboldai_vars.prompt = data
    else:
        koboldai_vars.actions[koboldai_vars.editln - 1] = data

    koboldai_vars.mode = "play"
    update_story_chunk(koboldai_vars.editln)
    ui1_command("texteffect", koboldai_vars.editln)
    ui1_command("editmode", "false")
    ui1_send_debug()


def ui1_delete_request():
    koboldai_vars.recentedit = True
    # Don't delete prompt
    if koboldai_vars.editln == 0:
        # Send error message
        pass
    else:
        koboldai_vars.actions.delete_action(koboldai_vars.editln - 1)
        koboldai_vars.mode = "play"
        ui1_remove_story_chunk(koboldai_vars.editln)
        ui1_command("editmode", "false")
    ui1_send_debug()
