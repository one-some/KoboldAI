import bleach
import markdown

import utils
from server.kaivars import koboldai_vars


def apply_input_formatting(txt: str) -> str:
    """Applies chosen formatting options to text submitted to AI"""

    # Add sentence spacing
    if koboldai_vars.frmtadsnsp and not koboldai_vars.chatmode:
        txt = utils.addsentencespacing(txt, koboldai_vars)

    return txt


def kml(txt: str) -> str:
    """KoboldAI Markup Formatting (Mixture of Markdown and sanitized html)"""
    txt = txt.replace(">", "&gt;")
    txt = bleach.clean(
        markdown.markdown(txt),
        tags=[
            "p",
            "em",
            "strong",
            "code",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "ul",
            "b",
            "i",
            "a",
            "span",
            "button",
        ],
        styles=["color", "font-weight"],
        attributes=["id", "class", "style", "href"],
    )
    return txt
