import utils
from server.kaivars import koboldai_vars

def apply_input_formatting(txt: str) -> str:
    """Applies chosen formatting options to text submitted to AI"""

    # Add sentence spacing
    if koboldai_vars.frmtadsnsp and not koboldai_vars.chatmode:
        txt = utils.addsentencespacing(txt, koboldai_vars)

    return txt
