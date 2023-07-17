import json
from threading import Thread
from typing import Any, Callable, Optional

import websockets.sync.server
from torch import LongTensor
from websockets.sync.server import ServerConnection

import ai
import utils
from modeling.inference_model import InferenceModel


class WebSocketAPIError(Exception):
    def __init__(self, message: str, nonce: Optional[str] = None) -> None:
        self.message = message
        self.nonce = nonce
        super().__init__(self.message)


def token_callback_factory(stream_callback: Callable) -> Callable:
    def _token_callback(
        model: InferenceModel,
        input_ids: LongTensor,
    ) -> None:
        data = [
            utils.applyoutputformatting(
                utils.decodenewlines(model.tokenizer.decode(x[-1])),
                no_sentence_trimming=True,
                no_single_line=True,
            )
            for x in input_ids
        ]
        stream_callback(data)

    return _token_callback


def handle_request(
    command: str,
    parameters: dict,
    stream_callback: Optional[Callable] = None,
) -> Any:
    print(parameters)
    if command == "ping":
        return "pong"
    elif command == "raw_generate":
        prompt = parameters.get("prompt", None)

        if not prompt:
            raise WebSocketAPIError("No 'prompt' parameter")
        if not isinstance(prompt, str):
            raise WebSocketAPIError("'prompt' is not a string")

        ai.model.raw_generate(
            prompt=prompt,
            max_new=25,
            token_callback=token_callback_factory(stream_callback),
        )
        return True
    raise WebSocketAPIError("Unknown command.")


def handle_raw_message(
    message: str,
    raw_stream_callback: Optional[Callable] = None,
) -> dict:
    try:
        request = json.loads(message)
    except json.decoder.JSONDecodeError:
        raise WebSocketAPIError("Unable to parse command")

    # The client may provide a nonce to identify responses
    # from individual requests.
    nonce = request.get("nonce", None)

    if nonce:
        nonce = str(nonce)
        if len(nonce) > 128:
            raise WebSocketAPIError("Nonce value is unreasonably long, chill out")

    try:
        command = str(request["cmd"])
    except KeyError:
        raise WebSocketAPIError("No 'cmd' passed")

    parameters = request.get("params", {})
    if not isinstance(parameters, dict):
        raise WebSocketAPIError("Malformed 'params'. Expected object or nothing")

    def format_out(result: Any) -> dict:
        out = {"result": result}
        if nonce:
            out["nonce"] = nonce
        return out

    return format_out(
        handle_request(
            command,
            parameters,
            stream_callback=lambda result: raw_stream_callback(format_out(result)),
        )
    )


def handler(websocket: ServerConnection):
    # TODO: The stream callback might be better if refactored into a generator
    stream_callback = lambda out: websocket.send(json.dumps(out))

    for message in websocket:
        try:
            out = handle_raw_message(message, stream_callback)
        except WebSocketAPIError as e:
            out = {"err": e.message}
            if e.nonce:
                out["nonce"] = e.nonce
        websocket.send(json.dumps(out))


def run_blocking():
    print(ai.model)
    with websockets.sync.server.serve(handler, host="localhost", port=4999) as server:
        server.serve_forever()


def start():
    Thread(target=run_blocking).start()
