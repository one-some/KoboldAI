from typing import Any, Callable, List, Optional, Type, TypeVar

import json
import functools
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow import fields, validate, EXCLUDE
from marshmallow.exceptions import ValidationError

from flask import jsonify, redirect, render_template, request
from flask import current_app as app
from flask.helpers import abort
from flask.wrappers import Response


from api.util import api_schema_wrap
from api.schema import (
    BasicBooleanSchema,
    BasicStringSchema,
    EmptySchema,
    GenerationInputSchema,
    KoboldSchema,
    ModelSelectionSchema,
    SoftPromptSettingSchema,
    StoryChunkSetTextSchema,
    StoryLoadSchema,
    StorySaveSchema,
    api_validation_error_response,
    config_endpoint_schema,
)

import fileops

# HACK: Still relying on kaivars...
from utils import koboldai_vars

tags = [
    {"name": "info", "description": "Metadata about this API"},
    {"name": "generate", "description": "Text generation endpoints"},
    {
        "name": "model",
        "description": "Information about the current text generation model",
    },
    {
        "name": "story",
        "description": "Endpoints for managing the story in the KoboldAI GUI",
    },
    {
        "name": "world_info",
        "description": "Endpoints for managing the world info in the KoboldAI GUI",
    },
    {"name": "config", "description": "Allows you to get/set various setting values"},
]

api_version = None  # This gets set automatically so don't change this value
api_versions: List[str] = []


class KoboldAPISpec(APISpec):
    class KoboldFlaskPlugin(FlaskPlugin):
        def __init__(self, api: "KoboldAPISpec", *args, **kwargs):
            self._kobold_api_spec = api
            super().__init__(*args, **kwargs)

        def path_helper(self, *args, **kwargs):
            return super().path_helper(*args, **kwargs)[
                len(self._kobold_api_spec._prefixes[0]) :
            ]

    def __init__(
        self,
        *args,
        title: str = "KoboldAI API",
        openapi_version: str = "3.0.3",
        version: str = "1.0.0",
        prefixes: List[str] = None,
        **kwargs,
    ):
        plugins = [KoboldAPISpec.KoboldFlaskPlugin(self), MarshmallowPlugin()]
        self._prefixes = prefixes if prefixes is not None else [""]
        self._kobold_api_spec_version = version
        api_versions.append(version)
        api_versions.sort(key=lambda x: [int(e) for e in x.split(".")])
        super().__init__(
            *args,
            title=title,
            openapi_version=openapi_version,
            version=version,
            plugins=plugins,
            servers=[{"url": self._prefixes[0]}],
            **kwargs,
        )
        for prefix in self._prefixes:
            app.route(prefix, endpoint="~KoboldAPISpec~" + prefix)(
                lambda: redirect(request.path + "/docs/")
            )
            app.route(prefix + "/", endpoint="~KoboldAPISpec~" + prefix + "/")(
                lambda: redirect("docs/")
            )
            app.route(prefix + "/docs", endpoint="~KoboldAPISpec~" + prefix + "/docs")(
                lambda: redirect("docs/")
            )
            app.route(
                prefix + "/docs/", endpoint="~KoboldAPISpec~" + prefix + "/docs/"
            )(
                lambda: render_template(
                    "swagger-ui.html", url=self._prefixes[0] + "/openapi.json"
                )
            )
            app.route(
                prefix + "/openapi.json",
                endpoint="~KoboldAPISpec~" + prefix + "/openapi.json",
            )(lambda: jsonify(self.to_dict()))

    def route(self, rule: str, methods=["GET"], **kwargs):
        __F = TypeVar("__F", bound=Callable[..., Any])

        if "strict_slashes" not in kwargs:
            kwargs["strict_slashes"] = False

        def new_decorator(f: __F) -> __F:
            @functools.wraps(f)
            def g(*args, **kwargs):
                global api_version
                api_version = self._kobold_api_spec_version
                try:
                    return f(*args, **kwargs)
                finally:
                    api_version = None

            for prefix in self._prefixes:
                g = app.route(prefix + rule, methods=methods, **kwargs)(g)

            with app.test_request_context():
                self.path(view=g, **kwargs)
            return g

        return new_decorator

    def get(self, rule: str, **kwargs):
        return self.route(rule, methods=["GET"], **kwargs)

    def post(self, rule: str, **kwargs):
        return self.route(rule, methods=["POST"], **kwargs)

    def put(self, rule: str, **kwargs):
        return self.route(rule, methods=["PUT"], **kwargs)

    def patch(self, rule: str, **kwargs):
        return self.route(rule, methods=["PATCH"], **kwargs)

    def delete(self, rule: str, **kwargs):
        return self.route(rule, methods=["DELETE"], **kwargs)


api_v1 = KoboldAPISpec(
    version="1.2.2",
    prefixes=["/api/v1", "/api/latest"],
    tags=tags,
)


@api_v1.get("/info/version")
@api_schema_wrap
def get_version():
    """---
    get:
      summary: Current API version
      tags:
        - info
      description: |-2
        Returns the version of the API that you are currently using.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: 1.0.0
    """
    return {"result": api_version}


@api_v1.get("/info/version/latest")
@api_schema_wrap
def get_version_latest():
    """---
    get:
      summary: Latest API version
      tags:
        - info
      description: |-2
        Returns the latest API version available.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: 1.0.0
    """
    return {"result": api_versions[-1]}


@api_v1.get("/info/version/list")
@api_schema_wrap
def get_version_list():
    """---
    get:
      summary: List API versions
      tags:
        - info
      description: |-2
        Returns a list of available API versions sorted in ascending order.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultsSchema
              example:
                results:
                  - 1.0.0
    """
    return {"results": api_versions}


@api_v1.post("/generate")
@api_schema_wrap
def post_generate(body: GenerationInputSchema):
    """---
    post:
      summary: Generate text
      tags:
        - generate
      description: |-2
        Generates text given a submission, sampler settings, soft prompt and number of return sequences.

        By default, the story, userscripts, memory, author's note and world info are disabled.

        Unless otherwise specified, optional values default to the values in the KoboldAI GUI.
      requestBody:
        required: true
        content:
          application/json:
            schema: GenerationInputSchema
            example:
              prompt: |-2
                Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze.
              top_p: 0.9
              temperature: 0.5
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: GenerationOutputSchema
              example:
                results:
                  - text: |-2
                       Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken.
        {api_validation_error_response}
        {api_not_implemented_response}
        {api_server_busy_response}
        {api_out_of_memory_response}
    """
    return _generate_text(body)


@api_v1.get("/model")
@api_schema_wrap
def get_model():
    """---
    get:
      summary: Retrieve the current model string
      description: |-2
        Gets the current model string, which is shown in the title of the KoboldAI GUI in parentheses, e.g. "KoboldAI Client (KoboldAI/fairseq-dense-13B-Nerys-v2)".
      tags:
        - model
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: KoboldAI/fairseq-dense-13B-Nerys-v2
    """
    return {"result": koboldai_vars.model}


@api_v1.put("/model")
@api_schema_wrap
def put_model(body: ModelSelectionSchema):
    """---
    put:
      summary: Load a model
      description: |-2
        Loads a model given its Hugging Face model ID, the path to a model folder (relative to the "models" folder in the KoboldAI root folder) or "ReadOnly" for no model.
      tags:
        - model
      requestBody:
        required: true
        content:
          application/json:
            schema: ModelSelectionSchema
            example:
              model: ReadOnly
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    set_aibusy(True)
    old_model = koboldai_vars.model
    koboldai_vars.model = body.model.strip()
    try:
        load_model(use_breakmodel_args=True, breakmodel_args_default_to_cpu=True)
    except Exception as e:
        koboldai_vars.model = old_model
        raise e
    set_aibusy(False)
    return {}


def prompt_validator(prompt: str):
    if len(prompt.strip()) == 0:
        raise ValidationError("String does not match expected pattern.")


class SubmissionInputSchema(KoboldSchema):
    prompt: str = fields.String(
        required=True,
        validate=prompt_validator,
        metadata={
            "pattern": r"^[\S\s]*\S[\S\s]*$",
            "description": "This is the submission.",
        },
    )
    disable_input_formatting: bool = fields.Boolean(
        load_default=True,
        metadata={
            "description": "When enabled, disables all input formatting options, overriding their individual enabled/disabled states."
        },
    )
    frmtadsnsp: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action."
        }
    )


@api_v1.post("/story/end")
@api_schema_wrap
def post_story_end(body: SubmissionInputSchema):
    """---
    post:
      summary: Add an action to the end of the story
      tags:
        - story
      description: |-2
        Inserts a single action at the end of the story in the KoboldAI GUI without generating text.
      requestBody:
        required: true
        content:
          application/json:
            schema: SubmissionInputSchema
            example:
              prompt: |-2
                 This is some text to put at the end of the story.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    set_aibusy(True)
    disable_set_aibusy = koboldai_vars.disable_set_aibusy
    koboldai_vars.disable_set_aibusy = True
    _standalone = koboldai_vars.standalone
    koboldai_vars.standalone = True
    numseqs = koboldai_vars.numseqs
    koboldai_vars.numseqs = 1
    try:
        actionsubmit(
            body.prompt, force_submit=True, no_generate=True, ignore_aibusy=True
        )
    finally:
        koboldai_vars.disable_set_aibusy = disable_set_aibusy
        koboldai_vars.standalone = _standalone
        koboldai_vars.numseqs = numseqs
    set_aibusy(False)
    return {}


@api_v1.get("/story/end")
@api_schema_wrap
def get_story_end():
    """---
    get:
      summary: Retrieve the last action of the story
      tags:
        - story
      description: |-2
        Returns the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkResultSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Could not retrieve the last action of the story because the story is empty.",
                            "type": "story_empty",
                        }
                    }
                ),
                mimetype="application/json",
                status=510,
            )
        )
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": koboldai_vars.prompt, "num": 0}}
    return {
        "result": {
            "text": koboldai_vars.actions[koboldai_vars.actions.get_last_key()],
            "num": koboldai_vars.actions.get_last_key() + 1,
        }
    }


@api_v1.get("/story/end/num")
@api_schema_wrap
def get_story_end_num():
    """---
    get:
      summary: Retrieve the num of the last action of the story
      tags:
        - story
      description: |-2
        Returns the `num` of the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkNumSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Could not retrieve the last action of the story because the story is empty.",
                            "type": "story_empty",
                        }
                    }
                ),
                mimetype="application/json",
                status=510,
            )
        )
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": 0}}
    return {"result": {"text": koboldai_vars.actions.get_last_key() + 1}}


@api_v1.get("/story/end/text")
@api_schema_wrap
def get_story_end_text():
    """---
    get:
      summary: Retrieve the text of the last action of the story
      tags:
        - story
      description: |-2
        Returns the text of the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkTextSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Could not retrieve the last action of the story because the story is empty.",
                            "type": "story_empty",
                        }
                    }
                ),
                mimetype="application/json",
                status=510,
            )
        )
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": koboldai_vars.prompt}}
    return {
        "result": {"text": koboldai_vars.actions[koboldai_vars.actions.get_last_key()]}
    }


@api_v1.put("/story/end/text")
@api_schema_wrap
def put_story_end_text(body: StoryChunkSetTextSchema):
    """---
    put:
      summary: Set the text of the last action of the story
      tags:
        - story
      description: |-2
        Sets the text of the last action of the story in the KoboldAI GUI to the desired value.
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryChunkSetTextSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
        {api_validation_error_response}
    """
    if not koboldai_vars.gamestarted:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Could not retrieve the last action of the story because the story is empty.",
                            "type": "story_empty",
                        }
                    }
                ),
                mimetype="application/json",
                status=510,
            )
        )
    value = body.value.rstrip()
    if len(koboldai_vars.actions) == 0:
        inlineedit(0, value)
    else:
        inlineedit(koboldai_vars.actions.get_last_key() + 1, value)
    return {}


@api_v1.post("/story/end/delete")
@api_schema_wrap
def post_story_end_delete(body: EmptySchema):
    """---
    post:
      summary: Remove the last action of the story
      tags:
        - story
      description: |-2
        Removes the last action of the story in the KoboldAI GUI.
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        510:
          description: Story too short
          content:
            application/json:
              schema: StoryTooShortErrorSchema
              example:
                detail:
                  msg: Could not delete the last action of the story because the number of actions in the story is less than or equal to 1.
                  type: story_too_short
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    if not koboldai_vars.gamestarted or not len(koboldai_vars.actions):
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Could not delete the last action of the story because the number of actions in the story is less than or equal to 1.",
                            "type": "story_too_short",
                        }
                    }
                ),
                mimetype="application/json",
                status=510,
            )
        )
    actionback()
    return {}


@api_v1.get("/story")
@api_schema_wrap
def get_story():
    """---
    get:
      summary: Retrieve the entire story
      tags:
        - story
      description: |-2
        Returns the entire story currently shown in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StorySchema
    """
    chunks = []
    if koboldai_vars.gamestarted:
        chunks.append({"num": 0, "text": koboldai_vars.prompt})
    for num, action in koboldai_vars.actions.items():
        chunks.append({"num": num + 1, "text": action})
    return {"results": chunks}


@api_v1.get("/story/nums")
@api_schema_wrap
def get_story_nums():
    """---
    get:
      summary: Retrieve a list of the nums of the chunks in the current story
      tags:
        - story
      description: |-2
        Returns the `num`s of the story chunks currently shown in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StorySchema
    """
    chunks = []
    if koboldai_vars.gamestarted:
        chunks.append(0)
    for num in koboldai_vars.actions.keys():
        chunks.append(num + 1)
    return {"results": chunks}


@api_v1.get("/story/nums/<int(signed=True):num>")
@api_schema_wrap
def get_story_nums_num(num: int):
    """---
    get:
      summary: Determine whether or not there is a story chunk with the given num
      tags:
        - story
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    if num == 0:
        return {"result": koboldai_vars.gamestarted}
    return {"result": num - 1 in koboldai_vars.actions}


@api_v1.get("/story/<int(signed=True):num>")
@api_schema_wrap
def get_story_num(num: int):
    """---
    get:
      summary: Retrieve a story chunk
      tags:
        - story
      description: |-2
        Returns information about a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(
                Response(
                    json.dumps(
                        {
                            "detail": {
                                "msg": "No chunk with the given num exists.",
                                "type": "key_error",
                            }
                        }
                    ),
                    mimetype="application/json",
                    status=404,
                )
            )
        return {"result": {"text": koboldai_vars.prompt, "num": num}}
    if num - 1 not in koboldai_vars.actions:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No chunk with the given num exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"result": {"text": koboldai_vars.actions[num - 1], "num": num}}


@api_v1.get("/story/<int(signed=True):num>/text")
@api_schema_wrap
def get_story_num_text(num: int):
    """---
    get:
      summary: Retrieve the text of a story chunk
      tags:
        - story
      description: |-2
        Returns the text inside a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkTextSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(
                Response(
                    json.dumps(
                        {
                            "detail": {
                                "msg": "No chunk with the given num exists.",
                                "type": "key_error",
                            }
                        }
                    ),
                    mimetype="application/json",
                    status=404,
                )
            )
        return {"value": koboldai_vars.prompt}
    if num - 1 not in koboldai_vars.actions:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No chunk with the given num exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.actions[num - 1]}


@api_v1.put("/story/<int(signed=True):num>/text")
@api_schema_wrap
def put_story_num_text(body: StoryChunkSetTextSchema, num: int):
    """---
    put:
      summary: Set the text of a story chunk
      tags:
        - story
      description: |-2
        Sets the text inside a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryChunkSetTextSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
        {api_validation_error_response}
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(
                Response(
                    json.dumps(
                        {
                            "detail": {
                                "msg": "No chunk with the given num exists.",
                                "type": "key_error",
                            }
                        }
                    ),
                    mimetype="application/json",
                    status=404,
                )
            )
        inlineedit(0, body.value.rstrip())
        return {}
    if num - 1 not in koboldai_vars.actions:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No chunk with the given num exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    inlineedit(num, body.value.rstrip())
    return {}


@api_v1.delete("/story/<int(signed=True):num>")
@api_schema_wrap
def post_story_num_delete(num: int):
    """---
    delete:
      summary: Remove a story chunk
      tags:
        - story
      description: |-2
        Removes a story chunk from the story in the KoboldAI GUI given its `num`. Cannot be used to delete the first action (the prompt).
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk. Must be larger than or equal to 1.
          schema:
            type: integer
            minimum: 1
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
        {api_server_busy_response}
    """
    if num < 1:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "num": ["Must be greater than or equal to 1."],
                        }
                    }
                ),
                mimetype="application/json",
                status=422,
            )
        )
    if num - 1 not in koboldai_vars.actions:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No chunk with the given num exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    inlinedelete(num)
    return {}


@api_v1.delete("/story")
@api_schema_wrap
def delete_story():
    """---
    delete:
      summary: Clear the story
      tags:
        - story
      description: |-2
        Starts a new blank story.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    newGameRequest()
    return {}


@api_v1.put("/story/load")
@api_schema_wrap
def put_story_load(body: StoryLoadSchema):
    """---
    put:
      summary: Load a story
      tags:
        - story
      description: |-2
        Loads a story given its filename (without the .json).
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryLoadSchema
            example:
              name: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    loadRequest(fileops.storypath(body.name.strip()))
    return {}


@api_v1.put("/story/save")
@api_schema_wrap
def put_story_save(body: StorySaveSchema):
    """---
    put:
      summary: Save the current story
      tags:
        - story
      description: |-2
        Saves the current story given its destination filename (without the .json).
      requestBody:
        required: true
        content:
          application/json:
            schema: StorySaveSchema
            example:
              name: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    saveRequest(fileops.storypath(body.name.strip()))
    return {}


@api_v1.get("/world_info")
@api_schema_wrap
def get_world_info():
    """---
    get:
      summary: Retrieve all world info entries
      tags:
        - world_info
      description: |-2
        Returns all world info entries currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI and the `entries` within the folders and within the parent `result` object are all sorted in the same order as they are in their respective parts of the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoSchema
    """
    folders = []
    entries = []
    ln = len(koboldai_vars.worldinfo)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in koboldai_vars.worldinfo_i:
            if wi["folder"] != last_folder:
                folder = []
                if wi["folder"] is not None:
                    folders.append(
                        {
                            "uid": wi["folder"],
                            "name": koboldai_vars.wifolders_d[wi["folder"]]["name"],
                            "entries": folder,
                        }
                    )
                last_folder = wi["folder"]
            (folder if wi["folder"] is not None else entries).append(
                {
                    k: v
                    for k, v in wi.items()
                    if k not in ("init", "folder", "num")
                    and (wi["selective"] or k != "keysecondary")
                }
            )
    return {"folders": folders, "entries": entries}


@api_v1.get("/world_info/uids")
@api_schema_wrap
def get_world_info_uids():
    """---
    get:
      summary: Retrieve the UIDs of all world info entries
      tags:
        - world_info
      description: |-2
        Returns in a similar format as GET /world_info except only the `uid`s are returned.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoUIDsSchema
    """
    folders = []
    entries = []
    ln = len(koboldai_vars.worldinfo)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in koboldai_vars.worldinfo_i:
            if wi["folder"] != last_folder:
                folder = []
                if wi["folder"] is not None:
                    folders.append({"uid": wi["folder"], "entries": folder})
                last_folder = wi["folder"]
            (folder if wi["folder"] is not None else entries).append(wi["uid"])
    return {"folders": folders, "entries": entries}


@api_v1.get("/world_info/uids/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_uids_uid(uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {
        "result": uid in koboldai_vars.worldinfo_u
        and koboldai_vars.worldinfo_u[uid]["init"]
    }


@api_v1.get("/world_info/folders")
@api_schema_wrap
def get_world_info_folders():
    """---
    get:
      summary: Retrieve all world info folders
      tags:
        - world_info
      description: |-2
        Returns details about all world info folders currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoFoldersSchema
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    return {
        "folders": [
            {
                "uid": folder,
                **{
                    k: v
                    for k, v in koboldai_vars.wifolders_d[folder].items()
                    if k != "collapsed"
                },
            }
            for folder in koboldai_vars.wifolders_l
        ]
    }


@api_v1.get("/world_info/folders/uids")
@api_schema_wrap
def get_world_info_folders_uids():
    """---
    get:
      summary: Retrieve the UIDs all world info folders
      tags:
        - world_info
      description: |-2
        Returns the `uid`s of all world info folders currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoFoldersUIDsSchema
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    return {"folders": koboldai_vars.wifolders_l}


@api_v1.get("/world_info/folders/none")
@api_schema_wrap
def get_world_info_folders_none():
    """---
    get:
      summary: Retrieve all world info entries not in a folder
      tags:
        - world_info
      description: |-2
        Returns all world info entries that are not in a world info folder.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesSchema
    """
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in reversed(koboldai_vars.worldinfo_i):
        if wi["folder"] is not None:
            break
        entries.append(
            {
                k: v
                for k, v in wi.items()
                if k not in ("init", "folder", "num")
                and (wi["selective"] or k != "keysecondary")
            }
        )
    return {"entries": list(reversed(entries))}


@api_v1.get("/world_info/folders/none/uids")
@api_schema_wrap
def get_world_info_folders_none_uids():
    """---
    get:
      summary: Retrieve the UIDs of all world info entries not in a folder
      tags:
        - world_info
      description: |-2
        Returns the `uid`s of all world info entries that are not in a world info folder.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesUIDsSchema
    """
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in reversed(koboldai_vars.worldinfo_i):
        if wi["folder"] is not None:
            break
        entries.append(wi["uid"])
    return {"entries": list(reversed(entries))}


@api_v1.get("/world_info/folders/none/uids/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_folders_none_uids_uid(uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID that is not in a world info folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {
        "result": uid in koboldai_vars.worldinfo_u
        and koboldai_vars.worldinfo_u[uid]["folder"] is None
        and koboldai_vars.worldinfo_u[uid]["init"]
    }


@api_v1.get("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_folders_uid(uid: int):
    """---
    get:
      summary: Retrieve all world info entries in the given folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      description: |-2
        Returns all world info entries that are in the world info folder with the given `uid`.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in koboldai_vars.wifolders_u[uid]:
        if wi["init"]:
            entries.append(
                {
                    k: v
                    for k, v in wi.items()
                    if k not in ("init", "folder", "num")
                    and (wi["selective"] or k != "keysecondary")
                }
            )
    return {"entries": entries}


@api_v1.get("/world_info/folders/<int(signed=True):uid>/uids")
@api_schema_wrap
def get_world_info_folders_uid_uids(uid: int):
    """---
    get:
      summary: Retrieve the UIDs of all world info entries in the given folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      description: |-2
        Returns the `uid`s of all world info entries that are in the world info folder with the given `uid`.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesUIDsSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in koboldai_vars.wifolders_u[uid]:
        if wi["init"]:
            entries.append(wi["uid"])
    return {"entries": entries}


@api_v1.get(
    "/world_info/folders/<int(signed=True):folder_uid>/uids/<int(signed=True):entry_uid>"
)
@api_schema_wrap
def get_world_info_folders_folder_uid_uids_entry_uid(folder_uid: int, entry_uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID in the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: folder_uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
        - name: entry_uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {
        "result": entry_uid in koboldai_vars.worldinfo_u
        and koboldai_vars.worldinfo_u[entry_uid]["folder"] == folder_uid
        and koboldai_vars.worldinfo_u[entry_uid]["init"]
    }


@api_v1.get("/world_info/folders/<int(signed=True):uid>/name")
@api_schema_wrap
def get_world_info_folders_uid_name(uid: int):
    """---
    get:
      summary: Retrieve the name of the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.wifolders_d[uid]["name"]}


@api_v1.put("/world_info/folders/<int(signed=True):uid>/name")
@api_schema_wrap
def put_world_info_folders_uid_name(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the name of the world info folder with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.wifolders_d[uid]["name"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_uid(uid: int):
    """---
    get:
      summary: Retrieve information about the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntrySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    wi = koboldai_vars.worldinfo_u[uid]
    return {
        k: v
        for k, v in wi.items()
        if k not in ("init", "folder", "num")
        and (wi["selective"] or k != "keysecondary")
    }


@api_v1.get("/world_info/<int(signed=True):uid>/comment")
@api_schema_wrap
def get_world_info_uid_comment(uid: int):
    """---
    get:
      summary: Retrieve the comment of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["comment"]}


@api_v1.put("/world_info/<int(signed=True):uid>/comment")
@api_schema_wrap
def put_world_info_uid_comment(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the comment of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["comment"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/content")
@api_schema_wrap
def get_world_info_uid_content(uid: int):
    """---
    get:
      summary: Retrieve the content of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["content"]}


@api_v1.put("/world_info/<int(signed=True):uid>/content")
@api_schema_wrap
def put_world_info_uid_content(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the content of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["content"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/key")
@api_schema_wrap
def get_world_info_uid_key(uid: int):
    """---
    get:
      summary: Retrieve the keys or primary keys of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["key"]}


@api_v1.put("/world_info/<int(signed=True):uid>/key")
@api_schema_wrap
def put_world_info_uid_key(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the keys or primary keys of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["key"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/keysecondary")
@api_schema_wrap
def get_world_info_uid_keysecondary(uid: int):
    """---
    get:
      summary: Retrieve the secondary keys of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["keysecondary"]}


@api_v1.put("/world_info/<int(signed=True):uid>/keysecondary")
@api_schema_wrap
def put_world_info_uid_keysecondary(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the secondary keys of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["keysecondary"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/selective")
@api_schema_wrap
def get_world_info_uid_selective(uid: int):
    """---
    get:
      summary: Retrieve the selective mode state of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["selective"]}


@api_v1.put("/world_info/<int(signed=True):uid>/selective")
@api_schema_wrap
def put_world_info_uid_selective(body: BasicBooleanSchema, uid: int):
    """---
    put:
      summary: Set the selective mode state of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicBooleanSchema
            example:
              value: true
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["selective"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/constant")
@api_schema_wrap
def get_world_info_uid_constant(uid: int):
    """---
    get:
      summary: Retrieve the constant mode state of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    return {"value": koboldai_vars.worldinfo_u[uid]["constant"]}


@api_v1.put("/world_info/<int(signed=True):uid>/constant")
@api_schema_wrap
def put_world_info_uid_constant(body: BasicBooleanSchema, uid: int):
    """---
    put:
      summary: Set the constant mode state of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicBooleanSchema
            example:
              value: true
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    koboldai_vars.worldinfo_u[uid]["constant"] = body.value
    setgamesaved(False)
    return {}


@api_v1.post("/world_info/folders/none")
@api_schema_wrap
def post_world_info_folders_none(body: EmptySchema):
    """---
    post:
      summary: Create a new world info entry outside of a world info folder, at the end of the world info
      tags:
        - world_info
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        {api_validation_error_response}
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit(
        "from_server",
        {"cmd": "wiexpand", "data": koboldai_vars.worldinfo[-1]["num"]},
        broadcast=True,
    )
    koboldai_vars.worldinfo[-1]["init"] = True
    addwiitem(folder_uid=None)
    return {"uid": koboldai_vars.worldinfo[-2]["uid"]}


@api_v1.post("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def post_world_info_folders_uid(body: EmptySchema, uid: int):
    """---
    post:
      summary: Create a new world info entry at the end of the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit(
        "from_server",
        {"cmd": "wiexpand", "data": koboldai_vars.wifolders_u[uid][-1]["num"]},
        broadcast=True,
    )
    koboldai_vars.wifolders_u[uid][-1]["init"] = True
    addwiitem(folder_uid=uid)
    return {"uid": koboldai_vars.wifolders_u[uid][-2]["uid"]}


@api_v1.delete("/world_info/<int(signed=True):uid>")
@api_schema_wrap
def delete_world_info_uid(uid: int):
    """---
    delete:
      summary: Delete the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info entry with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    deletewi(uid)
    return {}


@api_v1.post("/world_info/folders")
@api_schema_wrap
def post_world_info_folders(body: EmptySchema):
    """---
    post:
      summary: Create a new world info folder at the end of the world info
      tags:
        - world_info
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        {api_validation_error_response}
    """
    addwifolder()
    return {"uid": koboldai_vars.wifolders_l[-1]}


@api_v1.delete("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def delete_world_info_folders_uid(uid: int):
    """---
    delete:
      summary: Delete the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folders with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "No world info folder with the given uid exists.",
                            "type": "key_error",
                        }
                    }
                ),
                mimetype="application/json",
                status=404,
            )
        )
    deletewifolder(uid)
    return {}


def _make_f_get(obj, _var_name, _name, _schema, _example_yaml_value):
    def f_get():
        """---
        get:
          summary: Retrieve the current {} setting value
          tags:
            - config
          responses:
            200:
              description: Successful request
              content:
                application/json:
                  schema: {}
                  example:
                    value: {}
        """
        _obj = {"koboldai_vars": koboldai_vars}[obj]
        if _var_name.startswith("@"):
            return {"value": _obj[_var_name[1:]]}
        else:
            return {"value": getattr(_obj, _var_name)}

    f_get.__doc__ = f_get.__doc__.format(_name, _schema, _example_yaml_value)
    return f_get


def _make_f_put(
    schema_class: Type[KoboldSchema],
    obj,
    _var_name,
    _name,
    _schema,
    _example_yaml_value,
):
    def f_put(body: schema_class):
        """---
        put:
          summary: Set {} setting to specified value
          tags:
            - config
          requestBody:
            required: true
            content:
              application/json:
                schema: {}
                example:
                  value: {}
          responses:
            200:
              description: Successful request
              content:
                application/json:
                  schema: EmptySchema
            {api_validation_error_response}
        """
        _obj = {"koboldai_vars": koboldai_vars}[obj]
        if _var_name.startswith("@"):
            _obj[_var_name[1:]] = body.value
        else:
            setattr(_obj, _var_name, body.value)
        settingschanged()
        refresh_settings()
        return {}

    f_put.__doc__ = f_put.__doc__.format(
        _name,
        _schema,
        _example_yaml_value,
        api_validation_error_response=api_validation_error_response,
    )
    return f_put


def create_config_endpoint(method="GET", schema="MemorySchema"):
    _name = globals()[schema].KoboldMeta.name
    _var_name = globals()[schema].KoboldMeta.var_name
    _route_name = globals()[schema].KoboldMeta.route_name
    _obj = globals()[schema].KoboldMeta.obj
    _example_yaml_value = globals()[schema].KoboldMeta.example_yaml_value
    _schema = schema
    f = (
        _make_f_get(_obj, _var_name, _name, _schema, _example_yaml_value)
        if method == "GET"
        else _make_f_put(
            globals()[schema], _obj, _var_name, _name, _schema, _example_yaml_value
        )
    )
    f.__name__ = f"{method.lower()}_config_{_name}"
    f = api_schema_wrap(f)
    for api in (api_v1,):
        f = api.route(f"/config/{_route_name}", methods=[method])(f)


@api_v1.get("/config/soft_prompt")
@api_schema_wrap
def get_config_soft_prompt():
    """---
    get:
      summary: Retrieve the current soft prompt name
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SoftPromptSettingSchema
              example:
                value: ""
    """
    return {"value": koboldai_vars.spfilename.strip()}


class SoftPromptsListSchema(KoboldSchema):
    values: List[SoftPromptSettingSchema] = fields.List(
        fields.Nested(SoftPromptSettingSchema),
        required=True,
        metadata={"description": "Array of available softprompts."},
    )


@api_v1.get("/config/soft_prompts_list")
@api_schema_wrap
def get_config_soft_prompts_list():
    """---
    get:
      summary: Retrieve all available softprompt filenames
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SoftPromptsListSchema
              example:
                values: []
    """
    splist = []
    for sp in fileops.getspfiles(koboldai_vars.modeldim):

        splist.append({"value": sp["filename"]})
    return {"values": splist}


@api_v1.put("/config/soft_prompt")
@api_schema_wrap
def put_config_soft_prompt(body: SoftPromptSettingSchema):
    """---
    put:
      summary: Set soft prompt by name
      tags:
        - config
      requestBody:
        required: true
        content:
          application/json:
            schema: SoftPromptSettingSchema
            example:
              value: ""
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    if koboldai_vars.allowsp:
        spRequest(body.value)
        settingschanged()
    return {}


class SamplerSeedSettingSchema(KoboldSchema):
    value: int = fields.Integer(
        validate=validate.Range(min=0, max=2**64 - 1), required=True
    )


@api_v1.get("/config/sampler_seed")
@api_schema_wrap
def get_config_sampler_seed():
    """---
    get:
      summary: Retrieve the current global sampler seed value
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SamplerSeedSettingSchema
              example:
                value: 3475097509890965500
    """
    return {
        "value": __import__("tpu_mtj_backend").get_rng_seed()
        if koboldai_vars.use_colab_tpu
        else __import__("torch").initial_seed()
    }


@api_v1.put("/config/sampler_seed")
@api_schema_wrap
def put_config_sampler_seed(body: SamplerSeedSettingSchema):
    """---
    put:
      summary: Set the global sampler seed value
      tags:
        - config
      requestBody:
        required: true
        content:
          application/json:
            schema: SamplerSeedSettingSchema
            example:
              value: 3475097509890965500
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    if koboldai_vars.use_colab_tpu:
        import tpu_mtj_backend

        tpu_mtj_backend.socketio = socketio
        tpu_mtj_backend.set_rng_seed(body.value)
    else:
        import torch

        torch.manual_seed(body.value)
    koboldai_vars.seed = body.value
    return {}


def _generate_text(body: GenerationInputSchema):
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": "Server is busy; please try again later.",
                            "type": "service_unavailable",
                        }
                    }
                ),
                mimetype="application/json",
                status=503,
            )
        )
    if koboldai_vars.use_colab_tpu:
        import tpu_mtj_backend

        tpu_mtj_backend.socketio = socketio
    if hasattr(body, "sampler_seed"):
        # If a seed was specified, we need to save the global RNG state so we
        # can restore it later
        old_seed = koboldai_vars.seed
        old_rng_state = (
            tpu_mtj_backend.get_rng_state()
            if koboldai_vars.use_colab_tpu
            else torch.get_rng_state()
        )
        koboldai_vars.seed = body.sampler_seed
        # We should try to use a previously saved RNG state with the same seed
        if body.sampler_seed in koboldai_vars.rng_states:
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(
                    koboldai_vars.rng_states[body.sampler_seed]
                )
            else:
                torch.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
        else:
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(
                    tpu_mtj_backend.new_rng_state(body.sampler_seed)
                )
            else:
                torch.manual_seed(body.sampler_seed)
        koboldai_vars.rng_states[body.sampler_seed] = (
            tpu_mtj_backend.get_rng_state()
            if koboldai_vars.use_colab_tpu
            else torch.get_rng_state()
        )
    if hasattr(body, "sampler_order"):
        if len(body.sampler_order) < 7:
            body.sampler_order = [6] + body.sampler_order
    # This maps each property of the setting to use when sending the generate idempotently
    # To the object which typically contains it's value
    # This allows to set the property only for the API generation, and then revert the setting
    # To what it was before.
    mapping = {
        "disable_input_formatting": ("koboldai_vars", "disable_input_formatting", None),
        "disable_output_formatting": (
            "koboldai_vars",
            "disable_output_formatting",
            None,
        ),
        "rep_pen": ("koboldai_vars", "rep_pen", None),
        "rep_pen_range": ("koboldai_vars", "rep_pen_range", None),
        "rep_pen_slope": ("koboldai_vars", "rep_pen_slope", None),
        "top_k": ("koboldai_vars", "top_k", None),
        "top_a": ("koboldai_vars", "top_a", None),
        "top_p": ("koboldai_vars", "top_p", None),
        "tfs": ("koboldai_vars", "tfs", None),
        "typical": ("koboldai_vars", "typical", None),
        "temperature": ("koboldai_vars", "temp", None),
        "frmtadsnsp": ("koboldai_vars", "frmtadsnsp", "input"),
        "frmttriminc": ("koboldai_vars", "frmttriminc", "output"),
        "frmtrmblln": ("koboldai_vars", "frmtrmblln", "output"),
        "frmtrmspch": ("koboldai_vars", "frmtrmspch", "output"),
        "singleline": ("koboldai_vars", "singleline", "output"),
        "max_length": ("koboldai_vars", "genamt", None),
        "max_context_length": ("koboldai_vars", "max_length", None),
        "n": ("koboldai_vars", "numseqs", None),
        "quiet": ("koboldai_vars", "quiet", None),
        "sampler_order": ("koboldai_vars", "sampler_order", None),
        "sampler_full_determinism": ("koboldai_vars", "full_determinism", None),
        "stop_sequence": ("koboldai_vars", "stop_sequence", None),
    }
    saved_settings = {}
    set_aibusy(True)
    disable_set_aibusy = koboldai_vars.disable_set_aibusy
    koboldai_vars.disable_set_aibusy = True
    _standalone = koboldai_vars.standalone
    koboldai_vars.standalone = True
    show_probs = koboldai_vars.show_probs
    koboldai_vars.show_probs = False
    output_streaming = koboldai_vars.output_streaming
    koboldai_vars.output_streaming = False
    for key, entry in mapping.items():
        obj = {"koboldai_vars": koboldai_vars}[entry[0]]
        if (
            entry[2] == "input"
            and koboldai_vars.disable_input_formatting
            and not hasattr(body, key)
        ):
            setattr(body, key, False)
        if (
            entry[2] == "output"
            and koboldai_vars.disable_output_formatting
            and not hasattr(body, key)
        ):
            setattr(body, key, False)
        if getattr(body, key, None) is not None:
            if entry[1].startswith("@"):
                saved_settings[key] = obj[entry[1][1:]]
                obj[entry[1][1:]] = getattr(body, key)
            else:
                saved_settings[key] = getattr(obj, entry[1])
                setattr(obj, entry[1], getattr(body, key))
    try:
        if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            if any(q in body.soft_prompt for q in ("/", "\\")):
                raise RuntimeError
            old_spfilename = koboldai_vars.spfilename
            spRequest(body.soft_prompt.strip())
        genout = apiactionsubmit(
            body.prompt,
            use_memory=body.use_memory,
            use_story=body.use_story,
            use_world_info=body.use_world_info,
            use_authors_note=body.use_authors_note,
        )
        output = {"results": [{"text": txt} for txt in genout]}
    finally:
        for key in saved_settings:
            entry = mapping[key]
            obj = {"koboldai_vars": koboldai_vars}[entry[0]]
            if getattr(body, key, None) is not None:
                if entry[1].startswith("@"):
                    if obj[entry[1][1:]] == getattr(body, key):
                        obj[entry[1][1:]] = saved_settings[key]
                else:
                    if getattr(obj, entry[1]) == getattr(body, key):
                        setattr(obj, entry[1], saved_settings[key])
        koboldai_vars.disable_set_aibusy = disable_set_aibusy
        koboldai_vars.standalone = _standalone
        koboldai_vars.show_probs = show_probs
        koboldai_vars.output_streaming = output_streaming
        if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            spRequest(old_spfilename)
        if hasattr(body, "sampler_seed"):
            koboldai_vars.seed = old_seed
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(old_rng_state)
            else:
                torch.set_rng_state(old_rng_state)
        set_aibusy(False)
    return output


for schema in config_endpoint_schema:
    create_config_endpoint(schema=schema.__name__, method="GET")
    create_config_endpoint(schema=schema.__name__, method="PUT")
