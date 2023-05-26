from __future__ import annotations

# HACK: Still relying on kaivars...
from utils import koboldai_vars
import fileops

import os
from typing import Dict, List, Optional, Type
from marshmallow import Schema, fields, validate, EXCLUDE
from marshmallow.exceptions import ValidationError


config_endpoint_schemas: List[Type[KoboldSchema]] = []


def config_endpoint_schema(c: Type[KoboldSchema]):
    config_endpoint_schemas.append(c)
    return c


class KoboldSchema(Schema):
    pass


class EmptySchema(KoboldSchema):
    pass


class BasicTextResultInnerSchema(KoboldSchema):
    text: str = fields.String(required=True)


class BasicTextResultSchema(KoboldSchema):
    result: BasicTextResultInnerSchema = fields.Nested(BasicTextResultInnerSchema)


class BasicResultInnerSchema(KoboldSchema):
    result: str = fields.String(required=True)


class BasicResultSchema(KoboldSchema):
    result: BasicResultInnerSchema = fields.Nested(
        BasicResultInnerSchema, required=True
    )


class BasicResultsSchema(KoboldSchema):
    results: BasicResultInnerSchema = fields.List(
        fields.Nested(BasicResultInnerSchema), required=True
    )


class BasicStringSchema(KoboldSchema):
    value: str = fields.String(required=True)


class BasicBooleanSchema(KoboldSchema):
    value: bool = fields.Boolean(required=True)


class BasicUIDSchema(KoboldSchema):
    uid: str = fields.Integer(
        required=True,
        validate=validate.Range(min=-2147483648, max=2147483647),
        metadata={
            "description": "32-bit signed integer unique to this world info entry/folder."
        },
    )


class BasicErrorSchema(KoboldSchema):
    msg: str = fields.String(required=True)
    type: str = fields.String(required=True)


class StoryEmptyErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


class StoryTooShortErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


class OutOfMemoryErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


class NotFoundErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


api_out_of_memory_response = """507:
          description: Out of memory
          content:
            application/json:
              schema: OutOfMemoryErrorSchema
              examples:
                gpu.cuda:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 2.97 GiB already allocated; 0 bytes free; 2.99 GiB reserved in total by PyTorch)"
                      type: out_of_memory.gpu.cuda
                gpu.hip:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: HIP out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 2.97 GiB already allocated; 0 bytes free; 2.99 GiB reserved in total by PyTorch)"
                      type: out_of_memory.gpu.hip
                tpu.hbm:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: Compilation failed: Compilation failure: Ran out of memory in memory space hbm. Used 8.83G of 8.00G hbm. Exceeded hbm capacity by 848.88M."
                      type: out_of_memory.tpu.hbm
                cpu.default_cpu_allocator:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: DefaultCPUAllocator: not enough memory: you tried to allocate 209715200 bytes."
                      type: out_of_memory.cpu.default_cpu_allocator
                unknown.unknown:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory."
                      type: out_of_memory.unknown.unknown"""


class ValidationErrorSchema(KoboldSchema):
    detail: Dict[str, List[str]] = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.String(), validate=validate.Length(min=1)),
        required=True,
    )


api_validation_error_response = """422:
          description: Validation error
          content:
            application/json:
              schema: ValidationErrorSchema"""


class ServerBusyErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


api_server_busy_response = """503:
          description: Server is busy
          content:
            application/json:
              schema: ServerBusyErrorSchema
              example:
                detail:
                  msg: Server is busy; please try again later.
                  type: service_unavailable"""


class NotImplementedErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)


api_not_implemented_response = """501:
          description: Not implemented
          content:
            application/json:
              schema: NotImplementedErrorSchema
              example:
                detail:
                  msg: API generation is not supported in read-only mode; please load a model and then try again.
                  type: not_implemented"""


class SamplerSettingsSchema(KoboldSchema):
    rep_pen: Optional[float] = fields.Float(
        validate=validate.Range(min=1),
        metadata={"description": "Base repetition penalty value."},
    )
    rep_pen_range: Optional[int] = fields.Integer(
        validate=validate.Range(min=0),
        metadata={"description": "Repetition penalty range."},
    )
    rep_pen_slope: Optional[float] = fields.Float(
        validate=validate.Range(min=0),
        metadata={"description": "Repetition penalty slope."},
    )
    top_k: Optional[int] = fields.Integer(
        validate=validate.Range(min=0),
        metadata={"description": "Top-k sampling value."},
    )
    top_a: Optional[float] = fields.Float(
        validate=validate.Range(min=0),
        metadata={"description": "Top-a sampling value."},
    )
    top_p: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Top-p sampling value."},
    )
    tfs: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Tail free sampling value."},
    )
    typical: Optional[float] = fields.Float(
        validate=validate.Range(min=0, max=1),
        metadata={"description": "Typical sampling value."},
    )
    temperature: Optional[float] = fields.Float(
        validate=validate.Range(min=0, min_inclusive=False),
        metadata={"description": "Temperature value."},
    )


def soft_prompt_validator(soft_prompt: str):
    if len(soft_prompt.strip()) == 0:
        return
    if not koboldai_vars.allowsp:
        raise ValidationError("Cannot use soft prompts with current backend.")
    if any(q in soft_prompt for q in ("/", "\\")):
        return
    z, _, _, _, _ = fileops.checksp(
        "./softprompts/" + soft_prompt.strip(), koboldai_vars.modeldim
    )
    if isinstance(z, int):
        raise ValidationError("Must be a valid soft prompt name.")
    z.close()
    return True


def story_load_validator(name: str):
    if any(q in name for q in ("/", "\\")):
        return
    if len(name.strip()) == 0 or not os.path.isfile(fileops.storypath(name)):
        raise ValidationError("Must be a valid story name.")
    return True


def permutation_validator(lst: list):
    if any(not isinstance(e, int) for e in lst):
        return
    if min(lst) != 0 or max(lst) != len(lst) - 1 or len(set(lst)) != len(lst):
        raise ValidationError(
            "Must be a permutation of the first N non-negative integers, where N is the length of this array"
        )
    return True


class SoftPromptSettingSchema(KoboldSchema):
    value: str = fields.String(
        required=True,
        validate=[soft_prompt_validator, validate.Regexp(r"^[^/\\]*$")],
        metadata={
            "description": "Soft prompt name, or a string containing only whitespace for no soft prompt. If using the GET method and no soft prompt is loaded, this will always be the empty string."
        },
    )


class GenerationInputSchema(SamplerSettingsSchema):
    prompt: str = fields.String(
        required=True, metadata={"description": "This is the submission."}
    )
    use_memory: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the memory from the KoboldAI GUI when generating text."
        },
    )
    use_story: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the story from the KoboldAI GUI when generating text."
        },
    )
    use_authors_note: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the author's note from the KoboldAI GUI when generating text. This has no effect unless `use_story` is also enabled."
        },
    )
    use_world_info: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the world info from the KoboldAI GUI when generating text."
        },
    )
    use_userscripts: bool = fields.Boolean(
        load_default=False,
        metadata={
            "description": "Whether or not to use the userscripts from the KoboldAI GUI when generating text."
        },
    )
    soft_prompt: Optional[str] = fields.String(
        metadata={
            "description": "Soft prompt to use when generating. If set to the empty string or any other string containing no non-whitespace characters, uses no soft prompt."
        },
        validate=[soft_prompt_validator, validate.Regexp(r"^[^/\\]*$")],
    )
    max_length: int = fields.Integer(
        validate=validate.Range(min=1, max=512),
        metadata={"description": "Number of tokens to generate."},
    )
    max_context_length: int = fields.Integer(
        validate=validate.Range(min=1),
        metadata={"description": "Maximum number of tokens to send to the model."},
    )
    n: int = fields.Integer(
        validate=validate.Range(min=1, max=5),
        metadata={"description": "Number of outputs to generate."},
    )
    disable_output_formatting: bool = fields.Boolean(
        load_default=True,
        metadata={
            "description": "When enabled, all output formatting options default to `false` instead of the value in the KoboldAI GUI."
        },
    )
    frmttriminc: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes some characters from the end of the output such that the output doesn't end in the middle of a sentence. If the output is less than one sentence long, does nothing.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    frmtrmblln: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    frmtrmspch: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes `#/@%{}+=~|\^<>` from the output.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    singleline: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Output formatting option. When enabled, removes everything after the first line of the output, including the newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    disable_input_formatting: bool = fields.Boolean(
        load_default=True,
        metadata={
            "description": "When enabled, all input formatting options default to `false` instead of the value in the KoboldAI GUI"
        },
    )
    frmtadsnsp: Optional[bool] = fields.Boolean(
        metadata={
            "description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action.\n\nIf `disable_input_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."
        }
    )
    quiet: Optional[bool] = fields.Boolean(
        metadata={
            "description": "When enabled, Generated output will not be displayed in the console."
        }
    )
    sampler_order: Optional[List[int]] = fields.List(
        fields.Integer(),
        validate=[validate.Length(min=6), permutation_validator],
        metadata={
            "description": "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers."
        },
    )
    sampler_seed: Optional[int] = fields.Integer(
        validate=validate.Range(min=0, max=2**64 - 1),
        metadata={
            "description": "RNG seed to use for sampling. If not specified, the global RNG will be used."
        },
    )
    sampler_full_determinism: Optional[bool] = fields.Boolean(
        metadata={
            "description": "If enabled, the generated text will always be the same as long as you use the same RNG seed, input and settings. If disabled, only the *sequence* of generated texts that you get when repeatedly generating text will be the same given the same RNG seed, input and settings."
        }
    )
    stop_sequence: Optional[List[str]] = fields.List(
        fields.String(),
        metadata={
            "description": "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence."
        },
        validate=[validate.Length(max=10)],
    )


class GenerationResultSchema(KoboldSchema):
    text: str = fields.String(
        required=True, metadata={"description": "Generated output as plain text."}
    )


class GenerationOutputSchema(KoboldSchema):
    results: List[GenerationResultSchema] = fields.List(
        fields.Nested(GenerationResultSchema),
        required=True,
        metadata={"description": "Array of generated outputs."},
    )


class StoryNumsChunkSchema(KoboldSchema):
    num: int = fields.Integer(
        required=True,
        metadata={
            "description": "Guaranteed to not equal the `num` of any other active story chunk. Equals 0 iff this is the first action of the story (the prompt)."
        },
    )


class StoryChunkSchema(StoryNumsChunkSchema, KoboldSchema):
    text: str = fields.String(
        required=True, metadata={"description": "The text inside this story chunk."}
    )


class StorySchema(KoboldSchema):
    results: List[StoryChunkSchema] = fields.List(
        fields.Nested(StoryChunkSchema),
        required=True,
        metadata={
            "description": "Array of story actions. The array is sorted such that actions closer to the end of this array are closer to the end of the story."
        },
    )


class BasicBooleanResultSchema(KoboldSchema):
    result: bool = fields.Boolean(required=True)


class StoryNumsSchema(KoboldSchema):
    results: List[int] = fields.List(
        fields.Integer(),
        required=True,
        metadata={
            "description": "Array of story action nums. The array is sorted such that actions closer to the end of this array are closer to the end of the story."
        },
    )


class StoryChunkResultSchema(KoboldSchema):
    result: StoryChunkSchema = fields.Nested(StoryChunkSchema, required=True)


class StoryChunkNumSchema(KoboldSchema):
    value: int = fields.Integer(required=True)


class StoryChunkTextSchema(KoboldSchema):
    value: str = fields.String(required=True)


class StoryChunkSetTextSchema(KoboldSchema):
    value: str = fields.String(required=True, validate=validate.Regexp(r"^(.|\n)*\S$"))


class StoryLoadSchema(KoboldSchema):
    name: str = fields.String(
        required=True, validate=[story_load_validator, validate.Regexp(r"^[^/\\]*$")]
    )


class StorySaveSchema(KoboldSchema):
    name: str = fields.String(
        required=True, validate=validate.Regexp(r"^(?=.*\S)(?!.*[/\\]).*$")
    )


class WorldInfoEntrySchema(KoboldSchema):
    uid: int = fields.Integer(
        required=True,
        validate=validate.Range(min=-2147483648, max=2147483647),
        metadata={
            "description": "32-bit signed integer unique to this world info entry."
        },
    )
    content: str = fields.String(
        required=True,
        metadata={"description": 'The "What To Remember" for this entry.'},
    )
    key: str = fields.String(
        required=True,
        metadata={
            "description": "Comma-separated list of keys, or of primary keys if selective mode is enabled."
        },
    )
    keysecondary: str = fields.String(
        metadata={
            "description": "Comma-separated list of secondary keys if selective mode is enabled."
        }
    )
    selective: bool = fields.Boolean(
        required=True,
        metadata={
            "description": "Whether or not selective mode is enabled for this world info entry."
        },
    )
    constant: bool = fields.Boolean(
        required=True,
        metadata={
            "description": "Whether or not constant mode is enabled for this world info entry."
        },
    )
    comment: bool = fields.String(
        required=True,
        metadata={
            "description": "The comment/description/title for this world info entry."
        },
    )


class WorldInfoEntryResultSchema(KoboldSchema):
    result: WorldInfoEntrySchema = fields.Nested(WorldInfoEntrySchema, required=True)


class WorldInfoFolderBasicSchema(KoboldSchema):
    uid: int = fields.Integer(
        required=True,
        validate=validate.Range(min=-2147483648, max=2147483647),
        metadata={
            "description": "32-bit signed integer unique to this world info folder."
        },
    )
    name: str = fields.String(
        required=True, metadata={"description": "Name of this world info folder."}
    )


class WorldInfoFolderSchema(WorldInfoFolderBasicSchema):
    entries: List[WorldInfoEntrySchema] = fields.List(
        fields.Nested(WorldInfoEntrySchema), required=True
    )


class WorldInfoFolderUIDsSchema(KoboldSchema):
    uid: int = fields.Integer(
        required=True,
        validate=validate.Range(min=-2147483648, max=2147483647),
        metadata={
            "description": "32-bit signed integer unique to this world info folder."
        },
    )
    entries: List[int] = fields.List(
        fields.Integer(
            required=True,
            validate=validate.Range(min=-2147483648, max=2147483647),
            metadata={
                "description": "32-bit signed integer unique to this world info entry."
            },
        ),
        required=True,
    )


class WorldInfoEntriesSchema(KoboldSchema):
    entries: List[WorldInfoEntrySchema] = fields.List(
        fields.Nested(WorldInfoEntrySchema), required=True
    )


class WorldInfoFoldersSchema(KoboldSchema):
    folders: List[WorldInfoFolderBasicSchema] = fields.List(
        fields.Nested(WorldInfoFolderBasicSchema), required=True
    )


class WorldInfoSchema(WorldInfoEntriesSchema):
    folders: List[WorldInfoFolderSchema] = fields.List(
        fields.Nested(WorldInfoFolderSchema), required=True
    )


class WorldInfoEntriesUIDsSchema(KoboldSchema):
    entries: List[int] = fields.List(
        fields.Integer(
            required=True,
            validate=validate.Range(min=-2147483648, max=2147483647),
            metadata={
                "description": "32-bit signed integer unique to this world info entry."
            },
        ),
        required=True,
    )


class WorldInfoFoldersUIDsSchema(KoboldSchema):
    folders: List[int] = fields.List(
        fields.Integer(
            required=True,
            validate=validate.Range(min=-2147483648, max=2147483647),
            metadata={
                "description": "32-bit signed integer unique to this world info folder."
            },
        ),
        required=True,
    )


class WorldInfoUIDsSchema(WorldInfoEntriesUIDsSchema):
    folders: List[WorldInfoFolderSchema] = fields.List(
        fields.Nested(WorldInfoFolderUIDsSchema), required=True
    )


class ModelSelectionSchema(KoboldSchema):
    model: str = fields.String(
        required=True,
        validate=validate.Regexp(
            r"^(?!\s*NeoCustom)(?!\s*GPT2Custom)(?!\s*TPUMeshTransformerGPTJ)(?!\s*TPUMeshTransformerGPTNeoX)(?!\s*GooseAI)(?!\s*OAI)(?!\s*InferKit)(?!\s*Colab)(?!\s*API).*$"
        ),
        metadata={
            "description": 'Hugging Face model ID, the path to a model folder (relative to the "models" folder in the KoboldAI root folder) or "ReadOnly" for no model'
        },
    )


@config_endpoint_schema
class MemorySettingSchema(KoboldSchema):
    value = fields.String(required=True)

    class KoboldMeta:
        route_name = "memory"
        obj = "koboldai_vars"
        var_name = "memory"
        name = "memory"
        example_yaml_value = "Memory"


@config_endpoint_schema
class AuthorsNoteSettingSchema(KoboldSchema):
    value = fields.String(required=True)

    class KoboldMeta:
        route_name = "authors_note"
        obj = "koboldai_vars"
        var_name = "authornote"
        name = "author's note"
        example_yaml_value = "''"


@config_endpoint_schema
class AuthorsNoteTemplateSettingSchema(KoboldSchema):
    value = fields.String(required=True)

    class KoboldMeta:
        route_name = "authors_note_template"
        obj = "koboldai_vars"
        var_name = "authornotetemplate"
        name = "author's note template"
        example_yaml_value = '"[Author\'s note: <|>]"'


@config_endpoint_schema
class TopKSamplingSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0), required=True)

    class KoboldMeta:
        route_name = "top_k"
        obj = "koboldai_vars"
        var_name = "top_k"
        name = "top-k sampling"
        example_yaml_value = "0"


@config_endpoint_schema
class TopASamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0), required=True)

    class KoboldMeta:
        route_name = "top_a"
        obj = "koboldai_vars"
        var_name = "top_a"
        name = "top-a sampling"
        example_yaml_value = "0.0"


@config_endpoint_schema
class TopPSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)

    class KoboldMeta:
        route_name = "top_p"
        obj = "koboldai_vars"
        var_name = "top_p"
        name = "top-p sampling"
        example_yaml_value = "0.9"


@config_endpoint_schema
class TailFreeSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)

    class KoboldMeta:
        route_name = "tfs"
        obj = "koboldai_vars"
        var_name = "tfs"
        name = "tail free sampling"
        example_yaml_value = "1.0"


@config_endpoint_schema
class TypicalSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)

    class KoboldMeta:
        route_name = "typical"
        obj = "koboldai_vars"
        var_name = "typical"
        name = "typical sampling"
        example_yaml_value = "1.0"


@config_endpoint_schema
class TemperatureSamplingSettingSchema(KoboldSchema):
    value = fields.Float(
        validate=validate.Range(min=0, min_inclusive=False), required=True
    )

    class KoboldMeta:
        route_name = "temperature"
        obj = "koboldai_vars"
        var_name = "temp"
        name = "temperature"
        example_yaml_value = "0.5"


@config_endpoint_schema
class GensPerActionSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0, max=5), required=True)

    class KoboldMeta:
        route_name = "n"
        obj = "koboldai_vars"
        var_name = "numseqs"
        name = "Gens Per Action"
        example_yaml_value = "1"


@config_endpoint_schema
class MaxLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=512), required=True)

    class KoboldMeta:
        route_name = "max_length"
        obj = "koboldai_vars"
        var_name = "genamt"
        name = "max length"
        example_yaml_value = "80"


@config_endpoint_schema
class WorldInfoDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)

    class KoboldMeta:
        route_name = "world_info_depth"
        obj = "koboldai_vars"
        var_name = "widepth"
        name = "world info depth"
        example_yaml_value = "3"


@config_endpoint_schema
class AuthorsNoteDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)

    class KoboldMeta:
        route_name = "authors_note_depth"
        obj = "koboldai_vars"
        var_name = "andepth"
        name = "author's note depth"
        example_yaml_value = "3"


@config_endpoint_schema
class MaxContextLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=512, max=2048), required=True)

    class KoboldMeta:
        route_name = "max_context_length"
        obj = "koboldai_vars"
        var_name = "max_length"
        name = "max context length"
        example_yaml_value = "2048"


@config_endpoint_schema
class TrimIncompleteSentencesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "frmttriminc"
        obj = "koboldai_vars"
        var_name = "frmttriminc"
        name = "trim incomplete sentences (output formatting)"
        example_yaml_value = "false"


@config_endpoint_schema
class RemoveBlankLinesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "frmtrmblln"
        obj = "koboldai_vars"
        var_name = "frmtrmblln"
        name = "remove blank lines (output formatting)"
        example_yaml_value = "false"


@config_endpoint_schema
class RemoveSpecialCharactersSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "frmtrmspch"
        obj = "koboldai_vars"
        var_name = "frmtrmspch"
        name = "remove special characters (output formatting)"
        example_yaml_value = "false"


@config_endpoint_schema
class SingleLineSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "singleline"
        obj = "koboldai_vars"
        var_name = "singleline"
        name = "single line (output formatting)"
        example_yaml_value = "false"


@config_endpoint_schema
class AddSentenceSpacingSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "frmtadsnsp"
        obj = "koboldai_vars"
        var_name = "frmtadsnsp"
        name = "add sentence spacing (input formatting)"
        example_yaml_value = "false"


@config_endpoint_schema
class SamplerOrderSettingSchema(KoboldSchema):
    value = fields.List(
        fields.Integer(),
        validate=[validate.Length(min=6), permutation_validator],
        required=True,
    )

    class KoboldMeta:
        route_name = "sampler_order"
        obj = "koboldai_vars"
        var_name = "sampler_order"
        name = "sampler order"
        example_yaml_value = "[6, 0, 1, 2, 3, 4, 5]"


@config_endpoint_schema
class SamplerFullDeterminismSettingSchema(KoboldSchema):
    value = fields.Boolean(required=True)

    class KoboldMeta:
        route_name = "sampler_full_determinism"
        obj = "koboldai_vars"
        var_name = "full_determinism"
        name = "sampler full determinism"
        example_yaml_value = "false"
