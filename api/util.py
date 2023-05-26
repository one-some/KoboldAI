def new_make_min_max_attributes(validators, min_attr, max_attr) -> dict:
    # Patched apispec function that creates "exclusiveMinimum"/"exclusiveMaximum" OpenAPI attributes insteaed of "minimum"/"maximum" when using validators.Range or validators.Length with min_inclusive=False or max_inclusive=False
    attributes = {}
    min_list = [validator.min for validator in validators if validator.min is not None]
    max_list = [validator.max for validator in validators if validator.max is not None]
    min_inclusive_list = [getattr(validator, "min_inclusive", True) for validator in validators if validator.min is not None]
    max_inclusive_list = [getattr(validator, "max_inclusive", True) for validator in validators if validator.max is not None]
    if min_list:
        if min_attr == "minimum" and not min_inclusive_list[max(range(len(min_list)), key=min_list.__getitem__)]:
            min_attr = "exclusiveMinimum"
        attributes[min_attr] = max(min_list)
    if max_list:
        if min_attr == "maximum" and not max_inclusive_list[min(range(len(max_list)), key=max_list.__getitem__)]:
            min_attr = "exclusiveMaximum"
        attributes[max_attr] = min(max_list)
    return attributes
make_min_max_attributes.__code__ = new_make_min_max_attributes.__code__

def api_format_docstring(f):
    f.__doc__ = eval('f"""{}"""'.format(f.__doc__.replace("\\", "\\\\")))
    return f

def api_catch_out_of_memory_errors(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            if any (s in traceback.format_exc().lower() for s in ("out of memory", "not enough memory")):
                for line in reversed(traceback.format_exc().split("\n")):
                    if any(s in line.lower() for s in ("out of memory", "not enough memory")) and line.count(":"):
                        line = line.split(":", 1)[1]
                        line = re.sub(r"\[.+?\] +data\.", "", line).strip()
                        raise KoboldOutOfMemoryError("KoboldAI ran out of memory: " + line, type="out_of_memory.gpu.cuda" if "cuda out of memory" in line.lower() else "out_of_memory.gpu.hip" if "hip out of memory" in line.lower() else "out_of_memory.tpu.hbm" if "memory space hbm" in line.lower() else "out_of_memory.cpu.default_memory_allocator" if "defaultmemoryallocator" in line.lower() else "out_of_memory.unknown.unknown")
                raise KoboldOutOfMemoryError(type="out_of_memory.unknown.unknown")
            raise e
    return decorated


def api_schema_wrap(f):
    try:
        input_schema: Type[Schema] = next(
            iter(inspect.signature(f).parameters.values())
        ).annotation
    except:
        HAS_SCHEMA = False
    else:
        HAS_SCHEMA = inspect.isclass(input_schema) and issubclass(input_schema, Schema)
    f = api_format_docstring(f)
    f = api_catch_out_of_memory_errors(f)

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if HAS_SCHEMA:
            body = request.get_json()
            schema = input_schema.from_dict(input_schema().load(body))
            response = f(schema, *args, **kwargs)
        else:
            response = f(*args, **kwargs)
        if not isinstance(response, Response):
            response = jsonify(response)
        return response

    return decorated
