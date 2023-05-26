def apiactionsubmit(
    data,
    use_memory=False,
    use_world_info=False,
    use_story=False,
    use_authors_note=False,
):
    if not model or not model.capabilties.api_host:
        raise NotImplementedError(
            f"API generation isn't allowed on model '{koboldai_vars.model}'"
        )

    data = applyinputformatting(data)

    if koboldai_vars.memory != "" and koboldai_vars.memory[-1] != "\n":
        mem = koboldai_vars.memory + "\n"
    else:
        mem = koboldai_vars.memory

    if use_authors_note and koboldai_vars.authornote != "":
        anotetxt = ("\n" + koboldai_vars.authornotetemplate + "\n").replace(
            "<|>", koboldai_vars.authornote
        )
    else:
        anotetxt = ""

    MIN_STORY_TOKENS = 8
    story_tokens = []
    mem_tokens = []
    wi_tokens = []

    story_budget = (
        lambda: koboldai_vars.max_length
        - koboldai_vars.sp_length
        - koboldai_vars.genamt
        - len(tokenizer._koboldai_header)
        - len(story_tokens)
        - len(mem_tokens)
        - len(wi_tokens)
    )
    budget = lambda: story_budget() + MIN_STORY_TOKENS
    if budget() < 0:
        abort(
            Response(
                json.dumps(
                    {
                        "detail": {
                            "msg": f"Your Max Tokens setting is too low for your current soft prompt and tokenizer to handle. It needs to be at least {koboldai_vars.max_length - budget()}.",
                            "type": "token_overflow",
                        }
                    }
                ),
                mimetype="application/json",
                status=500,
            )
        )

    if use_memory:
        mem_tokens = tokenizer.encode(utils.encodenewlines(mem))[-budget() :]

    if use_world_info:
        # world_info, _ = checkworldinfo(data, force_use_txt=True, scan_story=use_story)
        world_info = koboldai_vars.worldinfo_v2.get_used_wi()
        wi_tokens = tokenizer.encode(utils.encodenewlines(world_info))[-budget() :]

    if use_story:
        if koboldai_vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[
                -budget() :
            ]

    story_tokens = (
        tokenizer.encode(utils.encodenewlines(data))[-story_budget() :] + story_tokens
    )

    if use_story:
        for i, action in enumerate(reversed(koboldai_vars.actions.values())):
            if story_budget() <= 0:
                assert story_budget() == 0
                break
            story_tokens = (
                tokenizer.encode(utils.encodenewlines(action))[-story_budget() :]
                + story_tokens
            )
            if i == koboldai_vars.andepth - 1:
                story_tokens = (
                    tokenizer.encode(utils.encodenewlines(anotetxt))[-story_budget() :]
                    + story_tokens
                )
        if not koboldai_vars.useprompt:
            story_tokens = (
                tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[
                    -budget() :
                ]
                + story_tokens
            )

    tokens = tokenizer._koboldai_header + mem_tokens + wi_tokens + story_tokens
    assert story_budget() >= 0
    minimum = len(tokens) + 1
    maximum = len(tokens) + koboldai_vars.genamt

    if not koboldai_vars.use_colab_tpu and koboldai_vars.model not in [
        "Colab",
        "API",
        "CLUSTER",
        "OAI",
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ]:
        genout = apiactionsubmit_generate(tokens, minimum, maximum)
    elif koboldai_vars.use_colab_tpu or koboldai_vars.model in (
        "TPUMeshTransformerGPTJ",
        "TPUMeshTransformerGPTNeoX",
    ):
        genout = apiactionsubmit_tpumtjgenerate(tokens, minimum, maximum)

    return genout
