def llm_with_lc_hf(model_name: str):
    """
    Create llm with langchain huggingface pipeline
    """
    raise NotImplementedError("will bring this back after huggingface pipeline is fixed")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # pipe = pipeline(
    #     "text-generation",
    #     model=hf_model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=100,
    #     top_k=10,
    #     temperature=0.6,
    #     return_full_text=False,
    # )
    # hf_llm = HuggingFacePipeline(pipeline=pipe)
    # chat_hf = ChatHuggingFace(llm=hf_llm)
    # return chat_hf
