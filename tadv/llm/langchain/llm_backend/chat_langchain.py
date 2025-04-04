from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def llm_with_lc_hf(model_name: str):
    """
    Create llm with langchain huggingface pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        top_k=50,
        temperature=0.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
