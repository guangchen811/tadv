from langchain_openai import ChatOpenAI

from tadv.llm.langchain.llm_backend.chat_langchain import llm_with_lc_hf


def get_langchain_model(model_name: str):
    openai_api_model_list = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.5-preview"]
    if model_name in openai_api_model_list:
        model_api = ChatOpenAI(model_name=model_name, temperature=0.6)
    elif model_name == "meta-llama/Llama-2-7b-chat-hf":
        # cluster only, don't use this model on macbook local.
        import platform
        assert platform.system() != "Darwin", "This model should not be used on macOS."
        model_api = llm_with_lc_hf(model_name)
    else:
        raise ValueError(f"Model name {model_name} not supported.")
    return model_api
