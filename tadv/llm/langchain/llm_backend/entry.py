from langchain_openai import ChatOpenAI

from tadv.llm.langchain.llm_backend.chat_langchain import llm_with_lc_hf


def get_langchain_model(model_name: str):
    model_name_package_map = {
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo"),
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
        "gpt-4o": ChatOpenAI(model="gpt-4o"),
        "gpt-4.5-preview": ChatOpenAI(model="gpt-4.5-preview"),
        "Phi-3-mini-4k-instruct": llm_with_lc_hf("microsoft/Phi-3-mini-4k-instruct"),
    }

    try:
        model_api = model_name_package_map[model_name]
    except KeyError:
        raise ValueError(f"Invalid model name: {model_name}")
    return model_api
