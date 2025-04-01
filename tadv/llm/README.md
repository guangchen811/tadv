#### LLM

Currently, we use [langchain](https://www.langchain.com/) as the tool for llm api calls. We plan to extend it
to [dspy](https://dspy-docs.vercel.app/) in the future.

As shown in the following figure, we decompose the data validation task into two three sub-tasks:

- Column access detection: detect the accessed column that needs to be validated based on the downstream queries or
  machine learning pipelines.
- Assumption generation: generate assumptions based on the accessed column and the context information.
- Rule generation: generate formal rules in the form
  of [deequ](https://github.com/awslabs/python-deequ/blob/master/pydeequ/checks.py) for evaluation.

The prompts during the API calls can be found [here](/tadv/llm/langchain/_prompt.py). For more details, you
can look at the [test case](/tests/llm/langchain).

TODO: add more details.
