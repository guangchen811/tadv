import gradio as gr
from gradio_highlightedcode import HighlightedCode

from tadv.data_models import ConstraintsWithSources
from tadv.loader import FileLoader
from tadv.utils import get_project_root


def update_highlighted_code(highlighted_code, lines):
    highlighted_lines = []
    sorted(lines, key=lambda x: x[0])
    for (start_line, end_line) in lines:
        highlight_lines = [(start_line, "#87CEEB"), (end_line, "#ffffff")]
        highlighted_lines.extend(highlight_lines)
    if highlighted_code.endswith("#"):
        used_highlighted_code = highlighted_code[:-1]
    else:
        used_highlighted_code = highlighted_code + "#"
    new_highlighted_code = HighlightedCode(
        value=used_highlighted_code, language="python", highlights=highlighted_lines, interactive=False,
    )
    return new_highlighted_code


def constraint_ui(constraints_data, code):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                highlighted_code = HighlightedCode(value=code, language="python", highlights=[], interactive=False)
            with gr.Column(scale=2):
                buttons = []
                for column_name, content in constraints_data.items():
                    with gr.Accordion(column_name, open=False):
                        with gr.Group():
                            for code_block in content['code']:
                                rule, status = code_block
                                gr.Markdown(f"`{rule}` → **{status}**")

                        with gr.Group():
                            gr.Markdown("### Assumptions")
                            for assumption in content['assumptions']:
                                text = assumption["text"]
                                sources = ", ".join(
                                    f"Lines {src['start_line']}-{src['end_line']}"
                                    for src in assumption["sources"]
                                )
                                line_numbers = []
                                for src in assumption["sources"]:
                                    line_numbers.append([src["start_line"], src["end_line"] + 1])
                                gr.Markdown(f"- {text}  \n_Sources: {sources}_")
                                btn = gr.Button("Highlight")
                                buttons.append((btn, line_numbers))
        for btn, lines in buttons:
            btn.click(
                fn=update_highlighted_code,
                inputs=[highlighted_code, gr.State(lines)],
                outputs=[highlighted_code]
            )
    return demo


def get_constraint_file_path(trick):
    return (
            get_project_root() /
            "data_processed" /
            "playground-series-s4e10" /
            "ml_inference_classification" /
            "0" /
            "constraints" /
            "classification_0" /
            f"tadv_constraints_with_scope__gpt-4o__{trick}.yaml"
    )


code_file_path = (
        get_project_root() /
        "data/playground-series-s4e10/scripts/ml_inference/classification_0.py"
)
code = FileLoader.load_py_file(code_file_path)
# Load the constraints from a YAML file
constraint_file_path = get_constraint_file_path('code_with_line_numbers')
constraint = ConstraintsWithSources.from_yaml(constraint_file_path)

demo = constraint_ui(constraint.to_dict()["constraints"], code)
demo.launch()
