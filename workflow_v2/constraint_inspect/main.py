import gradio as gr
from gradio.components import Code

from tadv.data_models import ConstraintsWithSources
from tadv.loader import FileLoader
from tadv.utils import get_project_root

if __name__ == "__main__":
    assumption_generation_options = ['None', 'code_with_line_numbers', 'code_with_pygments_highlighting']
    selected_trick = gr.State(value='None')


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


    def load_contents(trick):
        code_file_path = (
                get_project_root() /
                "data/playground-series-s4e10/scripts/ml_inference/classification_0.py"
        )
        constraint_file_path = get_constraint_file_path(trick)
        code = FileLoader.load_py_file(code_file_path)
        constraint = ConstraintsWithSources.from_yaml(constraint_file_path)
        return code, constraint.to_yaml()


    def display_view(trick):
        return load_contents(trick)


    with gr.Blocks() as demo:
        with gr.Row():
            dropdown = gr.Dropdown(choices=assumption_generation_options, value='None',
                                   label="Assumption Generation Trick")
        with gr.Row():
            code_display = Code(label="Code", language="python")
            constraint_display = Code(label="Constraint", language="yaml")


        def update_view(trick):
            return display_view(trick)


        dropdown.change(update_view, inputs=[dropdown], outputs=[code_display, constraint_display])
        code_display.value, constraint_display.value = display_view('None')

    demo.launch()
