import ast
import re


class DeequConstraint:
    def __init__(self, constraint_type: str, params: dict, hint: str = None):
        self.constraint_type = constraint_type
        self.params = params
        self.hint = hint

    def __repr__(self):
        repr_ = f"DeequConstraint(constraint_type='{self.constraint_type}'"
        if self.params:
            params_str = ', '.join(f"{k}={repr(v)}" for k, v in self.params.items())
            repr_ += f", {params_str}"
        if self.hint:
            repr_ += f", hint='{self.hint}'"
        repr_ += ")"
        return repr_

    @classmethod
    def from_string(cls, input_str):
        match = re.match(r"(\w+)\s*\((.*)\)", input_str.strip())
        if not match:
            raise ValueError(f"Invalid input string: {input_str}")

        constraint_type, args_str = match.groups()

        # Create a fake function call for AST parsing
        fake_call = f"f({args_str})"
        try:
            tree = ast.parse(fake_call, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Malformed argument string: {args_str}") from e

        # Extract keyword arguments
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Parsed expression is not a call")

        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
        return cls(
            constraint_type=constraint_type,
            params=kwargs,
            hint=kwargs.pop('hint', None) if 'hint' in kwargs else None
        )

    def to_deequ_code(self):
        hint_str = f", hint='{self.hint}'" if self.hint else ""
        params_str = ', '.join(f"{k}={repr(v)}" for k, v in self.params.items())
        return f".{self.constraint_type}({params_str}{hint_str})"
