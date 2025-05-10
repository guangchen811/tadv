from dataclasses import dataclass

from tadv.constraints_code.deequ import DeequCode


@dataclass
class GXCode:
    function_name: str
    function_args: list

    def to_deequ_code(self) -> DeequCode:
        """
        Convert the GX code to Deequ code.
        """
        # Convert the function name and arguments to a string
        function_args_str = ", ".join(
            [f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in self.function_args]
        )
        function_name_str = self.function_name.replace("gx.", "")
        # Create the Deequ code
        deequ_code = DeequCode(
            function_name=function_name_str,
            function_args=[function_args_str],
        )
