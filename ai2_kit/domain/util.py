import io
import re

def __export_remote_functions():

    def substitute_vars(string, vars: dict):
        # Define a regular expression for placeholders
        placeholder_pattern = re.compile(r"\$\{?(\w+)(?:-(\w+))?\}?")
        # Define a function to replace the placeholders with keyword arguments
        def replace_placeholder(match):
            # Get the variable name and the default value from the match object
            var_name = match.group(1)
            default_value = match.group(2)
            # Check if the variable name is in the keyword arguments
            if var_name in vars:
                # Return the value of the keyword argument
                return vars[var_name]
            else:
                # Check if there is a default value
                if default_value is not None:
                    # Return the default value
                    return default_value
                else:
                    # Raise an exception if there is no default value
                    raise ValueError(f"Missing keyword argument: {var_name}")
        # Return the string with the placeholders replaced
        return placeholder_pattern.sub(replace_placeholder, string)

    def process_cp2k_macro(fp):
        # Define a regular expression for @SET directive with case insensitive flag
        set_pattern = re.compile(r"@SET\s+(\w+)\s+(.+)", re.IGNORECASE) # Added re.IGNORECASE flag
        # Initialize an empty dictionary for variables and a list for output lines
        variables = {}
        output_lines = []
        # Read line by line from the input file object
        for line in fp:
            # Strip whitespace and comments
            line = line.strip()
            line = line.split("#")[0]
            # Skip empty lines
            if not line:
                continue
            # Match @SET directive
            set_match = set_pattern.match(line)
            if set_match:
                # Get the variable name and value
                var_name = set_match.group(1)
                var_value = set_match.group(2).strip()

                # Check if the variable name is valid
                if var_name[0].isdigit():
                    # Raise an exception if the variable name starts with a number
                    raise ValueError(f"Invalid variable name: {var_name}")
                else:
                    # Assign the value to the variable in the dictionary
                    variables[var_name] = var_value
                # Skip the @SET line
                continue

            # Append the line to the output list
            output_lines.append(line)

        # Return the variables dictionary and the output list as a single string with newline characters
        return variables, "\n".join(output_lines)

    def parse_cp2k_input(fp):
        # Initialize an empty dictionary and a stack
        output = {}
        stack = []
        # Open the input file and read line by line
        for line in fp:
            # Strip whitespace and comments
            line = line.strip()
            line = line.split("#")[0]
            # Skip empty lines
            if not line:
                continue
            # Check if the line starts with &
            if line.startswith("&"):
                # Get the keyword name and strip the trailing whitespace
                keyword = line[1:].strip()
                # Split the keyword by whitespace and check if the first token is END
                tokens = keyword.split()
                if tokens[0] and tokens[0].upper() == "END":
                    # Pop the last section from the stack
                    stack.pop()
                else:
                    # If the stack is empty, add the keyword to the output dictionary
                    if not stack:
                        output[keyword] = {}
                        stack.append(keyword)
                    else:
                        # If the stack is not empty, get the current section from the output dictionary
                        current_section = output
                        for section in stack:
                            current_section = current_section[section]
                        # Add the keyword to the current section as a sub-dictionary
                        current_section[keyword] = {}
                        stack.append(keyword)
                # Continue to the next line
                continue
            # Split the line by whitespace
            tokens = line.split()
            # Get the value name and value
            value_name = tokens[0]
            value = " ".join(tokens[1:])
            # Get the current section from the output dictionary
            current_section = output
            for section in stack:
                current_section = current_section[section]
            # Add the value name and value to the current section
            current_section[value_name] = value
        return output

    # TODO: handle coords
    def load_cp2k_input(fp):
        variables, processed_text = process_cp2k_macro(fp)
        substituted_text = substitute_vars(processed_text, variables)
        return parse_cp2k_input(io.StringIO(substituted_text))

    def loads_cp2k_input(text):
        return load_cp2k_input(io.StringIO(text))

    # TODO: handle coords
    def dumps_cp2k_input(input_dict):
        # Initialize an empty list for output lines
        output_lines = []
        # Define a helper function to recursively dump the sections and values
        def dump_section(section_dict, indent=0):
            # Loop through the keys and values in the section dictionary
            for key, value in section_dict.items():
                # Check if the value is a sub-dictionary
                if isinstance(value, dict):
                    # Add a line with the section name and indentation
                    output_lines.append(" " * indent + f"&{key}")
                    # Recursively dump the sub-section with increased indentation
                    dump_section(value, indent + 3)
                    # Add a line with the end of section and indentation
                    output_lines.append(" " * indent + "&END")
                else:
                    # Add a line with the value name and value and indentation
                    output_lines.append(" " * indent + f"{key}  {value}")
        # Dump the input dictionary using the helper function
        dump_section(input_dict)
        # Return the output list as a single string with newline characters
        return "\n".join(output_lines)

    def dump_cp2k_input(input_dict, fp):
        fp.write(dumps_cp2k_input(input_dict))

    return dump_cp2k_input, dumps_cp2k_input, load_cp2k_input, loads_cp2k_input,


(
    dump_cp2k_input, dumps_cp2k_input,
    load_cp2k_input, loads_cp2k_input,
) = __export_remote_functions()
