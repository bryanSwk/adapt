def read_from_template(filepath: str, **kwargs):
    with open(filepath, 'r') as template_file:
        fstring_template = template_file.read()

    formatted_string = fstring_template
    for key, value in kwargs.items():
        formatted_string = formatted_string.replace("{" + key + "}", value)

    return formatted_string