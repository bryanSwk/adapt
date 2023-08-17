from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import json

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.ggmlv3.q5_K_M.bin",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    n_ctx= 2048,
    n_batch=126,
    callback_manager=callback_manager,
    lora_path="ggml-adapter-model.bin",
    verbose=True,
)

functions = """{{
    "function": "search_google",
    "description": "Search the web for content on Google. This allows users to search online/the internet/the web for content.",
    "arguments": [
        {{
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }}
    ]
}}

{{
    "function": "search_arxiv",
    "description": "Search for research papers on ArXiv. Make use of AND, OR and NOT operators as appropriate to join terms within the query.",
    "arguments": [
        {{
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }}
    ]
}}"""

function_mapping = {
    'search_google': search_google,
    'search_arxiv': search_arxiv
}

def create_prompt_template(input, function_list, context=None):
    if context:
        return f"""
<s>[INST] <<SYS>>
You are a helpful research assistant. The following functions are available for you to fetch further data to answer user questions, if relevant:

{function_list}

To call a function, respond - immediately and only - with a JSON object of the following format:
{{
    "function": "function_name",
    "arguments": {{
        "argument1": "argument_value",
        "argument2": "argument_value"
    }}
}}

Use the following pieces of context to help answer the question at the end. If you don't know the answer,\

just say that you don't know, don't try to make up an answer.

{context}

<</SYS>>

{input} [/INST]
"""
    else:
        return f"""
<s>[INST] <<SYS>>
You are a helpful research assistant. The following functions are available for you to fetch further data to answer user questions, if relevant:

{function_list}

To call a function, respond - immediately and only - with a JSON object of the following format:
{{
    "function": "function_name",
    "arguments": {{
        "argument1": "argument_value",
        "argument2": "argument_value"
    }}
}}
<</SYS>>

{input} [/INST]
"""

def generate_text(
    prompt,
    max_tokens=256,
    temperature=0.2,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output
    return output_text

def is_json(text):
    try:
        json.loads(text)
    except ValueError:
        return False
    return True

def main_call(query):
    response = generate_text(create_prompt_template(query, function_list=functions))
    while is_json(response):
        results_function = function_mapping.get(eg['function'], lambda *args, **kwargs: [])
        results = results_function(eg['arguments']['query'])
        response = generate_text(create_prompt_template(query, function_list=functions, context=response))