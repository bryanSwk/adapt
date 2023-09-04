from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings

import json
import logging

import fastapi
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from src.func.search import search_arxiv, search_google

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI()

@app.on_event("startup")
def startup_event() -> None:
    """
    Startup event when fastAPI server starts up
    -------------------------------------------

    """
    logging.info("Starting server...")

    app.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    app.llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    n_ctx=2048,
    n_batch=512,
    # n_gpu_layers=50,
    n_threads=16,
    callback_manager=app.callback_manager,
    lora_base="./adapters/ggml-adapter-model.bin",
    # verbose=True,
)
    app.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 model_kwargs={'device': 'cuda'})

    logging.info("Start server complete")


functions = """{
    "function": "search_google",
    "description": "Search the web for content on Google. This allows users to search online/the internet/the web for content.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}

{
    "function": "search_arxiv",
    "description": "Search for research papers on ArXiv. Make use of AND, OR and NOT operators as appropriate to join terms within the query.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}"""

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
    temperature=0,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = app.llm(
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

@app.post("/", status_code=fastapi.status.HTTP_200_OK)
async def main_call(query: str):

    counter = 0
    response = generate_text(create_prompt_template(query, function_list=functions))
    print(f"Response:{response}")

    while is_json(response) and counter < 5:
        print("Function Call..")
        fn_json = json.loads(response)
        call_function = function_mapping.get(fn_json['function'], lambda *args, **kwargs: [])
        results, source = call_function(fn_json['arguments']['query'], app.embedder)
        response = generate_text(create_prompt_template(query, function_list=functions, context=results))
        print(response, source)
        counter+=1

    else:
        print("End")

    try:
        result_msg = {"message": f"{response}",
                      "status": "success"}
        return json.dumps(result_msg)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":

    uvicorn.run(
        "run_fastapi:app",
        host="127.0.0.1",
        reload=False,
        port=8000,
        log_level=None,
    )
