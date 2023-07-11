#!/usr/bin/env python3
from typing import List
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document, LLMResult
from langchain.llms import GPT4All, LlamaCpp
from langchain.chat_models import ChatOpenAI
import os
import argparse
import time
from callback_handler import CustomCallbackHandler
import openai

from constants import CHROMA_SETTINGS, SYSTEM_PROMPT, SYSTEM_SOURCE_TEMPLATE

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',512))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
n_threads = int(os.cpu_count() * 0.75)
n_gpu_layers = os.environ.get('N_GPU_LAYERS')
use_mlock = os.environ.get('USE_MLOCK')
open_ai_api_key = os.environ.get('OPEN_AI_API_KEY')
openai.api_key = open_ai_api_key

temperature = os.environ.get('TEMPERATURE')
max_tokens = os.environ.get('MAX_TOKENS')
top_p = os.environ.get('TOP_P')
frequence_penalty = os.environ.get('FREQUENCY_PENALTY')
presence_penalty = os.environ.get('PRESENCE_PENALTY')

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [CustomCallbackHandler()]
    if (model_type == "OpenAI"):
        callbacks = []
    (has_callback) = len(callbacks) != 0
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False,
                           n_gpu_layers=n_gpu_layers, use_mlock=use_mlock,top_p=0.9)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False, n_threads=n_threads)
        case "OpenAI":
            llm = ChatOpenAI(model=model_path, callbacks=callbacks, temperature=0.6, max_tokens=8000, openai_api_key=open_ai_api_key, client=openai.ChatCompletion)
        case _default:
            print(f"Model {model_type} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        if (has_callback):
            callbacks[0].clear_timer()
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        if model_type == "OpenAI":
            docs: List[Document] = db.search(query, search_type="similarity")
            docs_prompt = [
                SYSTEM_SOURCE_TEMPLATE.format(
                    source=doc.metadata["source"],
                    page=doc.metadata["page"] if "page" in doc.metadata else "unknown",
                    content=doc.page_content,
                ) for doc in docs]
            system_prompt = SYSTEM_PROMPT.format("\n".join([doc for doc in docs_prompt]))
            
            response = openai.ChatCompletion.create(
                model=model_path,
                messages=[
                    {
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content": query
                    },
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                top_p=float(top_p),
                frequency_penalty=float(frequence_penalty),
                presence_penalty=float(presence_penalty),
            )
            answer = response.choices[0].message.content
            token_consumed = response.usage.total_tokens
        else:
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        requested_time = callbacks[0].get_requested_time() if has_callback else round(end - start, 2)
        print(f"\n> Answer (took {requested_time} s.):")
        print(answer)
        if (token_consumed):
            print(f"\n>Token consumed: {token_consumed}")

        # Print the relevant sources used for the answer
        if model_type != "OpenAI":
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                # print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
