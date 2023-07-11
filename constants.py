import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

SYSTEM_PROMPT = """
You are an AI that acts as a search engine in personal documents.
You have to write answer
acting as all the informations retrieved from the user data are yours.
Mention the document source.

The informations that you have to use for generating the answer to the user requests are the following:

{0}
"""

SYSTEM_SOURCE_TEMPLATE="""
Filename: {source}
Page: {page}
Content:
---
{content}
---
"""