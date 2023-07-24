import json
import os
from flask import Flask, send_from_directory, request, Response
from flask_cors import CORS
from langchain.vectorstores import Chroma, VectorStore
from langchain.embeddings import HuggingFaceEmbeddings


from constants import CHROMA_SETTINGS
from ingest import ingest
from privateGPT import query

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db: VectorStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

@app.route('/')
@app.route('/<path:path>')
def ui(path = 'index.html'):
  return send_from_directory('web', path)

@app.route('/api/ingest', methods=['POST'])
def ingestApi():
  ingest(db)
  return True

@app.route('/api/query', methods=['POST'])
def queryApi():
  return Response(query(db, request.json["q"], False), mimetype="application/octet-stream")


@app.route('/api/stream/query', methods=['POST'])
def streamQueryApi():
    return Response(query(db, request.json["q"], True), mimetype="application/octet-stream")

if __name__ == '__main__':
    app.run(debug=False, port=8001)