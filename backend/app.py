from flask import Flask, request, jsonify
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import DirectoryLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Configurar logging para registrar eventos e información
def setup_logging():
    logging.basicConfig(level=logging.INFO)
setup_logging()

# Configuración de rutas y modelos
class DataConfig(BaseModel):
    folder: Path = Path.cwd() / "data"  # Directorio de trabajo actual como base de datos
    docs_folder: Path = folder  # Carpeta donde se almacenan los documentos

class LLMConfig(BaseModel):
    model: str = "llama3:latest"  # Modelo de lenguaje a utilizar

class Config(BaseModel):
    data: DataConfig = DataConfig()
    llm: LLMConfig = LLMConfig()

config = Config()

# Variables globales para almacenar el modelo de lenguaje y la base de datos vectorial
vectorstore = None
llm_model = None

def load_documents():
    """
    Carga los documentos desde la carpeta especificada y los divide en fragmentos.
    """
    loader = DirectoryLoader(path=config.data.docs_folder)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    return text_splitter.split_documents(data)

def initialize_vectorstore():
    """
    Inicializa la base de datos vectorial con los embeddings generados.
    """
    global vectorstore
    all_splits = load_documents()
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=local_embeddings,
        collection_name="new_collection_5",
        persist_directory="vectorstore"
    )

def load_initialization_data():
    """
    Carga los datos y modelos necesarios para la aplicación.
    """
    global vectorstore, llm_model
    logging.info("Starting initialization...")
    try:
        initialize_vectorstore()
        llm_model = ChatOllama(model=config.llm.model)
        logging.info("Initialization complete!")
        return True
    except Exception as e:
        logging.error(f"Error during initialization: {e}")
        return False

# Crear la aplicación Flask
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    """
    Endpoint para recibir consultas y devolver respuestas generadas.
    """
    data = request.get_json()
    user_query = data.get("query", "")
    answer = process_query(user_query)
    return jsonify({"answer": answer})

def process_query(query: str):
    """
    Procesa la consulta del usuario utilizando la base de datos vectorial y el modelo de lenguaje.
    """
    global vectorstore, llm_model
    docs = vectorstore.search(query, search_type="similarity_score_threshold", k=10)
    RAG_TEMPLATE = """
    Eres un asistente para tareas de respuesta a preguntas. Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no la sabes. Usa un máximo de tres oraciones y mantén la respuesta concisa.  

    <context>  
    {context}  
    </context>  

    Responde la siguiente pregunta:  

    {question}
    """
    
    def format_docs(docs: list[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | llm_model
        | StrOutputParser()
    )
    
    logging.info(f"Running chain with query: {query}")
    return chain.invoke({"context": docs, "question": query})

if __name__ == "__main__":
    # Carga de datos en el arranque
    if not load_initialization_data():
        logging.error("Failed to initialize. Exiting.")
        exit(1)
    
    # Configuración del puerto y ejecución de la aplicación
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
