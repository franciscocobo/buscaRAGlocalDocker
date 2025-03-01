from flask import Flask, request, jsonify
import os
import logging
from pathlib import Path
from pydantic import BaseModel
import fitz  # PyMuPDF

# Librerías de LangChain para procesamiento de lenguaje y embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configurar logging para registrar eventos e información
logging.basicConfig(level=logging.INFO)

# Configuración de rutas y modelos
class DataConfig(BaseModel):
    folder: Path = Path.cwd() / "data"  # Usa el directorio de trabajo actual como base de datos
    print(folder)
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

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF usando PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)  # Abre el archivo PDF
        text = "\n".join(page.get_text("text") for page in doc)  # Extrae el texto de cada página
        return text
    except Exception as e:
        logging.error(f"Error al leer PDF {pdf_path}: {e}")
        return ""

def load_documents():
    """Carga documentos desde la carpeta y los almacena en Chroma."""
    global vectorstore

    # Utiliza un divisor de texto recursivo para fragmentar el contenido
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Tamaño máximo de cada fragmento
        chunk_overlap=100,  # Solapamiento entre fragmentos para contexto
        length_function=len,  # Método para medir la longitud del texto
        separators=["\n\n", "\n", " ", ""],  # Prioridad de separación: párrafos, líneas, espacios
    )

    all_docs = []

    # Verifica que la carpeta de documentos exista
    if not config.data.docs_folder.exists():
        logging.warning(f"La carpeta de documentos {config.data.docs_folder} no existe.")
        return

    # Itera sobre los archivos en la carpeta
    for file in config.data.docs_folder.glob("*"):
        if file.suffix == ".pdf":
            text = extract_text_from_pdf(file)
        elif file.suffix == ".txt":
            text = file.read_text(encoding="utf-8")
        else:
            continue  # Ignora otros formatos

        if text.strip():  # Si el archivo contiene texto
            docs = text_splitter.split_text(text)  # Divide el texto en fragmentos
            all_docs.extend([Document(page_content=d) for d in docs])

    # Si hay documentos, los almacena en la base de datos vectorial Chroma
    if all_docs:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=OllamaEmbeddings(model="nomic-embed-text")
        )
        logging.info(f"Se han insertado {len(all_docs)} fragmentos en la base de datos vectorial.")
    else:
        logging.warning("No se encontraron documentos para cargar.")

def load_initialization_data():
    """Inicializa la base de datos Chroma y el modelo de lenguaje."""
    global vectorstore, llm_model

    logging.info("Iniciando carga de datos...")

    try:
        collection_name = "new_collection1"

        # Carga la base de datos Chroma
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        )

        # Si la base de datos está vacía, cargar documentos
        if vectorstore._collection.count() == 0:
            logging.info("Base de datos vacía. Cargando documentos...")
            load_documents()

        # Cargar modelo de lenguaje LLM
        llm_model = ChatOllama(model=config.llm.model)

        # Verificar cantidad de documentos cargados
        doc_count = vectorstore._collection.count()
        logging.info(f"Documentos en vector store: {doc_count}")

        logging.info("Inicialización completa.")
        return True
    except Exception as e:
        logging.error(f"Error durante la inicialización: {e}")
        return False

# Creación de la aplicación Flask
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    """Endpoint para procesar consultas del usuario."""
    data = request.get_json()
    user_query = data.get("query", "")
    answer = process_query(user_query)
    return jsonify({"answer": answer})

def process_query(query: str):
    """Realiza una búsqueda en la base de datos y genera una respuesta."""
    global vectorstore, llm_model

    # Realiza una búsqueda en la base de datos vectorial
    docs = vectorstore.similarity_search(query, k=10)
    
    if not docs:
        logging.info("No se encontraron documentos relevantes.")
        return "No se encontraron documentos relevantes."

    logging.info(f"Se han recuperado {len(docs)} documentos.")

    # Formatea los documentos recuperados
    def format_docs(docs: list[Document]):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Plantilla de prompt para el modelo RAG (Retrieval-Augmented Generation)
    RAG_TEMPLATE = """
    Eres un asistente para responder preguntas usando el siguiente contexto.
    Si no sabes la respuesta, di que no la sabes. Mantén la respuesta concisa.

    <context>
    {context}
    </context>

    Pregunta:
    {question}
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | llm_model
        | StrOutputParser()
    )

    return chain.invoke({"context": docs, "question": query})

if __name__ == "__main__":
    success = load_initialization_data()
    if not success:
        logging.error("Fallo en la inicialización. Saliendo.")
        exit(1)

    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
