from flask import Flask, request, jsonify
import os
import time

from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings


from pydantic import BaseModel
from pathlib import Path

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


import logging


class DataConfig(BaseModel):
    folder: Path = Path("/Users/currocobo/Documents/Proyectos Intor/intor_prototyping-main/data")


class LLMConfig(BaseModel):
    model: str = "deepseek-r1:7b"


class Config(BaseModel):
    data: DataConfig = DataConfig()
    llm: LLMConfig = LLMConfig()


llm = LLMConfig(model="llama3")
config = Config(llm=llm)


# Global objects/variables that will be loaded during initialization
vectorstore = None
llm_model = None


def load_initialization_data():
    """
    Load any data, models, or resources needed by your application.
    This could include loading ML models, database connections, etc.
    """
    global vectorstore, llm_model, config

    print("Starting initialization...")

    try:

        collection_name = "new_collection"
        persist_directory = (
            "/Users/currocobo/Documents/Proyectos Intor/intor_prototyping-main/data/vectorstore"
        )

        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Initialize the vector store
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=local_embeddings,
            persist_directory=persist_directory,
        )

        llm_model = ChatOllama(
            model=config.llm.model,
        )

        print("Initialization complete!")

    except Exception as e:
        print(f"Error during initialization: {e}")
        return False
    else:
        return True


# Create the Flask application
app = Flask(__name__)


@app.route("/query", methods=["POST"])
def query():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the query from the request
    user_query = data.get("query", "")

    # Process the query using our loaded resources
    answer = process_query(user_query)

    # Return the response
    return jsonify({"answer": answer})


def process_query(query: str):
    """
    Process the user query and return an answer using the loaded resources.
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

    # Run
    logging.info(f"Running chain with query: {query}")
    return chain.invoke({"context": docs, "question": query})


if __name__ == "__main__":
    # Load data at startup (eager loading)
    success = load_initialization_data()
    if not success:
        print("Failed to initialize. Exiting.")
        exit(1)

    # Choose a different port to avoid conflicts with AirPlay (5000)
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
