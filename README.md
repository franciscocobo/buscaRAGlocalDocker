# 🔍 Aplicación para Búsqueda de Documentos usando RAG en Local

Este repositorio contiene una aplicación de búsqueda de documentos utilizando **Flask** para el backend y **Streamlit** para el frontend. También incluye CI/CD con **GitHub Actions** y dockerización para un despliegue sencillo.

---

## 📌 1. Creación del Repositorio en GitHub y Configuración en Visual Studio Code

### 🔹 Crear un repositorio en GitHub
1. Ve a [GitHub](https://github.com/) y accede a tu cuenta.
2. Haz clic en **New Repository**.
3. Asigna un nombre al repositorio, por ejemplo, `buscaRAGlocal`.
4. Selecciona **Public** o **Private**, según prefieras.
5. Haz clic en **Create repository**.

### 🔹 Iniciar el repositorio en tu máquina
Abre **Visual Studio Code** y ejecuta el siguiente comando en la terminal para clonar el repositorio:

```sh
echo "# buscaRAGlocal" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/franciscocobo/buscaRAGlocal.git
git push -u origin main
```

### 🔹 Abrir el proyecto en VS Code
1. En VS Code, haz clic en **File > Open Folder**.
2. Selecciona la carpeta `buscaRAGlocal`.

---

## 📂 2. Estructura del Proyecto

Para automatizar la creación de carpetas y archivos, puedes ejecutar el siguiente script en Bash:

```bash
#!/bin/bash

mkdir -p backend frontend .github 

touch backend/app.py backend/requirements.txt backend/data

touch frontend/app.py frontend/requirements.txt 

touch .gitignore README.md

echo "Estructura del proyecto creada exitosamente 🎉"
```

### 🔹 Para ejecutar el script:
1. Crea un archivo `setup_project.sh` y pega el código anterior.
2. Otorga permisos de ejecución:
   ```sh
   chmod +x setup_project.sh
   ```
3. Ejecuta el script:
   ```sh
   ./setup_project.sh
   ```

La estructura de carpetas quedará así:

```sh
/buscaRAGlocal
│── /backend
│   ├── app.py                # API en Flask
│   ├── requirements.txt       # Dependencias del backend
│   /data                      # Directorio para almacenar datos (vacío)
│── /frontend
│   ├── app.py                 # Interfaz en Streamlit
│   ├── requirements.txt        # Dependencias del frontend
│── .gitignore                 # Ignorar archivos innecesarios
│── README.md                  # Documentación del proyecto
```

---

## 🛠 3. Configuración del Entorno y Dependencias

Para este proyecto usaremos **Python** y un entorno virtual (`venv`).

### 🔹 Configurar el entorno virtual para el backend
```sh
cd backend
python3 -m venv venv
source venv/bin/activate   # Para macOS/Linux
# venv\Scripts\activate    # Para Windows
pip install -r requirements.txt
```

#### 🔹 Instalar y configurar Ollama
```sh
pip install ollama
ollama pull llama3
ollama pull nomic-embed-text
ollama list
```

Verificar la instalación:
```sh
python -c "import flask; import langchain; import chromadb; print('Todos los paquetes están instalados correctamente!')"
```

### 🔹 Configurar dependencias para el frontend
```sh
cd backend
python3 -m venv venv
source venv/bin/activate   # Para macOS/Linux
# venv\Scripts\activate    # Para Windows
pip install -r requirements.txt
```

---

## 🚀 4. Implementación del Backend con Flask

### 📌 **backend/app.py**

```python
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

```

### 🔹 Para ejecutar el backend
```sh
python app.py
```

---

## 🎨 5. Implementación del Frontend con Streamlit

### 📌 **frontend/app.py**

```python
import streamlit as st
import requests
import json

# Título de la aplicación
st.title("AI Chatbot")

# Inicializar el historial del chat en el estado de sesión si no existe desde el inicio
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Definir la URL del endpoint de la API (debe reemplazarse con la URL real)
API_URL = "http://127.0.0.1:5001/query"

# Entrada de chat para que el usuario escriba su pregunta
if prompt := st.chat_input("Ask a question..."):
    # Agregar el mensaje del usuario al historial del chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Preparar la solicitud a la API
    payload = {"query": prompt}

    # Mostrar un spinner mientras se espera la respuesta de la API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Realizar la solicitud a la API
                response = requests.post(
                    API_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                )

                # Verificar si la solicitud fue exitosa
                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get(
                        "answer", "Sorry, I couldn't find an answer."
                    )

                    # Mostrar la respuesta del asistente en la interfaz
                    st.write(answer)

                    # Agregar la respuesta del asistente al historial del chat
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error(f"Error: API returned status code {response.status_code}")

            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# Agregar una barra lateral con información adicional
st.sidebar.title("About")
st.sidebar.info(
    "Este es un asistente de preguntas y respuestas que utiliza una API para responder a tus preguntas. "
    "Escribe tu pregunta en el chat y obtén una respuesta instantánea!"
)

# Botón opcional en la barra lateral para limpiar la conversación
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
```
### 🔹 Para ejecutar el frontend
```sh
streamlit run app.py
```
--- 
## 🎨 6. Para dockerizar la aplicación

# Build and start the services
docker compose up --build -d

sleep 10

# Pull the Llama model
docker compose exec ollama ollama pull llama3.2:latest
sleep 2

# Check if the model was pulled successfully
curl -X POST http://localhost:5001/query -H "Content-Type: application/json" -d '{"query": "What is RAG in the context of LLMs?"}'

---


