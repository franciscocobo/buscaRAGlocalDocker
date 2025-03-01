from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persist_directory = "/Users/currocobo/Documents/Proyectos Intor/intor_prototyping-main/data/vectorstore"
collection_name = "new_collection"

# Usa el mismo modelo de embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Carga la base de datos vectorial
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

# Ver cuántos documentos hay en la colección
print(f"Número de documentos en la colección: {vectorstore._collection.count()}")

# Opcional: Inspecciona algunos documentos
docs = vectorstore.similarity_search("prueba", k=5)
for doc in docs:
    print(doc.page_content)
