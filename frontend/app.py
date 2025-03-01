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