# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de dependencias e instala las librerías
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el resto del código del frontend
COPY . .

# Expone el puerto por defecto de Streamlit (8501)
EXPOSE 8501

# Comando para iniciar la aplicación de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
