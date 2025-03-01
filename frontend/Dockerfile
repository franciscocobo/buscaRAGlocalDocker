# Imagen base compatible con Mac Silicon
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto donde correrá Flask
EXPOSE 5001

# Ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=5001"]

