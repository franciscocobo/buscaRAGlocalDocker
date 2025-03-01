# Imagen base compatible con Mac Silicon
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    cmake \
    libmupdf-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt



# Copiar el código de la aplicación
COPY . .

# Exponer el puerto donde correrá Flask
EXPOSE 5001

# Ejecutar la aplicación
CMD ["python", "app.py"]
