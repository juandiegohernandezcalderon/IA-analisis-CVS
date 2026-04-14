#Sistema de Matching de CVs con Proyectos usando Llama 3 y Ollama
Este proyecto utiliza el modelo de lenguaje Llama 3 (a través de Ollama) para procesar hojas de vida (CVs) y descripciones de proyectos. El objetivo es:

Extraer información clave de los CVs (habilidades, experiencia, etc.).
Generar embeddings (representaciones numéricas) tanto para CVs como para proyectos.
Calcular la "sensibilidad" o "acople" entre CVs y proyectos utilizando la similitud del coseno de sus embeddings.
Estructura del Proyecto
app.py: Lógica principal del programa, orquestación del flujo.
config.py: Variables de configuración (URL de Ollama, nombres de modelos, rutas de datos).
utils.py: Funciones de utilidad (ej. cálculo de similitud del coseno).
requirements.txt: Dependencias de Python.
data/: Carpeta para almacenar los datos de entrada.
data/cvs/: Archivos de texto con el contenido de las hojas de vida.
data/projects/: Archivos de texto con las descripciones de los proyectos.
Requisitos Previos
Docker Desktop instalado y corriendo.
Contenedor de Ollama en ejecución (docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama).
Modelo Llama 3 descargado en Ollama (docker exec -it ollama ollama pull llama3).
Python 3.8+ instalado.
Configuración y Ejecución
Clona este repositorio (si lo estuvieras versionando con Git) o navega a la carpeta del proyecto.
Crea y activa un entorno virtual:
python -m venv venv
source venv/bin/activate (macOS/Linux) o .\venv\Scripts\activate (Windows)
Instala las dependencias:
pip install -r requirements.txt
Coloca tus archivos:
Pon tus hojas de vida (en formato .txt) en la carpeta data/cvs/.
Pon tus descripciones de proyectos (en formato .txt) en la carpeta data/projects/.
Ejecuta la aplicación:
python app.py
Uso
El programa procesará los CVs y proyectos, extrae información clave de los CVs, genera embeddings para ambos y calcula la similitud entre cada CV y cada proyecto, imprimiendo los resultados en la consola.
