# config.py

# URL base para la API de Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Nombre del modelo Llama 3 que has descargado en Ollama
# Asegúrate de que este nombre coincida con el que usaste en `ollama pull`
# Por ejemplo, si usaste `ollama pull llama3`, el nombre es "llama3"
# Si usaste `ollama pull llama3.1:8b`, el nombre es "llama3.1:8b"
OLLAMA_MODEL_NAME = "llama3" # O "llama3.1:8b" si descargaste esa versión

# Ruta a las carpetas donde están tus hojas de vida y descripciones de proyectos
CVS_DIR = "data/cvs"
PROJECTS_DIR = "data/projects"