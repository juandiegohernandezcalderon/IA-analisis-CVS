# app.py
import requests
import json
import os # Para listar archivos en directorios

# Importar configuraciones y utilidades
from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, CVS_DIR, PROJECTS_DIR
from utils import cosine_similarity

# --- Funciones para interactuar con Ollama ---

def get_ollama_response(endpoint, data, stream=False):
    """
    Función genérica para hacer peticiones POST a la API de Ollama.
    """
    url = f"{OLLAMA_BASE_URL}/api/{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=stream)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Ollama en {url}: {e}")
        return None

def generate_text_with_ollama(prompt: str) -> str | None:
    """
    Genera texto utilizando el modelo Llama 3 en Ollama.
    """
    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False # Obtener la respuesta completa de una vez
    }
    response = get_ollama_response("generate", data)
    if response:
        return response.json().get("response")
    return None

def get_embeddings_from_ollama(text: str) -> list[float] | None:
    """
    Obtiene los embeddings de un texto utilizando el modelo Llama 3 en Ollama.
    """
    data = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": text
    }
    response = get_ollama_response("embeddings", data)
    if response:
        return response.json().get("embedding")
    return None

# --- Funciones para leer archivos ---

def read_text_file(filepath: str) -> str | None:
    """Lee el contenido de un archivo de texto."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {filepath}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo {filepath}: {e}")
        return None

# --- Lógica principal del procesamiento ---

import re # Necesitarás importar 're' para usar expresiones regulares

def process_cv(cv_filepath: str):
    """Procesa una hoja de vida: extrae info y genera embedding."""
    print(f"\nProcesando CV: {cv_filepath}")
    cv_content = read_text_file(cv_filepath)
    if not cv_content:
        return None, None

    # 1. Extraer información estructurada (usando un prompt avanzado)
    extraction_prompt = f"""Extrae las siguientes entidades de la hoja de vida proporcionada en formato JSON.
    Las claves del JSON deben ser: "nombre", "email", "telefono", "habilidades" (lista de strings),
    "experiencia" (lista de objetos, cada uno con "puesto", "empresa", "duracion", "descripcion_breve"),
    "educacion" (lista de objetos, cada uno con "titulo", "institucion", "año_inicio", "año_fin").
    Si una entidad no se encuentra, su valor debe ser nulo o una lista vacía según corresponda.
    Tu respuesta DEBE ser SOLAMENTE el bloque JSON, sin texto explicativo adicional antes o después.
    ASEGÚRATE DE QUE EL JSON GENERADO ES VÁLIDO Y SIN ERRORES DE SINTAXIS (COMA EXTRA, LLAVE FALTANTE, ETC.).

    Hoja de Vida:
    {cv_content}

    JSON:
    """
    extracted_json_str = generate_text_with_ollama(extraction_prompt)
    extracted_info = None

    if extracted_json_str:
        try:
            # --- Lógica de limpieza de JSON más robusta ---
            # Patrón para encontrar el bloque JSON delimitado por ```json ... ```
            # re.DOTALL permite que '.' coincida con saltos de línea
            # re.IGNORECASE para ser robusto a ```JSON
            json_block_match = re.search(r"```json\s*(.*?)\s*```", extracted_json_str, re.DOTALL | re.IGNORECASE)

            json_only_str = None
            if json_block_match:
                # Si se encuentra el bloque ```json```, usa su contenido
                json_only_str = json_block_match.group(1).strip()
            else:
                # Si no se encuentra ```json```, intenta el enfoque anterior de buscar { y }
                # Esto es un fallback por si el modelo cambia su formato inesperadamente
                start_brace = extracted_json_str.find('{')
                end_brace = extracted_json_str.rfind('}')

                if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                    json_only_str = extracted_json_str[start_brace : end_brace + 1].strip()

            if not json_only_str:
                raise ValueError("No se pudo extraer un bloque JSON válido de la respuesta del modelo.")

            extracted_info = json.loads(json_only_str)
            print("Información extraída (parcial):", {k: extracted_info[k] for k in extracted_info if k in ["nombre", "habilidades"]})

        except json.JSONDecodeError as e:
            print(f"Error decodificando JSON de CV '{os.path.basename(cv_filepath)}': {e}")
            print(f"Respuesta cruda de Ollama (fragmento):\n{extracted_json_str[:500]}...") # Mostrar un fragmento
            extracted_info = None
        except ValueError as e: # Captura el error de ValueError que generamos si no se encuentra JSON
            print(f"Error procesando JSON de CV '{os.path.basename(cv_filepath)}': {e}")
            print(f"Respuesta cruda de Ollama (fragmento):\n{extracted_json_str[:500]}...")
            extracted_info = None
        except Exception as e: # Captura cualquier otro error inesperado
            print(f"Un error inesperado ocurrió al procesar el JSON de CV '{os.path.basename(cv_filepath)}': {e}")
            print(f"Respuesta cruda de Ollama (fragmento):\n{extracted_json_str[:500]}...")
            extracted_info = None


    # 2. Generar embedding del CV para la similitud
    cv_embedding = get_embeddings_from_ollama(cv_content)
    if cv_embedding:
        print(f"Embedding generado para CV: {len(cv_embedding)} dimensiones")
    else:
        print("No se pudo generar embedding para el CV.")

    return extracted_info, cv_embedding

def process_project(project_filepath: str):
    """Procesa un proyecto: extrae contenido y genera embedding."""
    print(f"\nProcesando Proyecto: {project_filepath}")
    project_content = read_text_file(project_filepath)
    if not project_content:
        return None, None

    # Para proyectos, no necesitamos extracción estructurada compleja por ahora, solo el embedding.
    # Pero podrías añadir un prompt si quisieras extraer requisitos clave del proyecto.
    
    project_embedding = get_embeddings_from_ollama(project_content)
    if project_embedding:
        print(f"Embedding generado para Proyecto: {len(project_embedding)} dimensiones")
    else:
        print("No se pudo generar embedding para el proyecto.")

    return project_content, project_embedding


def main():
    """Función principal para ejecutar el procesamiento."""
    print("Iniciando el proceso de matching de CVs y Proyectos...")

    # --- Cargar y Procesar Hojas de Vida ---
    cv_data = [] # Para almacenar info extraída y embeddings de CVs
    cv_files = [f for f in os.listdir(CVS_DIR) if f.endswith('.txt')] # Filtrar solo .txt
    
    if not cv_files:
        print(f"Advertencia: No se encontraron archivos .txt en la carpeta '{CVS_DIR}'.")

    for filename in cv_files:
        filepath = os.path.join(CVS_DIR, filename)
        extracted_info, embedding = process_cv(filepath)
        if extracted_info and embedding:
            cv_data.append({
                "filename": filename,
                "info": extracted_info,
                "embedding": embedding
            })

    # --- Cargar y Procesar Proyectos ---
    project_data = [] # Para almacenar contenido y embeddings de proyectos
    project_files = [f for f in os.listdir(PROJECTS_DIR) if f.endswith('.txt')] # Filtrar solo .txt
    
    if not project_files:
        print(f"Advertencia: No se encontraron archivos .txt en la carpeta '{PROJECTS_DIR}'.")

    for filename in project_files:
        filepath = os.path.join(PROJECTS_DIR, filename)
        content, embedding = process_project(filepath)
        if content and embedding:
            project_data.append({
                "filename": filename,
                "content": content,
                "embedding": embedding
            })

    # --- Realizar el Matching (Calcular Similitud) ---
    print("\nCalculando similitud entre CVs y Proyectos...")
    if not cv_data or not project_data:
        print("No hay suficientes CVs o Proyectos para realizar el matching.")
        return

    for cv in cv_data:
        cv_name = cv["info"].get("nombre", os.path.basename(cv["filename"]))
        print(f"\n--- CV: {cv_name} ---")
        matches = []
        for project in project_data:
            sim = cosine_similarity(cv["embedding"], project["embedding"])
            matches.append({"project": project["filename"], "similarity": sim})
        
        # Ordenar los proyectos por similitud (de mayor a menor)
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        for match in matches:
            print(f"  Acople con '{match['project']}': {match['similarity']:.4f}")
            # Puedes establecer un umbral aquí para considerar un buen acople
            if match['similarity'] > 0.7: # Ejemplo de umbral
                print("    (¡Buen acople!)")

if __name__ == "__main__":
    main()