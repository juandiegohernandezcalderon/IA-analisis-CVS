# Sistema de Matching de CVs con Proyectos (Llama 3 & Ollama)

Este proyecto utiliza el modelo de lenguaje **Llama 3** (ejecutado localmente a través de **Ollama**) para automatizar el análisis de hojas de vida (CVs) y su emparejamiento con descripciones de proyectos.



## 🎯 Objetivos del Proyecto

* **Extracción Inteligente:** Identificar automáticamente habilidades y experiencia clave de los CVs.
* **Embeddings de Texto:** Generar representaciones vectoriales numéricas tanto para CVs como para proyectos.
* **Cálculo de Afinidad:** Determinar la "sensibilidad" o nivel de acople mediante la **similitud del coseno** entre los embeddings generados.

## 📁 Estructura del Proyecto

```text
├── app.py                # Lógica principal y orquestación del flujo.
├── config.py             # Configuración (URLs de Ollama, modelos, rutas).
├── utils.py              # Funciones auxiliares (cálculo de similitud).
├── requirements.txt      # Dependencias de Python.
└── data/                 # Almacenamiento de datos de entrada.
    ├── cvs/              # Archivos .txt con las hojas de vida.
    └── projects/         # Archivos .txt con descripciones de proyectos.
