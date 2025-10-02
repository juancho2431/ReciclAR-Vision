# Plan de organización y hoja de ruta detallada

Este documento consolida tanto la estructura de carpetas creada en la fase inicial del proyecto ReciclAR Vision (V1) como la hoja de ruta completa para llegar al MVP descrito en el documento de requerimientos.

## Objetivos de la fase inicial

1. Estandarizar la arquitectura del repositorio para que cada etapa del proyecto (datos, entrenamiento, API y demo) tenga un espacio claramente definido.
2. Crear archivos de configuración base que faciliten la instalación de dependencias y la aplicación de herramientas de calidad de código.
3. Documentar los pasos necesarios para completar el MVP, explicando el propósito de cada etapa.

## Estructura de directorios

La estructura adoptada se inspira en proyectos de ciencia de datos y MLOps para mantener la separación de responsabilidades:

```
ReciclAR-Vision/
├── api/                 # Artefactos de despliegue FastAPI (routers, esquemas)
├── app/                 # Código específico para la demo web (Streamlit/Gradio)
├── data/
│   ├── processed/       # Datos listos para entrenamiento/inferencia
│   └── raw/             # Datos originales sin modificar
├── docs/                # Documentación del proyecto y reportes
├── notebooks/           # Exploración y experimentación interactiva
├── scripts/             # Scripts CLI para tareas recurrentes
├── src/
│   ├── api/             # Componentes reutilizables de API (p. ej., validaciones)
│   ├── app/             # Componentes compartidos con la demo
│   ├── data/            # Pipelines de datos, transformaciones y datasets
│   └── models/          # Arquitecturas y lógica de entrenamiento/inferencia
├── tests/               # Pruebas unitarias e integración
├── README.md
├── pyproject.toml       # Configuración de dependencias y herramientas
└── .gitignore           # Exclusiones para control de versiones
```

### Justificación de la estructura

- **Separación de `api/` y `app/`**: permite versionar por separado los artefactos de despliegue (FastAPI) y la interfaz de usuario (Streamlit/Gradio), aun cuando compartan código base desde `src/`.
- **Carpetas `data/`, `notebooks/`, `scripts/`**: siguen el patrón de proyectos reproducibles. `data/` se subdivide para diferenciar datos crudos de procesados, mientras que `scripts/` aloja comandos reutilizables (p. ej., descarga de datos, entrenamiento automatizado).
- **`src/` modular**: centraliza la lógica de negocio en módulos que luego pueden consumirse tanto por la API como por la demo, evitando duplicación.
- **`tests/` dedicado**: desde el inicio se prevé la inclusión de pruebas automatizadas, alineado con el requisito de mantener calidad y evitar regresiones.

## Hoja de ruta completa

La tabla siguiente resume todas las etapas previstas para la V1, el objetivo de cada una y la razón por la que se consideran críticas.

| # | Etapa | Qué se debe hacer | Por qué es importante |
|---|-------|-------------------|-----------------------|
| 1 | Configuración de calidad | Instalar y configurar `pre-commit` con Black, Ruff y mypy; definir reglas básicas de linting. | Garantiza que cualquier colaborador ejecute validaciones automáticas antes de cada commit y mantiene el estilo consistente. |
| 2 | Documentación de datos | Registrar en `docs/datos.md` las fuentes (TrashNet, capturas propias), licencias, versiones y protocolos de anonimización. | Cumple requisitos éticos y legales, y deja claro cómo reproducir el dataset. |
| 3 | Ingesta y particionado | Implementar scripts para descargar/importar datos, normalizar etiquetas y generar splits estratificados (70/15/15). | Produce un conjunto de entrenamiento coherente con las métricas objetivo (macro-F1 ≥ 0.90). |
| 4 | Preprocesamiento y aumentación | Desarrollar `src/data/transforms.py` y datasets PyTorch con Albumentations (redimensionado, normalización, flips, rotaciones, brillo/contraste, desenfoque ligero, jitter de color). | Incrementa la robustez del modelo frente a variaciones reales de captura. |
| 5 | Entrenamiento base | Crear scripts/notebooks para fine-tuning de MobileNetV3 y EfficientNet-B0, incluyendo early stopping, scheduler y logging. | Permite comparar arquitecturas ligeras y seleccionar la que cumpla con macro-F1 ≥ 0.90 y latencia objetivo. |
| 6 | Evaluación exhaustiva | Generar reportes con accuracy, macro-F1, matriz de confusión y análisis de errores; documentar hallazgos en `docs/report.md`. | Ofrece trazabilidad del rendimiento y guía iteraciones de mejora. |
| 7 | Grad-CAM y explicabilidad | Implementar utilidades en `src/models/explainability.py` para producir mapas Grad-CAM reutilizables. | Brinda transparencia al usuario final y respalda la depuración del modelo. |
| 8 | Servicio de inferencia | Construir `api/main.py` con FastAPI, incluyendo validaciones, carga perezosa del modelo, logging estructurado y endpoint `POST /predict`. | Expone el modelo para consumo externo con la latencia establecida (≤ 200 ms). |
| 9 | Demo interactiva | Desarrollar la demo en `app/` (Streamlit/Gradio) que permita subir/fotografiar imágenes, mostrar predicción, confianza, Grad-CAM y recomendaciones de reciclaje configurables. | Facilita validación con usuarios y comunica los resultados del sistema. |
| 10 | Empaquetado y despliegue | Crear `Dockerfile`, `docker-compose.yml` y scripts de arranque para API/demo; definir instrucciones de despliegue en `docs/deployment.md`. | Asegura que el MVP pueda ejecutarse de forma reproducible en distintos entornos. |
| 11 | Control de calidad y automatización | Configurar CI (GitHub Actions u otro) para linting, pruebas y builds; añadir pruebas unitarias y de integración mínimas en `tests/`. | Reduce regresiones y sostiene la calidad a medida que evoluciona el proyecto. |

## Dependencias iniciales

El archivo `pyproject.toml` centraliza la gestión de dependencias y herramientas de calidad:

- **Dependencias principales**: FastAPI, Streamlit/Gradio, PyTorch y Albumentations cubren el pipeline de inferencia y el entrenamiento planificado.
- **Dependencias opcionales (dev)**: pytest, black, ruff, mypy y pre-commit permiten establecer un workflow de calidad desde etapas tempranas.
- **Configuraciones**: se fijan lineamientos de formato (Black, Ruff) y pruebas (pytest) para favorecer consistencia entre colaboradores.

## Secuencia sugerida de ejecución

1. **Preparar entorno**: crear y activar un entorno virtual, instalar dependencias mediante `pip install -e .[dev]` y ejecutar `pre-commit install`.
2. **Reunir datos**: seguir la documentación de `docs/datos.md` (una vez creada) para descargar/limpiar TrashNet y capturas locales.
3. **Ejecutar scripts de ingesta**: usar los comandos en `scripts/` para generar los splits y verificar estadísticas de balanceo.
4. **Desarrollar pipeline de datos**: implementar y probar transformaciones y datasets reutilizables desde `src/data/`.
5. **Entrenar modelos**: lanzar los experimentos desde `notebooks/` o scripts CLI, registrando métricas y guardando checkpoints en una carpeta dedicada.
6. **Analizar resultados**: documentar métricas y errores en `docs/report.md`, iterando ajustes de hiperparámetros si es necesario.
7. **Integrar explicabilidad**: validar que los mapas Grad-CAM sean coherentes y almacenarlos para la demo.
8. **Construir API y demo**: conectar el pipeline de inferencia con FastAPI y la interfaz Streamlit/Gradio, incluyendo manejo de errores y mensajes al usuario.
9. **Empaquetar**: generar imágenes Docker y validar que API y demo se levanten correctamente mediante `docker-compose`.
10. **Automatizar calidad**: agregar CI, pruebas y monitoreo básico antes de declarar la V1 lista.

Este plan se actualizará conforme se complete cada etapa, manteniendo un historial claro de decisiones y su justificación.
