# ReciclAR Vision

ReciclAR Vision es un sistema de clasificación de residuos basado en visión por computador. El objetivo de la versión inicial (V1) es reconocer, a partir de una fotografía de un único objeto, la categoría de reciclaje entre seis clases: papel, cartón, plástico, vidrio, metal y otros/no reciclable.

## Estado del proyecto

El repositorio incluye:

- Pipelines de datos con Albumentations y PyTorch para entrenamiento e inferencia.
- Scripts reproducibles para generar splits estratificados (`scripts/create_split.py`) y entrenar modelos ligeros (`scripts/train.py`).
- Utilidades de Grad-CAM para explicar predicciones y generar mapas de calor.
- Servicio REST en FastAPI (`api/main.py`) y demo Streamlit (`app/main.py`) que consumen el mismo predictor.
- Pruebas automatizadas que validan el pipeline y el endpoint `/health` del servicio.

## Estructura inicial

```
ReciclAR-Vision/
├── api/                 # Código específico para el servicio FastAPI
├── app/                 # Código específico para la demo interactiva
├── data/                # Datos crudos y procesados (excluidos del repo)
├── docs/                # Documentación del proyecto
├── notebooks/           # Notebooks de exploración y experimentación
├── scripts/             # Scripts auxiliares (descarga, entrenamiento, etc.)
├── src/                 # Código fuente reutilizable (datasets, modelos, utils)
├── tests/               # Pruebas automatizadas
├── pyproject.toml       # Dependencias y herramientas de desarrollo
└── .gitignore           # Exclusiones para control de versiones
```

## Plan de trabajo resumido

1. **Preparar entorno**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   pre-commit install
   ```
2. **Crear el split de datos** (estructura por carpetas `data/raw/<clase>/<imagen>`):
   ```bash
   python scripts/create_split.py data/raw split.json
   ```
   Copia las rutas generadas a `data/processed/train|val|test` o usa `--split-json` al entrenar.
3. **Entrenar el modelo**
   ```bash
   python scripts/train.py data/processed --epochs 20 --backbone efficientnet_b0
   ```
   El checkpoint quedará en `artifacts/best_model.pt` junto con la configuración utilizada.
4. **Servir la API**
   ```bash
   export RECICLAR_MODEL_PATH=artifacts/best_model.pt
   uvicorn api.main:app --reload
   ```
   Visita `http://localhost:8000/docs` para probar el endpoint `POST /predict`.
5. **Ejecutar la demo Streamlit**
   ```bash
   export RECICLAR_MODEL_PATH=artifacts/best_model.pt
   streamlit run app/main.py
   ```
   La interfaz permite subir una imagen, ver la predicción y explorar Grad-CAM.
6. **Probar**
   ```bash
   pytest
   ```
   Garantiza que los transformaciones, splits y endpoint básico se comporten como se espera.

Para el detalle completo de cada etapa y su justificación consulta `docs/plan.md`.
