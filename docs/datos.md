# Gestión de datos

## Fuentes sugeridas
- **TrashNet**: https://github.com/garythung/trashnet (licencia MIT). Contiene imágenes clasificadas en papel, cartón, vidrio, metal, plástico y basura general.
- **Capturas propias**: fotografías en contexto local para mejorar la representatividad. Se recomienda documentar fecha, cámara utilizada y condiciones de iluminación.

## Lineamientos de recolección
1. Asegurar que cada imagen contenga un único objeto claramente visible.
2. Incluir variaciones de fondo, iluminación y perspectiva.
3. Evitar información personal identificable o fondos con rostros.
4. Guardar las imágenes en `data/raw/<clase>/<nombre>.jpg` con clases normalizadas (`paper`, `cardboard`, `plastic`, `glass`, `metal`, `other`).

## Preparación y particionado
1. Descargar y descomprimir TrashNet dentro de `data/raw` respetando la estructura de carpetas.
2. Añadir capturas propias siguiendo la convención de nombres.
3. Ejecutar el script de particionado estratificado:
   ```bash
   python scripts/create_split.py data/raw data/splits/v1.json
   ```
4. Copiar o enlazar simbólicamente las rutas listadas en el JSON a:
   - `data/processed/train/<clase>`
   - `data/processed/val/<clase>`
   - `data/processed/test/<clase>`

## Control de calidad
- Verificar balance aproximado de clases (≥ 400 imágenes por clase recomendado).
- Revisar imágenes con errores (objetos múltiples, blur severo) y depurarlas.
- Mantener un registro de versiones del dataset (por fecha y cambios principales).

## Consideraciones éticas
- Solicitar consentimiento cuando se utilicen imágenes capturadas en espacios privados.
- Almacenar metadatos sensibles (si existieran) en repositorios privados cifrados.
- Documentar posibles sesgos (materiales poco frecuentes, colores dominantes) y abordarlos en el análisis de errores.
