# App de Pricing Dinámico, Elasticidad y Proyección de Ventas

Aplicación web en **Streamlit** para analizar ventas de retail, limpiar datos, cruzar información socioeconómica NSE, calcular elasticidad por SKU y trimestre, simular escenarios de pricing y descargar recomendaciones.

## Objetivo

La app permite:

1. Cargar una base de ventas.
2. Cargar opcionalmente promociones.
3. Usar una base NSE predeterminada o cargar una modificada.
4. Limpiar datos.
5. Cruzar ventas con NSE.
6. Diagnosticar calidad con semáforo.
7. Calcular elasticidad log-log por SKU y bloques fijos de 3 meses.
8. Simular escenarios de pricing.
9. Recomendar el mejor escenario por SKU.
10. Descargar resultados en CSV.

## Estructura recomendada

```text
pricing-dinamico-retail/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── convertir_a_parquet.py
│
├── modules/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── quality.py
│   ├── elasticity.py
│   └── pricing.py
│
├── data/
└── assets/
```

Las carpetas `data/` y `assets/` son opcionales. La app no depende de archivos locales obligatorios: las bases se cargan desde la interfaz.

## Columnas mínimas esperadas en ventas

Para bases grandes, se recomienda subir la base en **CSV** o, mejor aún, **Parquet**. Parquet suele cargar mucho más rápido que Excel o CSV.

La base de ventas debe incluir como mínimo:

- `tran_date`
- `qty`
- `net_sale`
- `prod_nbr`
- `costo2`

Columnas recomendadas para enriquecer análisis:

- `dept_nm`
- `subdept_nm`
- `marca`
- `tipo_marca`
- `store_nm`
- `estado`
- `key`
- `categoria_est_socio`

## NSE

La app incluye una base NSE predeterminada generada en código. El usuario puede:

1. Descargarla.
2. Modificarla.
3. Subirla de nuevo.
4. Presionar **Aplicar cambios de NSE**.

El modelo final usa únicamente `categoria_est_socio`; no usa `est_socio_nbr` como variable final del Random Forest.

## Escenarios de pricing incluidos

- `+15%`
- `+10%`
- `+5%`
- `+0%`
- `-5%`
- `-10%`
- `-15%`
- `2x1`
- `3x2`
- `2do al 50%`

## Librerías usadas

- `streamlit`: interfaz web.
- `pandas` y `numpy`: manipulación de datos.
- `plotly`: visualizaciones interactivas.
- `statsmodels`: regresión OLS log-log para elasticidad.
- `scikit-learn`: Random Forest conservador para clasificación.
- `openpyxl`: lectura de archivos Excel.
- `pyarrow`: lectura/escritura de Parquet para acelerar bases grandes.

## Cómo correr localmente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

En Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Cómo subirlo a GitHub

1. Crea un repositorio público en GitHub.
2. Crea cada archivo con la opción **Add file > Create new file**.
3. Copia y pega el contenido de cada archivo.
4. Para los módulos, crea archivos dentro de `modules/`.
5. Haz commit de los cambios.
6. En tu computadora, clona el repositorio o descarga el ZIP.
7. Ejecuta `pip install -r requirements.txt`.
8. Ejecuta `streamlit run app.py`.

## Notas importantes

La app replica la lógica estadística central del notebook base:

- Limpieza crítica de ventas.
- Limpieza de `store_nm`.
- Creación de precio unitario, ingreso y margen.
- Semáforo de calidad.
- Cruce NSE flexible.
- Elasticidad log-log por SKU y trimestre.
- Fallback de elasticidad por SKU global, subdepartamento, departamento y total trimestre.
- Simulación con fórmula exponencial de elasticidad.
- Selección de mejor escenario por categoría.

Cuando una columna no existe, la app muestra errores o advertencias claras en lugar de romperse.


## Optimización de rendimiento

Esta versión evita leer la base de ventas en cada rerun de Streamlit. La base solo se lee cuando presionas **Procesar / actualizar datos**. Además, por defecto solo se cargan las columnas necesarias para limpieza, NSE, elasticidad, pricing, filtros y descargas. Esto reduce el tiempo de lectura sin sacrificar el modelo ni el análisis.

Si tu archivo de ventas es muy grande, conviértelo una vez a Parquet y súbelo en ese formato:

```bash
python convertir_a_parquet.py ventas.csv ventas.parquet
```

Si necesitas leer todas las columnas del archivo por algún motivo, cambia en `modules/config.py`:

```python
LEER_SOLO_COLUMNAS_NECESARIAS = False
```
