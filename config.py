"""Configuración general de la app."""

import pandas as pd

COSTO2_ES_UNITARIO = True
ELIMINAR_COSTO_MAYOR_O_IGUAL_PRECIO = False

MIN_OBSERVACIONES = 3
MIN_PRECIOS_DISTINTOS = 2
P_VALUE_MAX_CONFIABLE = 0.10

ELASTICIDAD_CAP_MIN = -5
ELASTICIDAD_CAP_MAX = 0
LIMITE_NUMERICO_RF = 1e12

# Optimización de rendimiento: por defecto se usa clasificación por reglas.
# Cambia a True solo si necesitas entrenar el Random Forest conservador.
USE_RANDOM_FOREST_CLASSIFIER = False

# Límites visuales para que los dashboards no intenten dibujar demasiados puntos.
MAX_SKUS_CURVA_ELASTICIDAD = 8
MAX_ROWS_PREVIEW = 30

# Columnas que sí necesita la app para evitar leer cientos de columnas innecesarias.
# Esto acelera mucho CSV/Excel grandes sin sacrificar el modelo, porque conserva
# las variables usadas en limpieza, NSE, elasticidad, pricing, filtros y descargas.
COLUMNAS_LECTURA_VENTAS = [
    "tran_date", "qty", "net_sale", "prod_nbr", "SKU", "costo2",
    "precio_base", "ingreso_base", "margen_unitario", "margen_total",
    "store_nm", "dept_nm", "subdept_nm", "marca", "tipo_marca",
    "categoria_est_socio", "estado", "key", "id_municipio", "ubica_geo",
]

COLUMNAS_LECTURA_NSE = [
    "key", "ubica_geo", "id_municipio", "estado", "municipio",
    "est_socio", "categoria_est_socio",
]

COLUMNAS_LECTURA_PROMOCIONES = [
    "prod_nbr", "SKU", "sku", "tran_date", "fecha", "fecha_inicio",
    "start_date", "end_date", "fecha_fin", "promo_id", "id_promocion",
    "descuento", "discount", "mecanica", "2x1", "3x2",
]

# Si subes CSV, leer solo estas columnas reduce mucho el tiempo de lectura.
LEER_SOLO_COLUMNAS_NECESARIAS = True

# Formatos rápidos: Parquet es mucho más veloz que CSV/Excel para bases grandes.
PERMITIR_PARQUET = True


UMBRAL_CV_VAR_ALTA = 2.0
UMBRAL_REGISTROS_REMOVIDOS_AMARILLO = 0.25
UMBRAL_REGISTROS_REMOVIDOS_ROJO = 0.50
MIN_FILAS_ANALISIS = 50
MIN_SKUS_ANALISIS = 5

ESCENARIOS_CAMBIO_PRECIO = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]

ESCENARIOS_PROMOCION = [
    {
        "Escenario_ID": "promo_2x1",
        "Nombre_Escenario": "Promoción 2x1",
        "Nombre_Corto": "2x1",
        "Tipo_Escenario": "Promoción",
        "Cambio_Efectivo": -0.50,
        "Mecanica_Promocion": "2x1",
    },
    {
        "Escenario_ID": "promo_3x2",
        "Nombre_Escenario": "Promoción 3x2",
        "Nombre_Corto": "3x2",
        "Tipo_Escenario": "Promoción",
        "Cambio_Efectivo": -1 / 3,
        "Mecanica_Promocion": "3x2",
    },
    {
        "Escenario_ID": "promo_2do_50",
        "Nombre_Escenario": "Promoción 2do al 50%",
        "Nombre_Corto": "2do al 50%",
        "Tipo_Escenario": "Promoción",
        "Cambio_Efectivo": -0.25,
        "Mecanica_Promocion": "2do al 50%",
    },
]

ESCENARIOS_PRICING = pd.DataFrame(
    [
        {
            "Escenario_ID": f"precio_{int(round(cambio * 100)):+d}pct",
            "Nombre_Escenario": f"Cambio de precio {cambio * 100:+.0f}%",
            "Nombre_Corto": f"{cambio * 100:+.0f}%",
            "Tipo_Escenario": "Cambio de precio",
            "Cambio_Efectivo": cambio,
            "Mecanica_Promocion": "No aplica",
        }
        for cambio in ESCENARIOS_CAMBIO_PRECIO
    ]
    + ESCENARIOS_PROMOCION
)

COLUMNAS_MINIMAS_VENTAS = ["tran_date", "qty", "net_sale", "prod_nbr", "costo2"]

COLUMNAS_DESCARGA_ELASTICIDAD = [
    "SKU",
    "dept_nm",
    "subdept_nm",
    "marca",
    "tipo_marca",
    "categoria_est_socio",
    "trimestre",
    "beta",
    "elasticidad",
    "alfa",
    "r2",
    "p-value",
    "observaciones",
    "diagnóstico",
]

COLUMNAS_DESCARGA_EXPERIMENTOS = [
    "SKU",
    "dept_nm",
    "marca",
    "tipo_marca",
    "categoria_est_socio",
    "trimestre",
    "escenario aplicado",
    "unidades simuladas",
    "ingreso simulado",
    "margen simulado",
    "mejor escenario",
    "categoría de SKU",
]

COLUMNAS_DESCARGA_MEJOR = [
    "SKU",
    "trimestre",
    "categoría de SKU",
    "dept_nm",
    "marca",
    "tipo_marca",
    "categoria_est_socio",
    "elasticidad",
    "unidades simuladas",
    "ingreso simulado",
    "margen simulado",
    "mejor escenario",
]

STATE_COORDINATES = {
    "aguascalientes": (21.8853, -102.2916),
    "baja california": (30.8406, -115.2838),
    "baja california sur": (26.0444, -111.6661),
    "campeche": (19.8301, -90.5349),
    "chiapas": (16.7569, -93.1292),
    "chihuahua": (28.6320, -106.0691),
    "ciudad de mexico": (19.4326, -99.1332),
    "cdmx": (19.4326, -99.1332),
    "coahuila": (27.0587, -101.7068),
    "coahuila de zaragoza": (27.0587, -101.7068),
    "colima": (19.2452, -103.7241),
    "durango": (24.0277, -104.6532),
    "guanajuato": (21.0190, -101.2574),
    "guerrero": (17.4392, -99.5451),
    "hidalgo": (20.0911, -98.7624),
    "jalisco": (20.6597, -103.3496),
    "mexico": (19.4969, -99.7233),
    "estado de mexico": (19.4969, -99.7233),
    "michoacan": (19.5665, -101.7068),
    "michoacan de ocampo": (19.5665, -101.7068),
    "morelos": (18.6813, -99.1013),
    "nayarit": (21.7514, -104.8455),
    "nuevo leon": (25.5922, -99.9962),
    "oaxaca": (17.0732, -96.7266),
    "puebla": (19.0414, -98.2063),
    "queretaro": (20.5888, -100.3899),
    "quintana roo": (19.1817, -88.4791),
    "san luis potosi": (22.1565, -100.9855),
    "sinaloa": (25.1721, -107.4795),
    "sonora": (29.2972, -110.3309),
    "tabasco": (17.8409, -92.6189),
    "tamaulipas": (24.2669, -98.8363),
    "tlaxcala": (19.3182, -98.2375),
    "veracruz": (19.1738, -96.1342),
    "veracruz de ignacio de la llave": (19.1738, -96.1342),
    "yucatan": (20.7099, -89.0943),
    "zacatecas": (22.7709, -102.5832),
}
