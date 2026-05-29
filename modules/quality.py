"""Diagnóstico de calidad de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    MIN_FILAS_ANALISIS,
    MIN_OBSERVACIONES,
    MIN_PRECIOS_DISTINTOS,
    MIN_SKUS_ANALISIS,
    UMBRAL_CV_VAR_ALTA,
    UMBRAL_REGISTROS_REMOVIDOS_AMARILLO,
    UMBRAL_REGISTROS_REMOVIDOS_ROJO,
)


def _coeficiente_variacion(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    media = s.mean()
    if len(s) == 0 or pd.isna(media) or media == 0:
        return np.nan
    return float(s.std() / abs(media))


def calculate_quality_diagnosis(
    ventas_limpias: pd.DataFrame,
    resumen_limpieza: pd.DataFrame,
    summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula semáforo de calidad siguiendo la lógica del notebook base."""
    ventas = ventas_limpias.copy()

    variables_calidad = {
        "qty": "cantidad de unidades",
        "net_sale": "venta neta",
        "precio_unitario": "precio unitario",
        "costo_unitario": "costo unitario",
        "margen_unitario": "margen unitario",
        "margen_total": "margen total",
    }

    metricas_varianza = []
    for col, nombre in variables_calidad.items():
        cv = _coeficiente_variacion(ventas[col]) if col in ventas.columns else np.nan
        metricas_varianza.append(
            {
                "Variable": nombre,
                "Columna": col,
                "Coeficiente_Variacion": cv,
                "Varianza_Alta": bool(pd.notna(cv) and cv >= UMBRAL_CV_VAR_ALTA),
            }
        )

    calidad_varianza = pd.DataFrame(metricas_varianza)

    variables_varianza_alta = calidad_varianza.loc[
        calidad_varianza["Varianza_Alta"], "Variable"
    ].tolist()

    filas_originales = int(summary.get("filas_originales", 0))
    registros_removidos = int(summary.get("registros_removidos", filas_originales - len(ventas)))
    porcentaje_removido = float(summary.get("porcentaje_removido", 1 if filas_originales == 0 else registros_removidos / filas_originales))

    skus_unicos = ventas["prod_nbr"].nunique() if "prod_nbr" in ventas.columns else 0
    skus_con_obs_suficientes = int((ventas.groupby("prod_nbr").size() >= MIN_OBSERVACIONES).sum()) if "prod_nbr" in ventas.columns else 0
    skus_con_precios_suficientes = (
        int((ventas.groupby("prod_nbr")["precio_unitario"].nunique() >= MIN_PRECIOS_DISTINTOS).sum())
        if {"prod_nbr", "precio_unitario"}.issubset(ventas.columns)
        else 0
    )

    motivos = []

    if len(ventas) < MIN_FILAS_ANALISIS:
        motivos.append("datos insuficientes por pocas filas limpias")

    if skus_unicos < MIN_SKUS_ANALISIS:
        motivos.append("datos insuficientes por pocos SKUs")

    if skus_con_obs_suficientes == 0 or skus_con_precios_suficientes == 0:
        motivos.append("ningún SKU cumple mínimos de observaciones o variación de precio")

    if porcentaje_removido >= UMBRAL_REGISTROS_REMOVIDOS_ROJO:
        motivos.append("se removió más del 50% de los registros")

    if len(motivos) > 0:
        semaforo = "🔴 Rojo"
        interpretacion = "Hay datos insuficientes para confiar en el análisis."
    elif len(variables_varianza_alta) > 0 or porcentaje_removido >= UMBRAL_REGISTROS_REMOVIDOS_AMARILLO:
        semaforo = "🟡 Amarillo"
        interpretacion = "Los datos son parciales; se puede continuar con advertencias."
        if variables_varianza_alta:
            motivos.append("varianza alta en " + ", ".join(variables_varianza_alta))
        if porcentaje_removido >= UMBRAL_REGISTROS_REMOVIDOS_AMARILLO:
            motivos.append("se removió una proporción relevante de registros en limpieza")
    else:
        semaforo = "🟢 Verde"
        interpretacion = "Los datos están listos para el análisis."
        motivos.append("sin alertas críticas")

    semaforo_datos = pd.DataFrame(
        [
            {
                "Semaforo": semaforo,
                "Interpretacion": interpretacion,
                "Filas_Originales": filas_originales,
                "Filas_Limpias": int(len(ventas)),
                "Registros_Removidos": registros_removidos,
                "%_Registros_Removidos": porcentaje_removido * 100,
                "Porcentaje_Datos_Faltantes_Original": summary.get("faltantes_pct_original", np.nan),
                "Duplicados_Originales": summary.get("duplicados_originales", np.nan),
                "Valores_Infinitos_Detectados": summary.get("infinitos_detectados_original", np.nan),
                "Registros_Precio_Invalido": summary.get("registros_precio_invalido", np.nan),
                "Registros_Cantidad_Invalida": summary.get("registros_cantidad_invalida", np.nan),
                "Registros_Costo_Mayor_O_Igual_Precio": summary.get("registros_costo_mayor_o_igual_precio", np.nan),
                "SKUs_Unicos": skus_unicos,
                "SKUs_Con_Obs_Suficientes": skus_con_obs_suficientes,
                "SKUs_Con_Precios_Suficientes": skus_con_precios_suficientes,
                "Motivos": "; ".join(motivos),
            }
        ]
    )

    return semaforo_datos, calidad_varianza
