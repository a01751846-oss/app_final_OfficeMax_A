"""Cálculo de elasticidad trimestral."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import MIN_OBSERVACIONES, MIN_PRECIOS_DISTINTOS
from .utils import build_quarter_label


def integrate_promotions(ventas_base: pd.DataFrame, promociones: pd.DataFrame | None) -> pd.DataFrame:
    """Integra promociones a nivel SKU-mes de forma opcional."""
    ventas = ventas_base.copy()
    ventas["num_promociones"] = 0
    ventas["tiene_promocion"] = 0

    if promociones is None or promociones.empty:
        return ventas

    promos = promociones.copy()
    promos.columns = promos.columns.astype(str).str.strip()

    posibles_cols_sku = ["prod_nbr", "SKU", "sku", "producto", "product_id"]
    col_sku = next((col for col in posibles_cols_sku if col in promos.columns), None)

    posibles_cols_fecha_inicio = [
        "tran_date",
        "fecha",
        "fecha_inicio",
        "inicio_promocion",
        "fecha_ini",
        "start_date",
        "Start_Date",
    ]
    posibles_cols_fecha_fin = [
        "fecha_fin",
        "fin_promocion",
        "fecha_final",
        "end_date",
        "End_Date",
        "fecha_termino",
        "fecha_término",
    ]

    col_inicio = next((col for col in posibles_cols_fecha_inicio if col in promos.columns), None)
    col_fin = next((col for col in posibles_cols_fecha_fin if col in promos.columns), None)

    if col_sku is None or col_inicio is None:
        return ventas

    promos[col_sku] = promos[col_sku].astype(str)
    promos[col_inicio] = pd.to_datetime(promos[col_inicio], errors="coerce")

    if col_fin is not None:
        promos[col_fin] = pd.to_datetime(promos[col_fin], errors="coerce")
    else:
        col_fin = col_inicio

    promos[col_fin] = promos[col_fin].fillna(promos[col_inicio])

    registros = []
    for _, promo in promos.dropna(subset=[col_sku, col_inicio]).iterrows():
        fecha_inicio = promo[col_inicio]
        fecha_fin = promo[col_fin] if pd.notna(promo[col_fin]) else fecha_inicio
        if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
            continue
        meses_promo = pd.period_range(fecha_inicio.to_period("M"), fecha_fin.to_period("M"), freq="M")
        for mes_promo in meses_promo:
            registros.append({"prod_nbr": str(promo[col_sku]), "mes": mes_promo})

    promociones_mes = pd.DataFrame(registros)
    if promociones_mes.empty:
        return ventas

    promociones_mes = (
        promociones_mes.groupby(["prod_nbr", "mes"], as_index=False)
        .size()
        .rename(columns={"size": "num_promociones"})
    )
    promociones_mes["tiene_promocion"] = 1

    ventas = ventas.merge(
        promociones_mes,
        on=["prod_nbr", "mes"],
        how="left",
        suffixes=("", "_promo"),
    )

    ventas["num_promociones"] = ventas["num_promociones_promo"].fillna(0)
    ventas["tiene_promocion"] = ventas["tiene_promocion_promo"].fillna(0)
    ventas = ventas.drop(
        columns=[c for c in ["num_promociones_promo", "tiene_promocion_promo"] if c in ventas.columns],
        errors="ignore",
    )

    return ventas


def build_three_month_blocks(ventas_base: pd.DataFrame) -> list[dict]:
    """Crea bloques fijos de 3 meses, igual que el notebook base."""
    meses_ordenados = sorted(ventas_base["mes"].dropna().unique())
    bloques = []
    for i in range(0, len(meses_ordenados), 3):
        meses_bloque = meses_ordenados[i : i + 3]
        if len(meses_bloque) < 3:
            continue
        periodo_3m = f"{meses_bloque[0]} a {meses_bloque[-1]}"
        bloques.append(
            {
                "bloque_id": len(bloques) + 1,
                "mes_inicio": meses_bloque[0],
                "mes_fin": meses_bloque[-1],
                "meses": meses_bloque,
                "periodo_3m": periodo_3m,
                "trimestre": build_quarter_label(periodo_3m),
            }
        )
    return bloques


def diagnosticar_elasticidad(beta) -> str:
    """Diagnóstico interpretativo de elasticidad."""
    if pd.isna(beta):
        return "Datos insuficientes"
    if beta >= 0:
        return "Relación positiva / revisar datos"
    if beta > -1:
        return "Inelástica"
    if beta == -1:
        return "Elasticidad unitaria"
    return "Elástica"


def preparar_df_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ventas por día y precio para estimar demanda diaria por nivel de precio.
    """
    cols_req = ["fecha_dia", "precio_modelo", "qty", "net_sale"]
    df_m = df.dropna(subset=cols_req).copy()
    df_m = df_m[(df_m["qty"] > 0) & (df_m["net_sale"] > 0) & (df_m["precio_modelo"] > 0)].copy()

    if df_m.empty:
        return pd.DataFrame(columns=["qty_modelo", "precio_modelo"])

    df_agg = (
        df_m.groupby(["fecha_dia", "precio_modelo"], as_index=False)
        .agg(qty_modelo=("qty", "sum"), venta_modelo=("net_sale", "sum"))
    )

    df_agg["precio_modelo"] = df_agg["venta_modelo"] / df_agg["qty_modelo"]
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_agg = df_agg.dropna(subset=["qty_modelo", "precio_modelo"])
    df_agg = df_agg[(df_agg["qty_modelo"] > 0) & (df_agg["precio_modelo"] > 0)].copy()

    return df_agg


def estimar_elasticidad_loglog(
    df: pd.DataFrame,
    fuente: str = "SKU-trimestre",
    min_observaciones: int = MIN_OBSERVACIONES,
    min_precios_distintos: int = MIN_PRECIOS_DISTINTOS,
) -> dict:
    """
    Estima elasticidad con OLS log-log:
    log(qty_modelo) = alfa + beta * log(precio_modelo).
    """
    df_modelo = preparar_df_modelo(df)

    n_modelo = len(df_modelo)
    precios_distintos = df_modelo["precio_modelo"].nunique() if not df_modelo.empty else 0
    qty_distintas = df_modelo["qty_modelo"].nunique() if not df_modelo.empty else 0

    if n_modelo < min_observaciones:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": f"Menos de {min_observaciones} observaciones agregadas",
        }

    if precios_distintos < min_precios_distintos:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Sin variación suficiente de precio",
        }

    if qty_distintas < 2:
        return {
            "Beta": 0.0,
            "Elasticidad": 0.0,
            "Alfa": float(np.log(df_modelo["qty_modelo"].iloc[0])),
            "R2": 0.0,
            "P_Value": 1.0,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Cantidad agregada constante; elasticidad aproximada a 0",
        }

    df_modelo["log_qty"] = np.log(df_modelo["qty_modelo"])
    df_modelo["log_precio"] = np.log(df_modelo["precio_modelo"])
    df_modelo = df_modelo.replace([np.inf, -np.inf], np.nan)
    df_modelo = df_modelo.dropna(subset=["log_qty", "log_precio"])

    if len(df_modelo) < min_observaciones:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique() if not df_modelo.empty else 0,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Observaciones insuficientes después de logs",
        }

    X = sm.add_constant(df_modelo["log_precio"], has_constant="add")
    y = df_modelo["log_qty"]

    try:
        modelo = sm.OLS(y, X).fit()
        beta = modelo.params.get("log_precio", np.nan)
        alfa = modelo.params.get("const", np.nan)
        r2 = modelo.rsquared if np.isfinite(modelo.rsquared) else np.nan
        p_value = modelo.pvalues.get("log_precio", np.nan)

        if pd.isna(p_value) and len(df_modelo) <= 2:
            motivo = "Modelo estimado, pero p-value no disponible por pocos grados de libertad"
        else:
            motivo = "Modelo estimado correctamente"

        return {
            "Beta": beta,
            "Elasticidad": beta,
            "Alfa": alfa,
            "R2": r2,
            "P_Value": p_value,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique(),
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": motivo,
        }
    except Exception as exc:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique(),
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": f"Error statsmodels: {str(exc)[:120]}",
        }


def _moda_segura(df: pd.DataFrame, col: str):
    if col in df.columns:
        m = df[col].dropna().mode()
        if not m.empty:
            return m.iloc[0]
    return np.nan


def mejor_estimacion_con_fallback(
    df_sku_trimestre: pd.DataFrame,
    sku: str,
    bloque: dict,
    df_bloque: pd.DataFrame,
    ventas_completa: pd.DataFrame,
) -> dict:
    """
    Intenta SKU-trimestre y luego fallback:
    SKU global, subdepartamento-trimestre, departamento-trimestre, total-trimestre.
    """
    estimacion = estimar_elasticidad_loglog(df_sku_trimestre, fuente="SKU-trimestre")

    if pd.notna(estimacion["Elasticidad"]):
        return estimacion

    df_sku_global = ventas_completa[ventas_completa["prod_nbr"].astype(str) == str(sku)].copy()
    est_sku_global = estimar_elasticidad_loglog(df_sku_global, fuente="SKU-global")
    if pd.notna(est_sku_global["Elasticidad"]):
        return est_sku_global

    if "subdept_nm" in df_sku_trimestre.columns:
        modos = df_sku_trimestre["subdept_nm"].dropna().mode()
        if not modos.empty:
            subdept = modos.iloc[0]
            df_subdept_bloque = df_bloque[df_bloque["subdept_nm"] == subdept].copy()
            est_subdept = estimar_elasticidad_loglog(df_subdept_bloque, fuente="Subdepartamento-trimestre")
            if pd.notna(est_subdept["Elasticidad"]):
                return est_subdept

    if "dept_nm" in df_sku_trimestre.columns:
        modos = df_sku_trimestre["dept_nm"].dropna().mode()
        if not modos.empty:
            dept = modos.iloc[0]
            df_dept_bloque = df_bloque[df_bloque["dept_nm"] == dept].copy()
            est_dept = estimar_elasticidad_loglog(df_dept_bloque, fuente="Departamento-trimestre")
            if pd.notna(est_dept["Elasticidad"]):
                return est_dept

    est_total = estimar_elasticidad_loglog(df_bloque, fuente="Total-trimestre")
    if pd.notna(est_total["Elasticidad"]):
        return est_total

    return estimacion


def calculate_elasticity(
    ventas_limpias: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Calcula elasticidad trimestral robusta por SKU."""
    ventas = ventas_limpias.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()

    ventas["tran_date"] = pd.to_datetime(ventas["tran_date"], errors="coerce")
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas["net_sale"] = pd.to_numeric(ventas["net_sale"], errors="coerce")
    ventas["prod_nbr"] = ventas["prod_nbr"].astype(str)

    if "precio_unitario" not in ventas.columns:
        ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]
    else:
        ventas["precio_unitario"] = pd.to_numeric(ventas["precio_unitario"], errors="coerce")
        ventas["precio_unitario"] = ventas["precio_unitario"].fillna(ventas["net_sale"] / ventas["qty"])

    ventas = ventas.replace([np.inf, -np.inf], np.nan)
    ventas = ventas.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "precio_unitario"]).copy()
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0) & (ventas["precio_unitario"] > 0)].copy()

    ventas["mes"] = ventas["tran_date"].dt.to_period("M")
    ventas["fecha_dia"] = ventas["tran_date"].dt.date
    ventas["precio_modelo"] = ventas["precio_unitario"].round(2)

    ventas = integrate_promotions(ventas, promociones)
    bloques = build_three_month_blocks(ventas)

    resultados = []

    for bloque in bloques:
        df_bloque = ventas[ventas["mes"].isin(bloque["meses"])].copy()

        for sku, df_sku in df_bloque.groupby("prod_nbr"):
            n_obs = len(df_sku)
            precios_distintos = df_sku["precio_unitario"].round(2).nunique()

            estimacion = mejor_estimacion_con_fallback(
                df_sku_trimestre=df_sku,
                sku=sku,
                bloque=bloque,
                df_bloque=df_bloque,
                ventas_completa=ventas,
            )

            beta = estimacion["Elasticidad"]

            resultados.append(
                {
                    "prod_nbr": sku,
                    "SKU": sku,
                    "periodo_3m": bloque["periodo_3m"],
                    "trimestre": bloque["trimestre"],
                    "mes_inicio": str(bloque["mes_inicio"]),
                    "mes_fin": str(bloque["mes_fin"]),
                    "Beta": estimacion["Beta"],
                    "Elasticidad": estimacion["Elasticidad"],
                    "Alfa": estimacion["Alfa"],
                    "R2": estimacion["R2"],
                    "P_Value": estimacion["P_Value"],
                    "Observaciones": n_obs,
                    "Precios_Distintos": precios_distintos,
                    "Observaciones_Modelo": estimacion["Observaciones_Modelo"],
                    "Precios_Distintos_Modelo": estimacion["Precios_Distintos_Modelo"],
                    "Fuente_Elasticidad": estimacion["Fuente_Elasticidad"],
                    "Motivo_Modelo": estimacion["Motivo_Modelo"],
                    "Tiene_Promocion": df_sku["tiene_promocion"].max(),
                    "Num_Promociones": df_sku["num_promociones"].sum(),
                    "Diagnostico": diagnosticar_elasticidad(beta),
                    "dept_nm": _moda_segura(df_sku, "dept_nm"),
                    "subdept_nm": _moda_segura(df_sku, "subdept_nm"),
                    "marca": _moda_segura(df_sku, "marca"),
                    "tipo_marca": _moda_segura(df_sku, "tipo_marca"),
                    "categoria_est_socio": _moda_segura(df_sku, "categoria_est_socio"),
                    "estado": _moda_segura(df_sku, "estado"),
                }
            )

    elasticidad = pd.DataFrame(resultados)
    if not elasticidad.empty:
        elasticidad = elasticidad.replace([np.inf, -np.inf], np.nan)
        elasticidad = elasticidad.sort_values(by=["prod_nbr", "mes_inicio"]).reset_index(drop=True)

    return elasticidad, ventas, bloques


def build_elasticity_download(elasticidad_df: pd.DataFrame) -> pd.DataFrame:
    """Construye CSV de elasticidad con nombres solicitados."""
    if elasticidad_df is None or elasticidad_df.empty:
        return pd.DataFrame()

    out = elasticidad_df.copy()
    rename_map = {
        "Beta": "beta",
        "Elasticidad": "elasticidad",
        "Alfa": "alfa",
        "R2": "r2",
        "P_Value": "p-value",
        "Observaciones": "observaciones",
        "Diagnostico": "diagnóstico",
    }
    out = out.rename(columns=rename_map)
    columns = [
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
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def build_dynamic_explanation_elasticity(
    df_filtered: pd.DataFrame,
    filtros: dict,
) -> str:
    """Explicación dinámica para el dashboard de elasticidad."""
    if df_filtered is None or df_filtered.empty:
        return "No hay datos suficientes para explicar esta selección."

    elasticidad_prom = df_filtered["Elasticidad"].mean()
    r2_prom = df_filtered["R2"].mean()
    p_prom = df_filtered["P_Value"].mean()
    obs_total = df_filtered["Observaciones"].sum()
    diagnostico = (
        df_filtered["Diagnostico"].dropna().mode().iloc[0]
        if "Diagnostico" in df_filtered.columns and not df_filtered["Diagnostico"].dropna().mode().empty
        else "Sin diagnóstico"
    )

    riesgos = []
    if pd.notna(r2_prom) and r2_prom < 0.30:
        riesgos.append("R² bajo")
    if pd.notna(p_prom) and p_prom > 0.10:
        riesgos.append("p-value alto")
    if obs_total < 30:
        riesgos.append("pocas observaciones")
    if pd.notna(elasticidad_prom) and elasticidad_prom >= 0:
        riesgos.append("elasticidad positiva o sospechosa")

    filtros_txt = ", ".join([f"{k}: {v}" for k, v in filtros.items() if v not in [None, "Todos", "Todas", []]])
    filtros_txt = filtros_txt or "sin filtros específicos"

    riesgo_txt = " Riesgos detectados: " + ", ".join(riesgos) + "." if riesgos else " No se detectan alertas críticas inmediatas."

    return (
        f"Con {filtros_txt}, la elasticidad promedio es {elasticidad_prom:.3f} "
        f"y el diagnóstico dominante es '{diagnostico}'. "
        f"Un valor entre 0 y -1 sugiere demanda inelástica; menor a -1 sugiere demanda elástica; "
        f"un valor positivo debe revisarse porque puede indicar ruido o relación precio-demanda no esperada."
        f"{riesgo_txt}"
    )
