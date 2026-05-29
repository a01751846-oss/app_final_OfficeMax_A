"""Funciones utilitarias de lectura, validación, limpieza y UI."""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from .config import STATE_COORDINATES


def normalize_text(value) -> str:
    """Normaliza texto para comparaciones robustas."""
    if pd.isna(value):
        return ""
    txt = str(value).strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8")
    txt = re.sub(r"[_\-]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _normalizar_lista_columnas(usecols: Optional[Sequence[str]]) -> tuple[str, ...] | None:
    """Normaliza usecols para poder cachear lecturas con columnas seleccionadas."""
    if not usecols:
        return None
    return tuple(dict.fromkeys(str(c).strip() for c in usecols if str(c).strip()))


def _usecols_callable(columnas_permitidas: tuple[str, ...] | None):
    """Crea un filtro flexible de columnas para pandas."""
    if not columnas_permitidas:
        return None
    permitidas = {str(c).strip() for c in columnas_permitidas}
    return lambda col: str(col).strip() in permitidas


def _dtype_map_for_columns(columnas: tuple[str, ...] | None) -> dict:
    """Tipos ligeros para columnas categóricas; las numéricas se limpian después."""
    if not columnas:
        columnas = tuple()
    text_cols = {
        "prod_nbr", "SKU", "store_nm", "dept_nm", "subdept_nm", "marca", "tipo_marca",
        "categoria_est_socio", "estado", "key", "id_municipio", "ubica_geo", "municipio",
        "est_socio", "sku", "promo_id", "id_promocion", "mecanica",
    }
    return {c: "string" for c in columnas if c in text_cols}


@st.cache_data(show_spinner=False, max_entries=6)
def _read_uploaded_file_cached(
    file_name: str,
    file_size: int,
    file_bytes: bytes,
    usecols_tuple: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Lee archivos desde bytes con caché.

    Optimización importante:
    - Si usecols_tuple viene definido, pandas solo carga esas columnas.
    - Esto conserva la calidad del modelo porque las columnas seleccionadas son las que usa la app.
    - Parquet se lee mucho más rápido que CSV/Excel cuando está disponible.
    """
    del file_size  # ayuda a invalidar cache si cambia el archivo, aunque no se usa directamente.
    name = file_name.lower()
    usecols = _usecols_callable(usecols_tuple)
    dtype_map = _dtype_map_for_columns(usecols_tuple)

    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(file_bytes), columns=list(usecols_tuple) if usecols_tuple else None)

    if name.endswith(".csv"):
        last_error = None
        for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    usecols=usecols,
                    dtype=dtype_map if dtype_map else None,
                    low_memory=False,
                )
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except ValueError as exc:
                # Si usecols no encuentra ninguna columna, permite que el error suba con mensaje claro.
                raise ValueError(f"No se pudieron seleccionar columnas del CSV. Revisa encabezados. Detalle: {exc}") from exc
        return pd.read_csv(
            io.BytesIO(file_bytes),
            encoding_errors="replace",
            usecols=usecols,
            dtype=dtype_map if dtype_map else None,
            low_memory=False,
        )

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(
            io.BytesIO(file_bytes),
            usecols=usecols,
            dtype=dtype_map if dtype_map else None,
        )

    raise ValueError("Formato no compatible. Sube un archivo CSV, XLSX, XLS o PARQUET.")


def read_uploaded_file(uploaded_file, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Lee CSV, Excel o Parquet cargado desde Streamlit.

    Nota de rendimiento:
    Esta función debe llamarse solo al presionar el botón de procesar, no en cada rerun.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        file_bytes = uploaded_file.getvalue()
        usecols_tuple = _normalizar_lista_columnas(usecols)
        return _read_uploaded_file_cached(
            uploaded_file.name,
            getattr(uploaded_file, "size", len(file_bytes)),
            file_bytes,
            usecols_tuple,
        )
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo '{uploaded_file.name}': {exc}") from exc


def get_uploaded_file_info(uploaded_file) -> str:
    """Devuelve nombre y tamaño sin leer todo el archivo como DataFrame."""
    if uploaded_file is None:
        return "Sin archivo"
    size = getattr(uploaded_file, "size", None)
    if size is None:
        return uploaded_file.name
    mb = size / (1024 ** 2)
    return f"{uploaded_file.name} ({mb:.1f} MB)"

def validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    """Regresa las columnas faltantes."""
    if df is None or df.empty:
        return list(required_columns)
    disponibles = set(df.columns.astype(str).str.strip())
    return [col for col in required_columns if col not in disponibles]


def clean_store_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia espacios dobles, iniciales y finales de store_nm."""
    df = df.copy()
    if "store_nm" in df.columns:
        df["store_nm"] = df["store_nm"].where(
            df["store_nm"].isna(),
            df["store_nm"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        )
    return df



def parse_transaction_dates(series: pd.Series) -> pd.Series:
    """
    Convierte tran_date a datetime de forma explícita y rápida.

    La base de ventas suele venir en formato mexicano dd/mm/YYYY.
    Usar format="%d/%m/%Y" evita el warning de pandas y reduce ambigüedad.
    El fallback con dayfirst=True conserva compatibilidad si aparecen fechas ISO
    u otros formatos válidos en algunas filas.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    fechas = series.astype("string").str.strip()

    # Ruta rápida y sin warning para el formato esperado de la base: 31/12/2025.
    parsed = pd.to_datetime(fechas, format="%d/%m/%Y", errors="coerce")

    # Fallback para formatos mixtos como 2025-12-31 o timestamps.
    pendientes = parsed.isna() & fechas.notna() & (fechas != "")
    if pendientes.any():
        parsed.loc[pendientes] = pd.to_datetime(
            fechas.loc[pendientes],
            errors="coerce",
            dayfirst=True,
        )

    return parsed

def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convierte texto monetario o numérico a número."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def clean_sales_data(
    sales_df: pd.DataFrame,
    costo2_es_unitario: bool = True,
    eliminar_costo_mayor_o_igual_precio: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Limpia la base de ventas siguiendo la lógica del notebook:
    duplicados, fechas, numéricos, qty/net_sale válidos, precio/costo/margen.
    """
    if sales_df is None or sales_df.empty:
        raise ValueError("La base de ventas está vacía.")

    ventas = sales_df.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()
    filas_originales = len(ventas)
    resumen = []

    def registrar(paso: str, antes: int, despues: int) -> None:
        resumen.append(
            {
                "Paso": paso,
                "Filas_Antes": int(antes),
                "Filas_Despues": int(despues),
                "Registros_Removidos": int(antes - despues),
                "%_Removido": float(((antes - despues) / antes * 100) if antes > 0 else 0),
            }
        )

    ventas = clean_store_names(ventas)

    antes = len(ventas)
    ventas = ventas.drop_duplicates().copy()
    registrar("Eliminar registros duplicados", antes, len(ventas))

    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])

    for col in ["qty", "net_sale", "prod_nbr", "costo2"]:
        ventas[col] = _safe_numeric(ventas[col])

    antes = len(ventas)
    ventas = ventas.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"]).copy()
    registrar("Eliminar nulos en columnas críticas", antes, len(ventas))

    antes = len(ventas)
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0)].copy()
    registrar("Eliminar qty <= 0 y net_sale <= 0", antes, len(ventas))

    ventas["prod_nbr"] = ventas["prod_nbr"].astype("Int64").astype(str)
    ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]

    antes = len(ventas)
    ventas = ventas.replace([np.inf, -np.inf], np.nan)
    ventas = ventas.dropna(subset=["precio_unitario"]).copy()
    ventas = ventas[ventas["precio_unitario"] > 0].copy()
    registrar("Eliminar precio <= 0 o no calculable", antes, len(ventas))

    if costo2_es_unitario:
        ventas["costo_unitario"] = ventas["costo2"]
    else:
        ventas["costo_unitario"] = ventas["costo2"] / ventas["qty"]

    ventas = ventas.replace([np.inf, -np.inf], np.nan)

    antes = len(ventas)
    ventas = ventas.dropna(subset=["costo_unitario"]).copy()
    ventas = ventas[ventas["costo_unitario"] >= 0].copy()
    registrar("Eliminar costo negativo o no calculable", antes, len(ventas))

    ventas["Costo_Mayor_O_Igual_Precio_Linea"] = ventas["costo_unitario"] >= ventas["precio_unitario"]
    filas_costo_invalido = int(ventas["Costo_Mayor_O_Igual_Precio_Linea"].sum())

    if filas_costo_invalido > 0 and eliminar_costo_mayor_o_igual_precio:
        antes = len(ventas)
        ventas = ventas[~ventas["Costo_Mayor_O_Igual_Precio_Linea"]].copy()
        registrar("Eliminar costo unitario >= precio", antes, len(ventas))
    else:
        registrar("Verificar costo unitario < precio", len(ventas), len(ventas))

    ventas["precio_base"] = pd.to_numeric(
        ventas["precio_base"], errors="coerce"
    ) if "precio_base" in ventas.columns else ventas["precio_unitario"]
    ventas["precio_base"] = ventas["precio_base"].fillna(ventas["precio_unitario"])

    ventas["ingreso_base"] = pd.to_numeric(
        ventas["ingreso_base"], errors="coerce"
    ) if "ingreso_base" in ventas.columns else ventas["precio_base"] * ventas["qty"]
    ventas["ingreso_base"] = ventas["ingreso_base"].fillna(ventas["precio_base"] * ventas["qty"])

    ventas["margen_unitario"] = pd.to_numeric(
        ventas["margen_unitario"], errors="coerce"
    ) if "margen_unitario" in ventas.columns else ventas["precio_base"] - ventas["costo_unitario"]
    ventas["margen_unitario"] = ventas["margen_unitario"].fillna(
        ventas["precio_base"] - ventas["costo_unitario"]
    )

    ventas["margen_total"] = pd.to_numeric(
        ventas["margen_total"], errors="coerce"
    ) if "margen_total" in ventas.columns else ventas["margen_unitario"] * ventas["qty"]
    ventas["margen_total"] = ventas["margen_total"].fillna(ventas["margen_unitario"] * ventas["qty"])

    ventas["periodo_mensual"] = ventas["tran_date"].dt.to_period("M").astype(str)

    summary = {
        "filas_originales": int(filas_originales),
        "filas_limpias": int(len(ventas)),
        "registros_removidos": int(filas_originales - len(ventas)),
        "porcentaje_removido": float(((filas_originales - len(ventas)) / filas_originales) if filas_originales else 1),
        "duplicados_originales": int(sales_df.duplicated().sum()),
        "faltantes_pct_original": float(sales_df.isna().mean().mean() * 100),
        "infinitos_detectados_original": int(
            np.isinf(sales_df.select_dtypes(include=[np.number]).to_numpy()).sum()
        )
        if len(sales_df.select_dtypes(include=[np.number]).columns) > 0
        else 0,
        "registros_precio_invalido": int((ventas["precio_unitario"] <= 0).sum()) if "precio_unitario" in ventas else 0,
        "registros_cantidad_invalida": int((ventas["qty"] <= 0).sum()) if "qty" in ventas else 0,
        "registros_costo_mayor_o_igual_precio": filas_costo_invalido,
    }

    return ventas.reset_index(drop=True), pd.DataFrame(resumen), summary


def normalizar_categoria_est_socio(valor):
    """Normaliza est_socio o categoria_est_socio a bajo, medio bajo, medio alto, alto."""
    if pd.isna(valor):
        return np.nan

    try:
        numero = float(str(valor).strip().replace(",", "."))
        if np.isfinite(numero):
            codigo = str(int(round(numero)))
            mapa_num = {"1": "bajo", "2": "medio bajo", "3": "medio alto", "4": "alto"}
            if codigo in mapa_num:
                return mapa_num[codigo]
    except Exception:
        pass

    txt = normalize_text(valor)
    mapa_txt = {
        "bajo": "bajo",
        "medio bajo": "medio bajo",
        "medio alto": "medio alto",
        "alto": "alto",
    }
    return mapa_txt.get(txt, np.nan)


def _limpiar_id_municipio(valor):
    """Homologa llaves geográficas numéricas."""
    if pd.isna(valor):
        return np.nan
    txt = str(valor).strip().replace(",", "")
    txt = re.sub(r"\.0$", "", txt)
    return txt


@st.cache_data(show_spinner=False)
def build_default_nse() -> pd.DataFrame:
    """
    Crea una base NSE predeterminada mínima y editable.
    El usuario puede reemplazarla por una base INEGI completa.
    """
    rows = [
        ("Aguascalientes-Aguascalientes", 1001, "Aguascalientes", "Aguascalientes", 3),
        ("Mexicali-Baja California", 2002, "Baja California", "Mexicali", 3),
        ("La Paz-Baja California Sur", 3003, "Baja California Sur", "La Paz", 3),
        ("Campeche-Campeche", 4002, "Campeche", "Campeche", 2),
        ("Tuxtla Gutierrez-Chiapas", 7101, "Chiapas", "Tuxtla Gutierrez", 2),
        ("Chihuahua-Chihuahua", 8019, "Chihuahua", "Chihuahua", 3),
        ("Saltillo-Coahuila", 5030, "Coahuila", "Saltillo", 3),
        ("Colima-Colima", 6002, "Colima", "Colima", 3),
        ("Cuauhtemoc-Ciudad de Mexico", 9015, "Ciudad de Mexico", "Cuauhtemoc", 4),
        ("Durango-Durango", 10005, "Durango", "Durango", 3),
        ("Leon-Guanajuato", 11020, "Guanajuato", "Leon", 3),
        ("Acapulco de Juarez-Guerrero", 12001, "Guerrero", "Acapulco de Juarez", 2),
        ("Pachuca de Soto-Hidalgo", 13048, "Hidalgo", "Pachuca de Soto", 3),
        ("Guadalajara-Jalisco", 14039, "Jalisco", "Guadalajara", 4),
        ("Toluca-Mexico", 15106, "Estado de Mexico", "Toluca", 3),
        ("Morelia-Michoacan", 16053, "Michoacan", "Morelia", 3),
        ("Cuernavaca-Morelos", 17007, "Morelos", "Cuernavaca", 3),
        ("Tepic-Nayarit", 18017, "Nayarit", "Tepic", 3),
        ("Monterrey-Nuevo Leon", 19039, "Nuevo Leon", "Monterrey", 4),
        ("Oaxaca de Juarez-Oaxaca", 20067, "Oaxaca", "Oaxaca de Juarez", 2),
        ("Puebla-Puebla", 21114, "Puebla", "Puebla", 3),
        ("Queretaro-Queretaro", 22014, "Queretaro", "Queretaro", 4),
        ("Benito Juarez-Quintana Roo", 23005, "Quintana Roo", "Benito Juarez", 3),
        ("San Luis Potosi-San Luis Potosi", 24028, "San Luis Potosi", "San Luis Potosi", 3),
        ("Culiacan-Sinaloa", 25006, "Sinaloa", "Culiacan", 3),
        ("Hermosillo-Sonora", 26030, "Sonora", "Hermosillo", 3),
        ("Centro-Tabasco", 27004, "Tabasco", "Centro", 3),
        ("Tampico-Tamaulipas", 28038, "Tamaulipas", "Tampico", 3),
        ("Tlaxcala-Tlaxcala", 29033, "Tlaxcala", "Tlaxcala", 3),
        ("Veracruz-Veracruz", 30193, "Veracruz", "Veracruz", 3),
        ("Merida-Yucatan", 31050, "Yucatan", "Merida", 4),
        ("Zacatecas-Zacatecas", 32056, "Zacatecas", "Zacatecas", 3),
    ]
    df = pd.DataFrame(rows, columns=["key", "ubica_geo", "estado", "municipio", "est_socio"])
    df["categoria_est_socio"] = df["est_socio"].apply(normalizar_categoria_est_socio)
    return df


def merge_sales_with_nse(sales_df: pd.DataFrame, nse_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """
    Cruza ventas con NSE de forma flexible.
    Prioriza categoria_est_socio y usa est_socio solo para construirla.
    """
    ventas = sales_df.copy()
    info = {
        "nse_usado": False,
        "registros_con_nse": 0,
        "registros_sin_nse": len(ventas),
        "porcentaje_asignado": 0.0,
        "mensaje": "No se aplicó cruce NSE.",
    }

    if "categoria_est_socio" in ventas.columns:
        ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)

    if nse_df is None or nse_df.empty:
        if "categoria_est_socio" not in ventas.columns:
            ventas["categoria_est_socio"] = np.nan
        return ventas, info

    nse = nse_df.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()
    nse.columns = nse.columns.astype(str).str.strip()

    # Construir categoria_est_socio en NSE si no viene explícita.
    if "categoria_est_socio" in nse.columns:
        nse["categoria_est_socio"] = nse["categoria_est_socio"].apply(normalizar_categoria_est_socio)
    elif "est_socio" in nse.columns:
        nse["categoria_est_socio"] = nse["est_socio"].apply(normalizar_categoria_est_socio)
    else:
        if "categoria_est_socio" not in ventas.columns:
            ventas["categoria_est_socio"] = np.nan
        info["mensaje"] = "La base NSE no contiene categoria_est_socio ni est_socio."
        return ventas, info

    # Si la base NSE es granular por hogares, calcular moda por municipio.
    if "ubica_geo" in nse.columns:
        nse["id_municipio"] = nse["ubica_geo"].apply(_limpiar_id_municipio)

    if "id_municipio" in nse.columns:
        nse["id_municipio"] = nse["id_municipio"].apply(_limpiar_id_municipio)
        nse_municipio = (
            nse.dropna(subset=["id_municipio", "categoria_est_socio"])
            .groupby("id_municipio", as_index=False)
            .agg(
                categoria_est_socio=("categoria_est_socio", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                estado=("estado", lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else np.nan) if "estado" in nse.columns else ("id_municipio", "first"),
            )
        )
    else:
        nse_municipio = pd.DataFrame()

    # Catálogo key -> ubica_geo/id_municipio si viene en NSE o en catálogo predeterminado.
    if "key" in ventas.columns:
        ventas["key"] = ventas["key"].astype(str).str.strip().str.lower()

    if "key" in nse.columns:
        nse["key"] = nse["key"].astype(str).str.strip().str.lower()

    if "key" in ventas.columns and "key" in nse.columns:
        cols_key = ["key"]
        if "ubica_geo" in nse.columns:
            cols_key.append("ubica_geo")
        if "id_municipio" in nse.columns:
            cols_key.append("id_municipio")
        if "estado" in nse.columns:
            cols_key.append("estado")
        if "categoria_est_socio" in nse.columns:
            cols_key.append("categoria_est_socio")

        nse_key = nse[cols_key].drop_duplicates("key").copy()
        nse_key = nse_key.rename(
            columns={
                "categoria_est_socio": "categoria_est_socio_nse",
                "estado": "estado_nse",
            }
        )
        ventas = ventas.drop(columns=["id_municipio"], errors="ignore") if "id_municipio" in ventas.columns else ventas
        ventas = ventas.merge(nse_key, on="key", how="left")
        if "ubica_geo" in ventas.columns and "id_municipio" not in ventas.columns:
            ventas = ventas.rename(columns={"ubica_geo": "id_municipio"})
        elif "ubica_geo" in ventas.columns and "id_municipio" in ventas.columns:
            ventas["id_municipio"] = ventas["id_municipio"].fillna(ventas["ubica_geo"])
            ventas = ventas.drop(columns=["ubica_geo"], errors="ignore")

        if "categoria_est_socio_nse" in ventas.columns:
            if "categoria_est_socio" not in ventas.columns:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio_nse"]
            else:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio"].fillna(ventas["categoria_est_socio_nse"])
            ventas = ventas.drop(columns=["categoria_est_socio_nse"], errors="ignore")
        if "estado_nse" in ventas.columns:
            if "estado" not in ventas.columns:
                ventas["estado"] = ventas["estado_nse"]
            else:
                ventas["estado"] = ventas["estado"].fillna(ventas["estado_nse"])
            ventas = ventas.drop(columns=["estado_nse"], errors="ignore")

    # Si hay id_municipio, cruzar por municipio.
    if "id_municipio" in ventas.columns and not nse_municipio.empty:
        ventas["id_municipio"] = ventas["id_municipio"].apply(_limpiar_id_municipio)
        ventas = ventas.merge(
            nse_municipio.rename(
                columns={
                    "categoria_est_socio": "categoria_est_socio_mpio",
                    "estado": "estado_mpio",
                }
            ),
            on="id_municipio",
            how="left",
        )
        if "categoria_est_socio_mpio" in ventas.columns:
            if "categoria_est_socio" not in ventas.columns:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio_mpio"]
            else:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio"].fillna(ventas["categoria_est_socio_mpio"])
            ventas = ventas.drop(columns=["categoria_est_socio_mpio"], errors="ignore")
        if "estado_mpio" in ventas.columns:
            if "estado" not in ventas.columns:
                ventas["estado"] = ventas["estado_mpio"]
            else:
                ventas["estado"] = ventas["estado"].fillna(ventas["estado_mpio"])
            ventas = ventas.drop(columns=["estado_mpio"], errors="ignore")

    if "categoria_est_socio" not in ventas.columns:
        ventas["categoria_est_socio"] = np.nan

    ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)
    asignados = int(ventas["categoria_est_socio"].notna().sum())
    total = len(ventas)

    info.update(
        {
            "nse_usado": True,
            "registros_con_nse": asignados,
            "registros_sin_nse": int(total - asignados),
            "porcentaje_asignado": float((asignados / total * 100) if total else 0),
            "mensaje": f"NSE asignado a {asignados:,} de {total:,} registros.",
        }
    )
    return ventas, info


def build_quarter_label(periodo_3m: str) -> str:
    """Convierte '2025-01 a 2025-03' en 'ene 2025 - mar 2025'."""
    if pd.isna(periodo_3m):
        return "Sin trimestre"
    txt = str(periodo_3m)
    meses = {
        1: "ene",
        2: "feb",
        3: "mar",
        4: "abr",
        5: "may",
        6: "jun",
        7: "jul",
        8: "ago",
        9: "sep",
        10: "oct",
        11: "nov",
        12: "dic",
    }
    try:
        partes = [p.strip() for p in txt.split(" a ")]
        if len(partes) != 2:
            return txt
        inicio = pd.Period(partes[0], freq="M")
        fin = pd.Period(partes[1], freq="M")
        return f"{meses[inicio.month]} {inicio.year} - {meses[fin.month]} {fin.year}"
    except Exception:
        return txt


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convierte DataFrame a CSV descargable."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def render_kpi_card(title: str, value, subtitle: str = "") -> None:
    """Renderiza una tarjeta KPI simple."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_dataframe_dependently(
    df: pd.DataFrame,
    filters: dict[str, object],
) -> pd.DataFrame:
    """Filtra un DataFrame con filtros dependientes tipo 'Todos' o listas."""
    out = df.copy()
    for col, value in filters.items():
        if col not in out.columns or value in [None, "Todos", "Todas"]:
            continue
        if isinstance(value, (list, tuple, set)):
            selected = [v for v in value if v not in ["Todos", "Todas"]]
            if selected:
                out = out[out[col].astype(str).isin([str(v) for v in selected])]
        else:
            out = out[out[col].astype(str) == str(value)]
    return out


def add_state_coordinates(df: pd.DataFrame, estado_col: str = "estado") -> pd.DataFrame:
    """Agrega lat/lon aproximados para estados de México."""
    out = df.copy()
    if estado_col not in out.columns:
        return out
    keys = out[estado_col].apply(normalize_text)
    out["lat"] = keys.map(lambda k: STATE_COORDINATES.get(k, (np.nan, np.nan))[0])
    out["lon"] = keys.map(lambda k: STATE_COORDINATES.get(k, (np.nan, np.nan))[1])
    return out


def format_money(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


def format_num(x, dec: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{dec}f}"


def format_pct(x, dec: int = 1) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:+,.{dec}f}%"
