"""App principal de Streamlit para pricing dinámico, elasticidad y proyección de ventas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.config import (
    COLUMNAS_MINIMAS_VENTAS,
    COLUMNAS_LECTURA_NSE,
    COLUMNAS_LECTURA_PROMOCIONES,
    COLUMNAS_LECTURA_VENTAS,
    ESCENARIOS_PRICING,
    LEER_SOLO_COLUMNAS_NECESARIAS,
    MAX_ROWS_PREVIEW,
    MAX_SKUS_CURVA_ELASTICIDAD,
)
from modules.quality import calculate_quality_diagnosis
from modules.utils import (
    add_state_coordinates,
    build_default_nse,
    clean_sales_data,
    convert_df_to_csv,
    filter_dataframe_dependently,
    format_money,
    format_num,
    merge_sales_with_nse,
    get_uploaded_file_info,
    read_uploaded_file,
    render_kpi_card,
    validate_columns,
)


st.set_page_config(
    page_title="Pricing dinámico retail",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 1.4rem;}
    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px 18px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        min-height: 112px;
    }
    .kpi-title {
        color: #4b5563;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #111827;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .kpi-subtitle {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 8px;
    }
    .section-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        background: #f9fafb;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    """Inicializa session_state."""
    defaults = {
        "active_nse_df": build_default_nse(),
        "nse_source": "Base NSE predeterminada",
        "processed": False,
        "ventas_limpias": pd.DataFrame(),
        "elasticidad": pd.DataFrame(),
        "ventas_base_elasticidad": pd.DataFrame(),
        "bloques": [],
        "base_pricing": pd.DataFrame(),
        "simulacion": pd.DataFrame(),
        "resumen_pricing": pd.DataFrame(),
        "semaforo": pd.DataFrame(),
        "calidad_varianza": pd.DataFrame(),
        "resumen_limpieza": pd.DataFrame(),
        "nse_info": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, str]:
    """Renderiza sidebar y lee archivos cargados."""
    st.sidebar.title("📊 Pricing dinámico")
    st.sidebar.caption("Carga tus bases y navega entre las vistas.")

    vista = st.sidebar.radio(
        "Vista",
        [
            "1. Carga y diagnóstico de datos",
            "2. Elasticidad",
            "3. Pricing dinámico + proyección de ventas",
        ],
    )

    st.sidebar.divider()

    st.sidebar.subheader("Archivos")
    with st.sidebar.expander("A. Base de ventas obligatoria", expanded=True):
        st.info(
            "Sube un CSV, Excel o Parquet con ventas. Columnas mínimas: "
            "`tran_date`, `qty`, `net_sale`, `prod_nbr`, `costo2`."
        )
        sales_file = st.file_uploader(
            "Base de ventas",
            type=["csv", "xlsx", "xls", "parquet"],
            key="sales_file",
        )

    with st.sidebar.expander("B. Base de promociones opcional", expanded=False):
        st.info(
            "Opcional. Acepta CSV, Excel o Parquet. Si no se carga, la app seguirá funcionando y marcará promociones en cero. "
            "Si se carga, debe incluir un SKU (`prod_nbr`, `SKU` o similar) y fecha de inicio."
        )
        promo_file = st.file_uploader(
            "Base de promociones",
            type=["csv", "xlsx", "xls", "parquet"],
            key="promo_file",
        )

    with st.sidebar.expander("C. Base NSE", expanded=False):
        st.info(
            "Puedes continuar con la base NSE predeterminada, descargarla para modificarla, "
            "o subir una versión propia. El modelo final usa `categoria_est_socio`."
        )

        default_nse = build_default_nse()
        st.download_button(
            "Descargar base NSE predeterminada",
            data=convert_df_to_csv(default_nse),
            file_name="base_nse_predeterminada.csv",
            mime="text/csv",
        )

        nse_file = st.file_uploader(
            "Subir base NSE modificada",
            type=["csv", "xlsx", "xls", "parquet"],
            key="nse_file",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Aplicar cambios de NSE", use_container_width=True):
                if nse_file is None:
                    st.warning("Primero sube una base NSE modificada.")
                else:
                    try:
                        st.session_state.active_nse_df = read_uploaded_file(nse_file, usecols=COLUMNAS_LECTURA_NSE if LEER_SOLO_COLUMNAS_NECESARIAS else None)
                        st.session_state.nse_source = f"Base NSE subida: {nse_file.name}"
                        st.success("Cambios de NSE aplicados.")
                    except Exception as exc:
                        st.error(str(exc))
        with col_b:
            if st.button("Continuar con base predeterminada", use_container_width=True):
                st.session_state.active_nse_df = default_nse
                st.session_state.nse_source = "Base NSE predeterminada"
                st.success("Se usará la base predeterminada.")

        st.caption(f"NSE activo: {st.session_state.nse_source}")

    # Importante para rendimiento:
    # no leemos las bases grandes en cada rerun. Solo mostramos nombre/tamaño
    # y la lectura real ocurre cuando el usuario presiona el botón de procesar.
    if sales_file is not None:
        st.sidebar.success(f"Ventas listas para procesar: {get_uploaded_file_info(sales_file)}")
    if promo_file is not None:
        st.sidebar.success(f"Promociones listas para procesar: {get_uploaded_file_info(promo_file)}")

    process = st.sidebar.button("Procesar / actualizar datos", type="primary", use_container_width=True)

    if process:
        if sales_file is None:
            st.sidebar.error("Primero sube la base de ventas.")
        else:
            try:
                columnas_ventas = COLUMNAS_LECTURA_VENTAS if LEER_SOLO_COLUMNAS_NECESARIAS else None
                columnas_promos = COLUMNAS_LECTURA_PROMOCIONES if LEER_SOLO_COLUMNAS_NECESARIAS else None

                with st.spinner("Leyendo archivos cargados..."):
                    sales_df = read_uploaded_file(sales_file, usecols=columnas_ventas)
                    promo_df = read_uploaded_file(promo_file, usecols=columnas_promos) if promo_file is not None else None

                st.sidebar.success("Archivos leídos correctamente.")
                process_pipeline(sales_df, promo_df, st.session_state.active_nse_df)
            except Exception as exc:
                st.session_state.processed = False
                st.sidebar.error(str(exc))

    return pd.DataFrame(), None, st.session_state.active_nse_df, vista


def process_pipeline(sales_df: pd.DataFrame, promo_df: pd.DataFrame | None, nse_df: pd.DataFrame | None) -> None:
    """Ejecuta limpieza, NSE, calidad, elasticidad y pricing."""
    if sales_df is None or sales_df.empty:
        st.sidebar.error("La base de ventas está vacía o no se pudo leer.")
        return

    missing = validate_columns(sales_df, COLUMNAS_MINIMAS_VENTAS)
    if missing:
        st.sidebar.error("Faltan columnas obligatorias: " + ", ".join(missing))
        st.session_state.processed = False
        return

    try:
        with st.spinner("Procesando datos, calculando elasticidad y simulando pricing..."):
            # Imports diferidos: evitan cargar statsmodels/scikit-learn al abrir la app.
            from modules.elasticity import calculate_elasticity
            from modules.pricing import simulate_pricing_scenarios

            ventas_limpias, resumen_limpieza, summary = clean_sales_data(sales_df)
            ventas_nse, nse_info = merge_sales_with_nse(ventas_limpias, nse_df)
            semaforo, calidad_varianza = calculate_quality_diagnosis(ventas_nse, resumen_limpieza, summary)
            elasticidad, ventas_base_elasticidad, bloques = calculate_elasticity(ventas_nse, promo_df)
            base_pricing, simulacion, resumen_pricing = simulate_pricing_scenarios(
                ventas_base_elasticidad,
                elasticidad,
                bloques,
            )

        st.session_state.ventas_limpias = ventas_nse
        st.session_state.resumen_limpieza = resumen_limpieza
        st.session_state.nse_info = nse_info
        st.session_state.semaforo = semaforo
        st.session_state.calidad_varianza = calidad_varianza
        st.session_state.elasticidad = elasticidad
        st.session_state.ventas_base_elasticidad = ventas_base_elasticidad
        st.session_state.bloques = bloques
        st.session_state.base_pricing = base_pricing
        st.session_state.simulacion = simulacion
        st.session_state.resumen_pricing = resumen_pricing
        st.session_state.processed = True
        st.sidebar.success("Base limpiada, cruzada con NSE y procesada correctamente.")

    except Exception as exc:
        st.session_state.processed = False
        st.sidebar.error(f"No se pudo procesar la base: {exc}")


def require_processed() -> bool:
    """Valida que haya datos procesados."""
    if not st.session_state.processed:
        st.warning(
            "Carga una base de ventas y presiona **Procesar / actualizar datos** en el sidebar para activar esta vista."
        )
        return False
    return True


def render_quality_view() -> None:
    """Vista 1: carga y diagnóstico."""
    st.title("1. Carga y diagnóstico de datos")
    st.caption("Validación, limpieza, cruce NSE y semáforo de calidad.")

    st.markdown(
        """
        Esta vista toma la base de ventas cargada, limpia columnas críticas, calcula precio/costo/margen,
        cruza el NSE y genera un diagnóstico de calidad. Si no cargaste promociones, la app continúa sin ellas.
        """
    )

    if not require_processed():
        return

    ventas = st.session_state.ventas_limpias
    semaforo = st.session_state.semaforo
    resumen_limpieza = st.session_state.resumen_limpieza
    calidad_varianza = st.session_state.calidad_varianza
    nse_info = st.session_state.nse_info

    if not semaforo.empty:
        row = semaforo.iloc[0]
        color = "#dc2626" if "Rojo" in row["Semaforo"] else "#f59e0b" if "Amarillo" in row["Semaforo"] else "#16a34a"
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:16px; padding:18px; background:#ffffff;">
                <h3 style="margin-top:0;">Semáforo de calidad: {row['Semaforo']}</h3>
                <p style="margin-bottom:0;">{row['Interpretacion']}</p>
                <small>{row['Motivos']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("KPIs de calidad")
    c1, c2, c3, c4 = st.columns(4)
    row = semaforo.iloc[0] if not semaforo.empty else {}
    with c1:
        render_kpi_card("Registros originales", f"{int(row.get('Filas_Originales', 0)):,}", "Antes de limpieza")
    with c2:
        render_kpi_card("Registros limpios", f"{int(row.get('Filas_Limpias', 0)):,}", "Después de limpieza")
    with c3:
        render_kpi_card("Registros eliminados", f"{int(row.get('Registros_Removidos', 0)):,}", f"{row.get('%_Registros_Removidos', 0):.1f}% removido")
    with c4:
        render_kpi_card("Datos faltantes", f"{row.get('Porcentaje_Datos_Faltantes_Original', 0):.1f}%", "Promedio original")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_kpi_card("Duplicados", f"{int(row.get('Duplicados_Originales', 0)):,}", "Detectados originalmente")
    with c6:
        render_kpi_card("Valores infinitos", f"{int(row.get('Valores_Infinitos_Detectados', 0)):,}", "Antes de limpieza")
    with c7:
        render_kpi_card("Precio inválido", f"{int(row.get('Registros_Precio_Invalido', 0)):,}", "Después de crear precio")
    with c8:
        render_kpi_card("Cantidad inválida", f"{int(row.get('Registros_Cantidad_Invalida', 0)):,}", "qty <= 0")

    st.subheader("Cruce NSE")
    st.info(nse_info.get("mensaje", "NSE no aplicado."))
    if "categoria_est_socio" in ventas.columns:
        st.dataframe(
            ventas["categoria_est_socio"]
            .fillna("Sin dato")
            .value_counts(dropna=False)
            .rename_axis("categoria_est_socio")
            .reset_index(name="Registros"),
            use_container_width=True,
        )

    with st.expander("Resumen de limpieza"):
        st.dataframe(resumen_limpieza, use_container_width=True)

    with st.expander("Métricas de varianza"):
        st.dataframe(calidad_varianza, use_container_width=True)

    st.subheader("Vista previa de la base limpia y cruzada")
    st.dataframe(ventas.head(MAX_ROWS_PREVIEW), use_container_width=True)
    st.success("La base fue limpiada y cruzada con NSE correctamente.")


def render_elasticity_view() -> None:
    """Vista 2: elasticidad."""
    st.title("2. Elasticidad")
    st.caption("Elasticidad log-log por SKU y bloques fijos de 3 meses.")

    if not require_processed():
        return

    from modules.elasticity import build_dynamic_explanation_elasticity, build_elasticity_download

    df = st.session_state.elasticidad.copy()
    ventas = st.session_state.ventas_base_elasticidad.copy()

    if df.empty:
        st.warning("No se generaron resultados de elasticidad. Revisa fechas, SKUs y variación de precios.")
        return

    with st.expander("Cómo interpretar este dashboard", expanded=True):
        st.markdown(
            """
            La elasticidad mide qué tanto cambia la demanda ante un cambio de precio.
            Una elasticidad entre **0 y -1** indica demanda inelástica: puede tolerar incrementos.
            Una elasticidad **menor a -1** indica demanda elástica: conviene tener cuidado con subidas y evaluar promociones.
            Una elasticidad **positiva** es sospechosa o requiere revisión, porque sugiere que precio y demanda suben juntos.
            Un **R² bajo** o **p-value alto** no invalida automáticamente el resultado, pero sí aumenta el riesgo de interpretación.
            """
        )

    st.subheader("Filtros")
    c1, c2, c3 = st.columns(3)

    dept_options = ["Todos"] + sorted(df["dept_nm"].dropna().astype(str).unique().tolist()) if "dept_nm" in df.columns else ["Todos"]
    with c1:
        dept = st.selectbox("Departamento", dept_options)

    df_dep = filter_dataframe_dependently(df, {"dept_nm": dept})

    tri_options = ["Todos"] + sorted(df_dep["trimestre"].dropna().astype(str).unique().tolist())
    with c2:
        trimestre = st.selectbox("Trimestre", tri_options)

    df_dep_tri = filter_dataframe_dependently(df_dep, {"trimestre": trimestre})

    sku_options = sorted(df_dep_tri["SKU"].dropna().astype(str).unique().tolist())
    with c3:
        skus = st.multiselect("SKU", sku_options, default=sku_options[: min(5, len(sku_options))])

    filtered = filter_dataframe_dependently(df_dep_tri, {"SKU": skus})

    if filtered.empty:
        st.warning("No hay datos para la selección actual.")
        return

    st.subheader("KPIs")
    elasticidad_prom = filtered["Elasticidad"].mean()
    beta_prom = filtered["Beta"].mean()
    r2_prom = filtered["R2"].mean()
    diagnostico_dom = (
        filtered["Diagnostico"].dropna().mode().iloc[0]
        if not filtered["Diagnostico"].dropna().mode().empty
        else "Sin diagnóstico"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Elasticidad promedio", format_num(elasticidad_prom, 3), "Promedio filtrado")
    with k2:
        render_kpi_card("Beta promedio", format_num(beta_prom, 3), "Modelo log-log")
    with k3:
        render_kpi_card("R² promedio", format_num(r2_prom, 3), "Ajuste promedio")
    with k4:
        render_kpi_card("SKUs analizados", f"{filtered['SKU'].nunique():,}", "Únicos")
    with k5:
        render_kpi_card("Diagnóstico dominante", diagnostico_dom, "Moda")

    st.subheader("Curva de elasticidad")
    curva_df = filtered.dropna(subset=["Alfa", "Elasticidad"]).copy()
    if curva_df.empty:
        st.warning("No hay alfa/elasticidad suficiente para construir la curva.")
    else:
        precios = np.linspace(
            max(0.01, ventas["precio_unitario"].quantile(0.05)),
            max(0.02, ventas["precio_unitario"].quantile(0.95)),
            60,
        )
        curva_rows = []
        for _, row in curva_df.head(MAX_SKUS_CURVA_ELASTICIDAD).iterrows():
            alfa = row["Alfa"]
            beta = row["Elasticidad"]
            for precio in precios:
                demanda = np.exp(alfa + beta * np.log(precio))
                if np.isfinite(demanda):
                    curva_rows.append(
                        {
                            "SKU": row["SKU"],
                            "Precio": precio,
                            "Demanda estimada": demanda,
                            "trimestre": row["trimestre"],
                        }
                    )
        curva_plot = pd.DataFrame(curva_rows)
        if not curva_plot.empty:
            fig = px.line(curva_plot, x="Demanda estimada", y="Precio", color="SKU", title="Curva de elasticidad estimada")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(build_dynamic_explanation_elasticity(filtered, {"departamento": dept, "trimestre": trimestre, "SKU": ", ".join(skus[:5])}))

    st.subheader("Serie de tiempo de demanda")
    ventas_f = ventas.copy()
    if dept != "Todos" and "dept_nm" in ventas_f.columns:
        ventas_f = ventas_f[ventas_f["dept_nm"].astype(str) == str(dept)]
    if skus:
        ventas_f = ventas_f[ventas_f["prod_nbr"].astype(str).isin(skus)]
    if ventas_f.empty:
        st.warning("No hay ventas para la serie de tiempo con estos filtros.")
    else:
        if "tiene_promocion" in ventas_f.columns and ventas_f["tiene_promocion"].sum() > 0:
            serie = (
                ventas_f.groupby([pd.Grouper(key="tran_date", freq="W"), "tiene_promocion"], as_index=False)
                .agg(Demanda=("qty", "sum"))
            )
            serie["Promoción"] = np.where(serie["tiene_promocion"] == 1, "Con promoción", "Sin promoción")
            fig = px.line(serie, x="tran_date", y="Demanda", color="Promoción", title="Demanda semanal con/sin promoción")
        else:
            serie = ventas_f.groupby(pd.Grouper(key="tran_date", freq="W"), as_index=False).agg(Demanda=("qty", "sum"))
            fig = px.line(serie, x="tran_date", y="Demanda", title="Demanda semanal")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(build_dynamic_explanation_elasticity(filtered, {"departamento": dept, "trimestre": trimestre}))

    st.subheader("Mapa geográfico de México")
    estado_col = "estado" if "estado" in filtered.columns else None
    if estado_col is None or filtered[estado_col].dropna().empty:
        st.warning("No hay columna `estado` disponible para construir el mapa.")
    else:
        geo = (
            filtered.dropna(subset=[estado_col])
            .groupby(estado_col, as_index=False)
            .agg(Elasticidad=("Elasticidad", "mean"), SKUs=("SKU", "nunique"))
        )
        geo["Elasticidad absoluta"] = geo["Elasticidad"].abs()
        geo = add_state_coordinates(geo, estado_col=estado_col).dropna(subset=["lat", "lon"])
        if geo.empty:
            st.warning("No se pudieron homologar los estados a coordenadas de México.")
        else:
            fig = px.scatter_geo(
                geo,
                lat="lat",
                lon="lon",
                color="Elasticidad absoluta",
                size="Elasticidad absoluta",
                hover_name=estado_col,
                hover_data={"Elasticidad": ":.3f", "SKUs": True, "lat": False, "lon": False},
                scope="north america",
                title="Intensidad de elasticidad absoluta por estado",
            )
            fig.update_geos(fitbounds="locations", visible=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(build_dynamic_explanation_elasticity(filtered, {"departamento": dept, "trimestre": trimestre, "estado": "mapa"}))

    st.subheader("Descarga")
    elasticidad_csv = build_elasticity_download(df)
    st.download_button(
        "Descargar CSV de elasticidad por SKU y trimestre",
        data=convert_df_to_csv(elasticidad_csv),
        file_name="elasticidad_por_sku_trimestre.csv",
        mime="text/csv",
    )


def render_pricing_view() -> None:
    """Vista 3: pricing dinámico y proyección."""
    st.title("3. Pricing dinámico + proyección de ventas")
    st.caption("Simulación de escenarios, KPIs, proyección y recomendación del mejor escenario.")

    if not require_processed():
        return

    from modules.pricing import build_dynamic_explanation_pricing, build_pricing_downloads

    sim = st.session_state.simulacion.copy()
    resumen = st.session_state.resumen_pricing.copy()

    if sim.empty:
        st.warning("No hay simulaciones de pricing. Revisa elasticidad, costos y bloques trimestrales.")
        return

    st.subheader("Filtros")

    f1, f2, f3 = st.columns(3)
    with f1:
        cat_options = ["Todas"] + sorted(sim["Categoria_RF"].dropna().astype(str).unique().tolist())
        categoria = st.selectbox("Categoría de SKU", cat_options)
    df_cat = filter_dataframe_dependently(sim, {"Categoria_RF": categoria})

    with f2:
        tri_options = ["Todos"] + sorted(df_cat["trimestre"].dropna().astype(str).unique().tolist())
        trimestre = st.selectbox("Trimestre", tri_options)
    df_tri = filter_dataframe_dependently(df_cat, {"trimestre": trimestre})

    with f3:
        dept_options = ["Todos"] + sorted(df_tri["dept_nm"].dropna().astype(str).unique().tolist()) if "dept_nm" in df_tri.columns else ["Todos"]
        dept = st.selectbox("Departamento", dept_options)
    df_dept = filter_dataframe_dependently(df_tri, {"dept_nm": dept})

    f4, f5, f6 = st.columns(3)
    with f4:
        estado_options = ["Todos"] + sorted(df_dept["estado"].dropna().astype(str).unique().tolist()) if "estado" in df_dept.columns else ["Todos"]
        estado = st.selectbox("Estado", estado_options)
    df_estado = filter_dataframe_dependently(df_dept, {"estado": estado})

    with f5:
        nse_options = ["Todos"] + sorted(df_estado["categoria_est_socio"].dropna().astype(str).unique().tolist()) if "categoria_est_socio" in df_estado.columns else ["Todos"]
        nse = st.selectbox("Nivel socioeconómico", nse_options)
    df_nse = filter_dataframe_dependently(df_estado, {"categoria_est_socio": nse})

    sku_options = sorted(df_nse["SKU"].dropna().astype(str).unique().tolist())
    with f6:
        sku = st.selectbox("SKU", ["Todos"] + sku_options)

    df_sku = filter_dataframe_dependently(df_nse, {"SKU": sku})

    escenario_options = ESCENARIOS_PRICING["Nombre_Escenario"].tolist()
    escenario = st.selectbox("Escenario de pricing", escenario_options)
    selected = df_sku[df_sku["Nombre_Escenario"] == escenario].copy()

    if selected.empty:
        st.warning("No hay resultados para la combinación de filtros y escenario seleccionado.")
        return

    card1, card2 = st.columns(2)
    with card1:
        cat_sel = selected["Categoria_RF"].dropna().mode().iloc[0] if not selected["Categoria_RF"].dropna().mode().empty else "Sin categoría"
        render_kpi_card("Categoría del SKU/grupo", cat_sel, "Según RF conservador o reglas")
    with card2:
        if sku != "Todos":
            best = resumen[resumen["SKU"].astype(str) == str(sku)].copy()
            if trimestre != "Todos":
                best = best[best["trimestre"].astype(str) == str(trimestre)]
            best_scen = best["Escenario_Ideal"].iloc[0] if not best.empty else "Sin dato"
            render_kpi_card("Mejor escenario", best_scen, "Para el SKU seleccionado")
        else:
            render_kpi_card("Mejor escenario", "Selecciona un SKU", "Disponible por SKU")

    st.subheader("KPIs proyectados")
    unidades = selected["Unidades_Simuladas"].sum()
    ingreso = selected["Ingreso_Simulado"].sum()
    margen = selected["Margen_Simulado"].sum()

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Unidades simuladas", format_num(unidades, 0), escenario)
    with k2:
        render_kpi_card("Ingreso predicho", format_money(ingreso), escenario)
    with k3:
        render_kpi_card("Margen predicho", format_money(margen), escenario)

    st.subheader("Ventas en dinero")
    money_df = selected.copy()
    money_group = (
        money_df.groupby("trimestre", as_index=False)
        .agg(Ventas_normales=("Ingreso_Base", "sum"), Ventas_simuladas=("Ingreso_Simulado", "sum"))
    )
    money_long = money_group.melt(id_vars="trimestre", value_vars=["Ventas_normales", "Ventas_simuladas"], var_name="Serie", value_name="Ventas")
    fig = px.line(money_long, x="trimestre", y="Ventas", color="Serie", markers=True, title="Ventas normales vs ventas simuladas")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Ventas en cantidad")
    qty_group = (
        selected.groupby("trimestre", as_index=False)
        .agg(Cantidad_normal=("Unidades_Base", "sum"), Cantidad_simulada=("Unidades_Simuladas", "sum"))
    )
    qty_long = qty_group.melt(id_vars="trimestre", value_vars=["Cantidad_normal", "Cantidad_simulada"], var_name="Serie", value_name="Unidades")
    fig = px.line(qty_long, x="trimestre", y="Unidades", color="Serie", markers=True, title="Cantidad normal vs cantidad simulada")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Ingreso vs margen")
    im_group = (
        selected.groupby("trimestre", as_index=False)
        .agg(Ingreso_simulado=("Ingreso_Simulado", "sum"), Margen_simulado=("Margen_Simulado", "sum"))
    )
    im_long = im_group.melt(id_vars="trimestre", value_vars=["Ingreso_simulado", "Margen_simulado"], var_name="Métrica", value_name="Monto")
    fig = px.bar(im_long, x="trimestre", y="Monto", color="Métrica", barmode="group", title="Ingreso simulado vs margen simulado")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Conclusión personalizada")
    st.info(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    with st.expander("Tabla de resultados filtrados"):
        cols = [
            "SKU",
            "trimestre",
            "Nombre_Escenario",
            "Categoria_RF",
            "Elasticidad",
            "R2",
            "P_Value",
            "Unidades_Base",
            "Unidades_Simuladas",
            "Ingreso_Base",
            "Ingreso_Simulado",
            "Margen_Base",
            "Margen_Simulado",
            "Escenario_Ideal",
        ]
        st.dataframe(selected[[c for c in cols if c in selected.columns]], use_container_width=True)

    st.subheader("Descargas")
    exp_csv, best_csv = build_pricing_downloads(sim, resumen)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Descargar CSV completo con todos los experimentos",
            data=convert_df_to_csv(exp_csv),
            file_name="pricing_todos_los_escenarios.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Descargar CSV con mejor escenario",
            data=convert_df_to_csv(best_csv),
            file_name="pricing_mejor_escenario.csv",
            mime="text/csv",
        )


def main() -> None:
    """Punto de entrada de la app."""
    init_state()
    _, _, _, vista = render_sidebar()

    if vista.startswith("1."):
        render_quality_view()
    elif vista.startswith("2."):
        render_elasticity_view()
    else:
        render_pricing_view()


if __name__ == "__main__":
    main()
