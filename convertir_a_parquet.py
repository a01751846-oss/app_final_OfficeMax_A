"""Convierte CSV o Excel a Parquet para acelerar la carga en Streamlit.

Uso:
    python convertir_a_parquet.py ventas.csv ventas.parquet
    python convertir_a_parquet.py ventas.xlsx ventas.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python convertir_a_parquet.py archivo_entrada.csv archivo_salida.parquet")
        raise SystemExit(1)

    entrada = Path(sys.argv[1])
    salida = Path(sys.argv[2])

    if not entrada.exists():
        print(f"No existe el archivo de entrada: {entrada}")
        raise SystemExit(1)

    print(f"Leyendo: {entrada}")

    if entrada.suffix.lower() == ".csv":
        ultimo_error = None
        for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(entrada, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as exc:
                ultimo_error = exc
                continue
        else:
            raise ultimo_error or ValueError("No se pudo leer el CSV.")
    elif entrada.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(entrada)
    else:
        print("Formato no soportado. Usa CSV, XLSX o XLS.")
        raise SystemExit(1)

    print(f"Filas: {len(df):,} | Columnas: {len(df.columns):,}")
    print(f"Guardando: {salida}")
    df.to_parquet(salida, index=False)
    print("Listo. Sube el archivo .parquet en la app para que cargue más rápido.")


if __name__ == "__main__":
    main()
