import pandas as pd

# Leer CSV (ajusta separador si no es coma)
df = pd.read_csv("TFG_Results_EmotionalAnalysis.kaggle_mistral_results_concurrente.csv", sep=",", dtype=str)

# Nombre de la columna ID (ajústalo si se llama distinto)
ID_COL = df.columns[1]  # en tu ejemplo parece la segunda columna (40156)

# Contar NA por fila
df["na_count"] = df.isna().sum(axis=1)

# Función para filtrar cada grupo de IDs
def filtrar_grupo(grupo):
    if len(grupo) == 1:
        return grupo  # no hay duplicados

    # mínimo número de NA dentro del grupo
    min_na = grupo["na_count"].min()

    # quedarse solo con las filas que tengan ese mínimo
    return grupo[grupo["na_count"] == min_na]

# Aplicar filtro por grupos
df_filtrado = df.groupby(ID_COL, group_keys=False).apply(filtrar_grupo)

# Eliminar columna auxiliar
df_filtrado = df_filtrado.drop(columns=["na_count"])

# Guardar resultado
df_filtrado.to_csv("output.csv", sep=";", index=False)