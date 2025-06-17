import pandas as pd

def carregar_dados():
    acidentes = pd.read_csv("data/raw/acidentes2025_todas_causas_tipos.csv", delimiter=";", encoding="latin1")
    frota = pd.read_excel(
    "data/raw/E_Frota_por_UF_Municipio_POTENCIA_Dezembro_2024.xlsx", 
    skiprows=3,
    header=None,
    names=["uf", "municipio", "total_veiculos", "potencia"]
)
    return acidentes, frota
