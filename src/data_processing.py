import pandas as pd
import unicodedata

def processar_dados(acidentes, frota):
    # Conversão de datas com tratamento de erro
    acidentes["data_inversa"] = pd.to_datetime(acidentes["data_inversa"], dayfirst=True, errors="coerce")
    acidentes = acidentes.dropna(subset=["data_inversa"])

    # Conversão da hora
    acidentes["hora"] = pd.to_datetime(acidentes["horario"], errors='coerce').dt.hour

    # Normalizar coluna 'municipio' da base de acidentes
    acidentes["municipio"] = acidentes["municipio"].str.upper()
    acidentes["municipio"] = acidentes["municipio"].apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ascii", errors="ignore").decode("utf-8")
    )

    # Garantir nomes corretos da base frota
    frota.columns = ["uf", "municipio", "total_veiculos", "potencia"]

    # Normalizar a coluna 'municipio' da frota
    frota["municipio"] = frota["municipio"].str.upper()
    frota["municipio"] = frota["municipio"].apply(
        lambda x: unicodedata.normalize("NFKD", x).encode("ascii", errors="ignore").decode("utf-8")
    )

    # 🔥 EVITAR REPLICAÇÃO CARTESIANA
    frota = frota.drop_duplicates(subset="municipio")

    # Faz o merge seguro
    df = acidentes.merge(frota, on="municipio", how="left")

    return df
