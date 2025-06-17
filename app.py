from src.data_collection import carregar_dados
from src.data_processing import processar_dados
from src.ml_pipeline import treinar_modelo
from src.dashboard import exibir_dashboard

acidentes, frota = carregar_dados()
df = processar_dados(acidentes, frota)
modelo = treinar_modelo(df)
exibir_dashboard(df, modelo)
