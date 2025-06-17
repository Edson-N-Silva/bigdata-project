import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import unicodedata
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pathlib
import sys

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Acidentes de Trânsito",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SISTEMA DE AUTENTICAÇÃO ---

# Carrega as credenciais do arquivo de configuração
config_file = pathlib.Path(__file__).parent / "config.yaml"

# Validação do arquivo de configuração
try:
    with config_file.open('r') as file:
        config = yaml.load(file, Loader=SafeLoader)
        if 'credentials' not in config or 'usernames' not in config['credentials']:
            st.error("O arquivo 'config.yaml' é inválido ou está mal formatado. Verifique a estrutura.")
            st.stop()
except FileNotFoundError:
    st.error("Arquivo 'config.yaml' não encontrado. Por favor, crie-o conforme o guia.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao ler o arquivo 'config.yaml': {e}")
    st.stop()

# Cria o objeto de autenticação
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- TELA DE LOGIN (LÓGICA CORRIGIDA) ---
# Renderiza o formulário de login e atualiza o st.session_state
authenticator.login()

# --- VERIFICAÇÃO DO STATUS DE AUTENTICAÇÃO ---
if st.session_state.get("authentication_status"):
    
    # --- APLICAÇÃO PRINCIPAL (SÓ APARECE APÓS LOGIN) ---
    authenticator.logout(location='sidebar')
    st.sidebar.title(f'Bem-vindo(a) *{st.session_state["name"]}*')

    @st.cache_data
    def load_and_process_data():
        """Carrega e processa os dados de acidentes e frota."""
        try:
            df_acidentes = pd.read_csv("https://github.com/Edson-N-Silva/bigdata-project/raw/main/src/acidentes2025_todas_causas_tipos.csv", sep=';', encoding='latin1', on_bad_lines='skip')
        except FileNotFoundError:
            st.error("Arquivo 'acidentes2025_todas_causas_tipos.csv' não encontrado. Verifique o caminho do arquivo.")
            return None
        try:
            df_frota = pd.read_excel("https://github.com/BrendoGarcia/bigdata-project/raw/main/src/E_Frota_por_UF_Municipio_POTENCIA_Dezembro_2024.xlsx", skiprows=3, header=None)
            df_frota.columns = ['uf', 'municipio', 'total_veiculos', 'potencia']
        except FileNotFoundError:
            st.error("Arquivo 'E_Frota_por_UF_Municipio_POTENCIA_Dezembro_2024.xlsx' não encontrado.")
            return None
        except Exception as e:
            st.error(f"Ocorreu um erro ao ler o arquivo Excel da frota: {e}")
            return None

        def normalize_text(text):
            if isinstance(text, str):
                return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').upper()
            return text

        df_acidentes['municipio'] = df_acidentes['municipio'].apply(normalize_text)
        df_frota['municipio'] = df_frota['municipio'].apply(normalize_text)
        df_frota = df_frota.drop_duplicates(subset=['municipio'])
        df = pd.merge(df_acidentes, df_frota, on='municipio', how='left')

        df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
        df['horario'] = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce').dt.time
        df['hora'] = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce').dt.hour
        df['latitude'] = df['latitude'].str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        df['dia_semana'] = df['data_inversa'].dt.day_name()
        df['periodo_dia'] = df['hora'].apply(lambda x: 'Dia' if 6 <= x < 18 else 'Noite')
        return df

    @st.cache_resource
    def train_model(_df):
        """Treina o modelo de machine learning."""
        df_model = _df.copy()
        df_model['risco'] = ((df_model['mortos'] > 0) | (df_model['feridos_graves'] > 0)).astype(int)
        if df_model['risco'].nunique() < 2:
            st.warning("Não há diversidade de classes de risco para treinar o modelo de previsão.")
            return None, None
        features = ['hora', 'causa_acidente', 'tipo_acidente', 'condicao_metereologica']
        df_model = df_model[features + ['risco']].dropna()
        encoders = {}
        for col in features:
            if df_model[col].dtype == 'object':
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col])
                encoders[col] = le
        X = df_model[features]
        y = df_model['risco']
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        return model, encoders

    df = load_and_process_data()

    if df is not None:
        model, encoders = train_model(df)
        st.sidebar.markdown("---")
        st.sidebar.title("Navegação do Dashboard")
        page_options = ["Visão Geral", "Análise por Município"]
        if model is not None:
            page_options.append("Previsão de Risco")
        page = st.sidebar.radio("Selecione uma página", page_options)
        
        if page == "Visão Geral":
            st.title("🗺️ Visão Geral dos Acidentes")
            st.markdown("---")
            total_acidentes = len(df)
            total_mortes = int(df['mortos'].sum())
            total_feridos = int(df['feridos_leves'].sum() + df['feridos_graves'].sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Acidentes", f"{total_acidentes:,}")
            col2.metric("Total de Mortes", f"{total_mortes:,}")
            col3.metric("Total de Feridos", f"{total_feridos:,}")
            st.markdown("---")
            st.subheader("🗺️ Análise Geográfica dos Acidentes")
            map_type = st.selectbox("Selecione o tipo de visualização do mapa:", ("Pontos de Acidente", "Mapa de Calor (Densidade)"))
            if map_type == "Pontos de Acidente":
                fig_map = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="classificacao_acidente", hover_name="municipio", hover_data={"causa_acidente": True, "data_inversa": "|%d de %B de %Y"}, size_max=15, zoom=3.5, center=dict(lat=-14.2350, lon=-51.9253), height=600, mapbox_style="carto-positron", title="Distribuição de Pontos de Acidente no Brasil")
                fig_map.update_traces(marker=dict(size=5, opacity=0.7))
            else:
                fig_map = px.density_mapbox(df, lat="latitude", lon="longitude", radius=10, center=dict(lat=-14.2350, lon=-51.9253), zoom=3.5, height=600, mapbox_style="carto-positron", color_continuous_scale="YlOrRd", title="Concentração de Acidentes no Brasil")
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Acidentes por Dia da Semana")
                order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df_weekday = df['dia_semana'].value_counts().reindex(order)
                df_weekday.index = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                fig_weekday = px.bar(df_weekday, x=df_weekday.index, y=df_weekday.values, labels={'x': 'Dia da Semana', 'y': 'Número de Acidentes'}, color_discrete_sequence=["#33C4FF"])
                st.plotly_chart(fig_weekday, use_container_width=True)
            with col2:
                st.subheader("☀️ Acidentes por Período do Dia")
                df_periodo = df['periodo_dia'].value_counts()
                fig_periodo = px.pie(df_periodo, names=df_periodo.index, values=df_periodo.values, title="Distribuição Dia vs. Noite", color_discrete_sequence=["#FFC300", "#36454F"])
                st.plotly_chart(fig_periodo, use_container_width=True)

        elif page == "Análise por Município":
            st.title("🏙️ Análise por Município")
            st.markdown("---")
            municipio_list = sorted(df["municipio"].dropna().unique())
            municipio = st.selectbox("Selecione o Município", municipio_list)
            df_municipio = df[df['municipio'] == municipio]
            st.subheader(f"📍 Localização dos Acidentes em {municipio}")
            if not df_municipio.empty:
                map_center = dict(lat=df_municipio['latitude'].mean(), lon=df_municipio['longitude'].mean())
                fig_map_municipio = px.scatter_mapbox(df_municipio, lat="latitude", lon="longitude", hover_name="causa_acidente", hover_data={"data_inversa": "|%d de %B de %Y"}, color="classificacao_acidente", zoom=10, height=500, center=map_center, mapbox_style="carto-positron", title=f"Pontos de Acidente em {municipio}")
                st.plotly_chart(fig_map_municipio, use_container_width=True)
            else:
                st.warning("Não há dados de geolocalização disponíveis para os acidentes neste município.")
            st.markdown("---")
            st.subheader(f"📈 Evolução dos Acidentes em {municipio}")
            df_time = df_municipio.set_index('data_inversa').resample('M').size().reset_index(name='contagem')
            fig_time = px.line(df_time, x='data_inversa', y='contagem', labels={'data_inversa': 'Data', 'contagem': 'Número de Acidentes'}, color_discrete_sequence=["#FF5733"])
            st.plotly_chart(fig_time, use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top 5 Causas de Acidentes")
                df_causas = df_municipio['causa_acidente'].value_counts().nlargest(5)
                fig_causas = px.bar(df_causas, y=df_causas.index, x=df_causas.values, orientation='h', labels={'y': 'Causa do Acidente', 'x': 'Número de Acidentes'}, color_discrete_sequence=["#33C4FF"])
                fig_causas.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_causas, use_container_width=True)
            with col2:
                st.subheader("🛣️ Acidentes por Tipo de Pista")
                df_pista = df_municipio['tipo_pista'].value_counts()
                fig_pista = px.bar(df_pista, x=df_pista.index, y=df_pista.values, labels={'x': 'Tipo de Pista', 'y': 'Número de Acidentes'}, color_discrete_sequence=["#FFC300"])
                st.plotly_chart(fig_pista, use_container_width=True)

        elif page == "Previsão de Risco":
            st.title("🔮 Previsão de Risco de Acidente")
            st.markdown("---")
            st.write("Selecione as condições para prever o risco de um acidente grave (com morte ou ferido grave).")
            col1, col2 = st.columns(2)
            with col1:
                hora = st.slider("Hora do Dia", 0, 23, 12)
                causa_list = sorted(df["causa_acidente"].dropna().unique())
                causa = st.selectbox("Causa Provável", causa_list)
            with col2:
                tipo_list = sorted(df["tipo_acidente"].dropna().unique())
                tipo = st.selectbox("Tipo de Acidente", tipo_list)
                clima_list = sorted(df["condicao_metereologica"].dropna().unique())
                clima = st.selectbox("Condição Climática", clima_list)
            if st.button("Calcular Risco", use_container_width=True, type="primary"):
                input_data = pd.DataFrame([[hora, causa, tipo, clima]], columns=['hora', 'causa_acidente', 'tipo_acidente', 'condicao_metereologica'])
                for col in ['causa_acidente', 'tipo_acidente', 'condicao_metereologica']:
                     if col in encoders:
                        le = encoders[col]
                        input_data[col] = input_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        input_data[col] = le.transform(input_data[col])
                risco = model.predict_proba(input_data)[0][1]
                st.success(f"**Risco Previsto de Acidente Grave: {risco:.2%}**")
                st.progress(float(risco))
    else:
        st.error("Não foi possível carregar e processar os dados. O dashboard não pode ser exibido.")

elif st.session_state.get("authentication_status") is False:
    st.error('Usuário ou senha incorreta.')
elif st.session_state.get("authentication_status") is None:
    st.warning('Por favor, insira seu usuário e senha.')
