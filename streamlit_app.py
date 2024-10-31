import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# Função para carregar e processar os dados
def load_data(file):
    df = pd.read_excel(file)
    df['Created'] = pd.to_datetime(df['Created'], format='%d/%m/%Y')
    return df

# Função para fazer previsões
def make_forecast(df):
    status_count = df[df['Status'] == 'Fechado'].groupby(df['Created']).size().reset_index(name='Quantidade')
    status_count.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(status_count)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return model, forecast, status_count

# Configuração do Streamlit
st.title("Previsão de Chamados Fechados")

uploaded_file = st.file_uploader("Escolha um arquivo Excel", type=["xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Dados Carregados:")
    st.dataframe(df.head())

    model, forecast, status_count = make_forecast(df)

    # Renomear colunas
    forecast = forecast.rename(columns={
        'ds': 'Data',
        'yhat': 'Previsão (Chamados Fechados)',
        'yhat_lower': 'Limite Inferior (Confiança)',
        'yhat_upper': 'Limite Superior (Confiança)'
    })

    st.write("Previsões:")
    st.dataframe(forecast[['Data', 'Previsão (Chamados Fechados)', 'Limite Inferior (Confiança)', 'Limite Superior (Confiança)']])
    
    # Gráfico de Barras dos Chamados Fechados
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(status_count['ds'], status_count['y'], color='blue')
    ax_bar.set_xlabel('Data')
    ax_bar.set_ylabel('Quantidade de Chamados Fechados')
    ax_bar.set_title('Chamados Fechados por Data')
    plt.xticks(rotation=45)
    st.pyplot(fig_bar)

    # Gráfico das Previsões
    fig_forecast, ax_forecast = plt.subplots()  
    model.plot(forecast, ax=ax_forecast)
    plt.title("Previsões de Chamados Fechados por Data")
    st.pyplot(fig_forecast)

    # Download do relatório
    output = BytesIO()
    forecast.to_excel(output, index=False)
    output.seek(0)
    st.download_button("Baixar Relatório de Previsão", output, "relatorio_previsao.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
