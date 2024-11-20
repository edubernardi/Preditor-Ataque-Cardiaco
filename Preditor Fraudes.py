import streamlit as st
import pandas as pd

# para rodar esse arquivo
# streamlit run '.\Preditor Fraudes.py'

st.title('Identificador de fraudes em cartão de crédito')

st.markdown('Uma aplicação que executa a predição de fraudes em transações de cartão de crédito\
            a partir da entrada do usuário que é avaliada por 5 modelos treinados préviamente.\
            \nÉ possível inserir os dados diretamente como texto, ou através de sliders da\
            interface web.')

st.page_link('pages/1_Text_Input.py', label="Entrada de texto")
st.page_link('pages/2_Slider_Input.py', label="Entrada por interface")

st.subheader('Informações do dataset')

st.markdown('É composto por 30 colunas, incluindo horário da transação, quantidade e 28 parâmetros\
            anonimizados e padronizados (cada variável possui média 0 para o dataset).\
            \nAinda assim, possuem pesos diferentes na predição, conforme o gráfico:')

st.image('./images/features.png')

st.subheader('Amostra do dataset')

dados = pd.read_csv('./data/dataset.csv')
st.dataframe(dados)