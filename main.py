import streamlit as st
import pandas as pd
import predict

# para rodar esse arquivo
# streamlit run '.\main.py'

st.title('Preditor de riscos de ataque cardíaco')

st.markdown('Executa a predição do risco de ataque cardíaco a partir de informações do paciente')

st.subheader('Treinar modelo')

train = st.button('Treinar')

if train:
    stats = predict.train()

    stats_df = pd.DataFrame(stats)
    st.table(stats_df)

st.subheader('Inserir dados')

patient = {}

patient['age'] = st.number_input('Idade', key="input_1", format="%0f")
patient['sex'] = 1 if st.selectbox("Gênero", ("Masculino", "Feminino")) == "Masculino" else 0
patient['cp'] = st.selectbox('Tipo de dor peitoral', ("Angina típica", "Angina atípica", "Dor não anginosa", "Assintomático"), key="input_3")
patient['trtbps'] = st.slider('Pressão sanguínea', min_value=100, max_value=200, value = 100, key="input_4")

patient['chol'] = st.slider('Colesterol', min_value=100, max_value=500, value = 250, key="input_5")
patient['fbs'] = 1 if st.selectbox('Taxa de açúcar no sangue maior do que 120 mg/dl', ('Verdadeiro', 'Falso'), key="input_6") == "Verdadeiro" else 0
patient['restecg'] = st.select_slider('Resultado eletrocardiograma', (0, 1, 2), key="input_7")

patient['thalachh'] = st.number_input('Máximo de batimentos por segundo', min_value=10, max_value=250, value = 100, key="input_8")
patient['exng'] = 1 if st.selectbox('Dor no peito induzida por exercício', ('Verdadeiro', 'Falso'), key="input_9") == "Verdadeiro" else 0
patient['oldpeak'] = st.slider('Alta histórica', min_value = 0.0, max_value = 10.0, step=0.1, key="input_10")

patient['slp'] = st.select_slider('Pressão sanguínea', (0, 1, 2), key="input_11")
patient['caa'] = st.select_slider('Elevação do segmento ST', (0, 1, 2, 3, 4), key="input_12")
patient['thall'] = st.select_slider('Talassemia', (0, 1, 2, 3), key="input_13")

submit = st.button('Avaliar paciente')

if submit:
    if patient['cp'] == "Angina típica":
        patient['cp'] = 0
    elif patient['cp'] == "Angina atípica":
        patient['cp'] = 1
    elif patient['cp'] == "Dor não anginosa":
        patient['cp'] = 2
    elif patient['cp'] == "Assintomático":
        patient['cp'] = 3

    predictions = predict.predict(patient)

    for prediction in predictions.keys():
        st.subheader(prediction)
        if predictions[prediction] == 1:
            st.text('Chance alta de ataque cardíaco')
        else:
            st.text('Chance baixa de ataque cardíaco')