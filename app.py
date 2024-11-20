import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import pickle
import datetime
from sklearn.preprocessing import RobustScaler

# para rodar esse arquivo
# streamlit run app.py

# verifica se a senha de acesso está correta
if not util.check_password():
    st.stop()

# Carrega os modelos de predição treinados
model_names = ['KNearest', 'Support Vector Classifier', 'DecisionTreeClassifier', 'Random Forest Classifier', 'LogisiticRegression']

models = {}

for name in model_names:
     models[name] = (pickle.load(open('./models/' + name + '.pkl', 'rb')))
    
st.title('Identificador de fraudes em cartão de crédito')

st.header('Inserir dados da transação')

columns = pd.read_csv('data/columns.csv', index_col=0)

transaction = {}
for column in columns:
    if column == 'Time':
            time = st.time_input("Horário da transação", datetime.time(12, 30))
            transaction[column] = time.hour * 3600 + time.minute * 60
    elif column == 'Amount':
            transaction[column] = st.number_input("Valor da transação", step=100.0    , min_value=columns.loc['Min', column], max_value=columns.loc['Max', column], value=columns.loc['Average', column])
    else:
        transaction[column] = st.slider(column, step=columns.loc['Max', column]/100, min_value=columns.loc['Min', column], max_value=columns.loc['Max', column], value=columns.loc['Average', column])

submit = st.button('Avaliar fraude')

if submit:
    values = pd.DataFrame([transaction])
    print(values)

    rob_scaler = RobustScaler()

    values['scaled_amount'] = rob_scaler.fit_transform(values['Amount'].values.reshape(-1,1))
    values['scaled_time'] = rob_scaler.fit_transform(values['Time'].values.reshape(-1,1))
    values.drop(['Time','Amount'], axis=1, inplace=True)
    
    for model in models.keys():
        results = models[model].predict(values)
        print(results)
        
        if len(results) == 1:
            st.subheader(model)
            if results[0] == 1:
                st.text('Fraude')
            else:
                st.text('Não é Fraude')
