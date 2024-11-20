import streamlit as st
import pandas as pd
import datetime
import predict

st.header('Inserir dados da transação')

columns = pd.read_csv('data/columns.csv', index_col=0)

transaction = {}

for column in columns:
    if column == 'Time':
        time = st.time_input("Horário da transação", datetime.time(12, 30))
        transaction[column] = time.hour * 3600 + time.minute * 60
    elif column == 'Amount':
        transaction[column] = st.number_input("Valor da transação", step=100.0, min_value=columns.loc['Min', column], max_value=columns.loc['Max', column], value=columns.loc['Average', column])
    else:
        transaction[column] = st.slider(column, step=columns.loc['Max', column]/100, min_value=columns.loc['Min', column], max_value=columns.loc['Max', column], value=columns.loc['Average', column])

submit = st.button('Avaliar fraude')

if submit:
    predictions = predict.predict(transaction)

    for prediction in predictions.keys():
        st.subheader(prediction)
        if predictions[prediction] == 1:
            st.text('Fraude')
        else:
            st.text('Não é fraude')