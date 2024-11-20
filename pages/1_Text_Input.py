import streamlit as st
import pandas as pd
import predict

st.header('Inserir dados da transação')

columns = pd.read_csv('data/columns.csv', index_col=0)
text = st.text_input('Inserir valores da transação, separados por vírgula')
submit = st.button('Avaliar fraude')

if submit:
    transaction = {}
    values = text.split(',')
    print(len(values), columns.shape[1])
    if len(values) == columns.shape[1]:
        for i, col_name in enumerate(columns.columns):  # Iterate over column names
            transaction[col_name] = values[i]
        predictions = predict.predict(transaction)
    
        for prediction in predictions.keys():
            st.subheader(prediction)
            if predictions[prediction] == 1:
                st.text('Fraude')
            else:
                st.text('Não é fraude')
    else:
        st.text('Entrada inválida')
    