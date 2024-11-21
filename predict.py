import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

def predict(transaction):
    # Carrega os modelos de predição treinados
    model_names = ['KNearest', 'Support Vector Classifier', 'DecisionTreeClassifier', 'Random Forest Classifier', 'LogisiticRegression']

    models = {}

    for name in model_names:
        models[name] = (pickle.load(open('./models/' + name + '.pkl', 'rb')))
    
    values = pd.DataFrame([transaction])
    print(values)

    with open('./scaler/scaler.pkl', 'rb') as scaler_file:
        rob_scaler = pickle.load(scaler_file)

    values['scaled_amount'] = rob_scaler.fit_transform(values['Amount'].values.reshape(-1,1))
    values['scaled_time'] = rob_scaler.fit_transform(values['Time'].values.reshape(-1,1))
    values.drop(['Time','Amount'], axis=1, inplace=True)
    
    predictions = {}

    for model in models.keys():
        results = models[model].predict(values)
        print(results)
        
        if len(results) == 1:
            predictions[model] = results[0]
        
    return predictions