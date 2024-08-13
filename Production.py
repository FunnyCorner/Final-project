from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Загрузка модели из файла
with open('optimized_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

def transform_fireplace(value):
    if isinstance(value, (int, float)) and value >= 1:
        return 1
    elif isinstance(value, str) and ('fireplace' in value.lower() or value.lower() == 'yes'):
        return 1
    return 0

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    data = request.get_json()

    # Преобразуем данные в DataFrame
    df = pd.DataFrame([data])

    # Применяем такие же преобразования, как и в процессе обучения
    df['fireplace'] = df['fireplace'].apply(transform_fireplace)
    df = pd.get_dummies(df, columns=['status', 'region'], drop_first=True)

    # Обеспечиваем, чтобы все нужные признаки присутствовали в правильном порядке
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Выполняем предсказание
    prediction = model.predict(df)
    
    # Возвращаем результат
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

