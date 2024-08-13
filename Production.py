from flask import Flask, request, jsonify
import pickle
import numpy as np


# Загрузка модели из файла
with open('optimized_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    data = request.get_json()

    # Преобразуем данные в нужный формат для модели
    features = np.array([data['square_meters'], data['rooms'], data['floor'], 
                         data['year_built'], data['has_parking'], data['district']])
    
    # Выполняем предсказание
    prediction = model.predict([features])
    
    # Возвращаем результат
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)