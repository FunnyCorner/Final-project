Цели проекта
Проект направлен на решение проблемы агентства недвижимости, которое сталкивается с высокой затратой времени на анализ предложений. Основная цель — разработать модель машинного обучения для прогнозирования стоимости недвижимости, что позволит агентству быстрее и эффективнее обрабатывать объявления, увеличивая количество сделок и прибыль.

Задачи проекта
Анализ и очистка данных:

Проведение разведывательного анализа данных.
Очистка данных от дубликатов, ошибок ввода и незначимых столбцов.
Преобразование текстовых данных в числовой формат.
Выделение значимых факторов:

Определение ключевых факторов, влияющих на стоимость недвижимости.
Моделирование:

Построение модели машинного обучения с использованием алгоритма Random Forest.
Подбор гиперпараметров модели для достижения наилучших результатов.
Разработка веб-сервиса:

Создание простого веб-сервиса для предсказания стоимости недвижимости на основе введённых данных.
Этапы работы над проектом
1. Подготовка данных
Загрузка данных из файла data.csv.
Удаление дублирующихся строк и ненужных столбцов.
Преобразование текстовых данных (например, площадь, количество ванных комнат и цена) в числовой формат.
2. Разведывательный анализ данных (EDA)
Анализ распределения признаков и выявление выбросов.
Обработка пропущенных значений для улучшения качества модели.
3. Моделирование
Разделение данных на обучающую и тестовую выборки.
Обучение модели Random Forest.
Оптимизация гиперпараметров модели с использованием GridSearchCV.
Оценка модели на тестовых данных с помощью метрики RMSE (корень среднеквадратичной ошибки).
4. Разработка веб-сервиса
Реализация веб-сервиса с использованием Flask для предсказания стоимости недвижимости на основе введённых пользователем данных.
Загрузка обученной модели из файла и использование её для предсказания.
Обработка запросов POST для получения данных и возврата предсказанной цены.
Полученные результаты
Точность модели: Оптимизированная модель Random Forest показала хорошую точность на тестовых данных, с RMSE, свидетельствующим о точности предсказаний модели.
Веб-сервис: Разработан веб-сервис, который позволяет пользователям вводить данные о недвижимости (например, площадь, количество комнат, этаж, год постройки, наличие парковки, район) и получать предсказание стоимости в реальном времени.



Описание кода
Файл Jupyter Notebook
Импорт библиотек:

Подключаются необходимые библиотеки для работы с данными (pandas, numpy), визуализации (matplotlib, seaborn), и машинного обучения (sklearn).
python
Копировать код
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
Загрузка и очистка данных:

Данные загружаются из файла data.csv. Удаляются дубликаты и ненужные столбцы. Преобразуются текстовые данные в числовой формат, используя регулярные выражения.
python
Копировать код
df = pd.read_csv("data.csv")
df = df.drop_duplicates()
df = df.drop(columns=['private pool', 'homeFacts', 'schools', 'mls-id', 'MlsId', 'street', 'zipcode', 'city'])
Подготовка признаков и целевой переменной:

Преобразуются столбцы, такие как baths, sqft, beds, и target в числовой формат для дальнейшего использования в модели.
python
Копировать код
df['baths'] = df['baths'].apply(lambda x: float(re.search(r'\d+\.?\d*', x).group()) if pd.notnull(x) else x)
df['sqft'] = df['sqft'].apply(lambda x: float(re.search(r'\d+\.?\d*', x).group().replace(',', '')) if pd.notnull(x) else x)
df['beds'] = df['beds'].apply(lambda x: float(re.search(r'\d+\.?\d*', x).group()) if pd.notnull(x) else x)
df['target'] = df['target'].apply(lambda x: float(x.replace('$', '').replace(',', '')) if pd.notnull(x) else x)
Разделение данных:

Данные делятся на обучающую и тестовую выборки для тренировки и оценки модели.
python
Копировать код
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Обучение модели и подбор гиперпараметров:

Используется GridSearchCV для подбора наилучших гиперпараметров модели Random Forest. Обучение проводится на обучающей выборке, а затем выбирается наилучшая модель.
python
Копировать код
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
Оценка модели:

Модель оценивается на тестовой выборке с использованием метрики RMSE.
python
Копировать код
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse}")
Файл Production.py
Импорт необходимых библиотек:

Flask используется для создания веб-сервиса. Модель загружается с помощью pickle.
python
Копировать код
from flask import Flask, request, jsonify
import pickle
import numpy as np
Загрузка модели:

Модель загружается из файла optimized_rf_model.pkl.
python
Копировать код
with open('optimized_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
Создание Flask-приложения:

Создается простое Flask-приложение с одним маршрутом для обработки POST-запросов, где пользователи могут отправить данные для предсказания стоимости недвижимости.
python
Копировать код
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['square_meters'], data['rooms'], data['floor'], 
                         data['year_built'], data['has_parking'], data['district']])
    prediction = model.predict([features])
    return jsonify({'predicted_price': prediction[0]})
Запуск сервера:

Flask-приложение запускается в режиме отладки.
python
Копировать код
if __name__ == '__main__':
    app.run(debug=True)

    #Параметры теста веб сервиса через postman:{
    "square_meters": 2000,
    "rooms": 3,
    "baths": 2,
    "fireplace": 1,
    "has_parking": 1,
    "stories": 2,
    "status": "for sale",
    "region": "some_region"
}
