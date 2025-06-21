import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, json
import os

# 1. Cargar dataset
df = pd.read_csv("weatherAUS.csv")

# 2. Selección de columnas y limpieza
df = df[['Humidity3pm', 'Rainfall', 'Pressure3pm', 'MaxTemp', 'RainToday', 'RainTomorrow']].dropna()

# 3. Conversión de variables categóricas a numéricas
df['RainToday_Yes'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
df['RainTomorrow'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# 4. Selección final de variables predictoras
selected_features = ['Humidity3pm', 'Rainfall', 'Pressure3pm', 'MaxTemp', 'RainToday_Yes']
X = df[selected_features]
y = df['RainTomorrow']

# 5. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenar modelo balanceado
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 7. Evaluar desempeño
print("📊 Reporte de clasificación:")
print(classification_report(y_test, model.predict(X_test)))

# 8. Crear carpeta si no existe
os.makedirs("models", exist_ok=True)

# 9. Guardar modelo
joblib.dump(model, "models/best_model.pkl")
print("✅ Modelo guardado: models/best_model.pkl")

# 10. Guardar orden de variables
with open("models/feature_list.json", "w") as f:
    json.dump(selected_features, f)
print("✅ Lista de variables guardada: models/feature_list.json")
