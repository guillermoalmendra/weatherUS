from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo no encontrado: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rainfall = float(request.form['var1'])
        humidity3pm = float(request.form['var2'])
        pressure3pm = float(request.form['var3'])
        maxtemp = float(request.form['var4'])
        rain_today_yes = float(request.form['var5'])

        input_array = np.array([[humidity3pm, rainfall, pressure3pm, maxtemp, rain_today_yes]])
        prediction = model.predict(input_array)[0]
        resultado = "Sí" if prediction == 1 else "No"

        # Explicación en tabla
        importances = model.feature_importances_
        feature_names = ['Humidity3pm', 'Rainfall', 'Pressure3pm', 'MaxTemp', 'RainToday_Yes']
        rows = ""
        for name, value, imp in zip(feature_names, input_array[0], importances):
            rows += f"<tr><td>{name}</td><td>{value}</td><td>{round(imp, 3)}</td><td>{round(imp * 100, 1)}%</td></tr>"

        explicacion = f"""
        <table class="explicacion-tabla">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Valor ingresado</th>
                    <th>Importancia (decimal)</th>
                    <th>Importancia (%)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """

        
        fondo = "lluvioso" if prediction == 1 else "soleado"
        return render_template(
             'index.html',
             prediction_text=f"Lloverá mañana: {resultado}",
            explicacion_text=explicacion,
            fondo=fondo
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
