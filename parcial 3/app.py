from flask import Flask, render_template, request
import numpy as np
import joblib, os

app = Flask(__name__)

# ───────────── Carga del modelo ─────────────
MODEL_PATH = os.path.join("models", "best_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo no encontrado en {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURES = [
    'Humidity3pm',      # 0
    'Rainfall',         # 1
    'Pressure3pm',      # 2
    'MaxTemp',          # 3
    'RainToday_Yes'     # 4
]

# ───────────── Explicación avanzada ─────────
def build_explanation(pred_label: str,
                      vals: np.ndarray,
                      imps: np.ndarray,
                      proba: float | None = None,
                      top_k: int = 3) -> str:
    """
    Genera un texto explicativo con detalles meteorológicos
    y, si se dispone, muestra la probabilidad estimada.
    """
    # --------- Interpretaciones simples por variable ---------
    heur = {
        'Humidity3pm': lambda v: (
            "una humedad EXCESIVA"       if v >= 80 else
            "una humedad elevada"        if v >= 65 else
            "una humedad moderada"           if v >= 50 else
            "una humedad baja"
        ),
        'Pressure3pm': lambda v: (
            "una presión MUY baja"       if v < 1000 else
            "una presión ligeramente baja" if v < 1010 else
            "una presión en rango medio" if v < 1020 else
            "una presión ALTA"
        ),
        'MaxTemp': lambda v: (
            "temperaturas MUY altas"     if v >= 30 else
            "temperaturas cálidas"       if v >= 23 else
            "temperaturas templadas"     if v >= 17 else
            "temperaturas frescas"
        ),
        'Rainfall': lambda v: (
            "lluvia acumulada significativa" if v >= 5 else
            "muy poca lluvia previa"
        ),
        'RainToday_Yes': lambda v: (
            "que hoy ya llovió**" if v == 1 else "que hoy no llovió"
        )
    }

    # Seleccionar las variables más influyentes
    idx_sorted = np.argsort(imps)[::-1][:top_k]
    detalles = []
    for idx in idx_sorted:
        feat, val = FEATURES[idx], vals[idx]
        frase = heur.get(feat, lambda x: f"{feat} = {x}")(val)

        # Valor numérico formateado
        if feat == 'Humidity3pm':
            frase += f" ({val:.0f} %)"
        elif feat == 'Pressure3pm':
            frase += f" ({val:.0f} hPa)"
        elif feat == 'MaxTemp':
            frase += f" ({val:.1f} °C)"
        elif feat == 'Rainfall':
            frase += f" ({val:.1f} mm)"
        detalles.append(frase)

    razones = ", ".join(detalles[:-1]) + " y " + detalles[-1] if len(detalles) > 1 else detalles[0]

    # Probabilidad (si existe)
    prob_txt = f"Probabilidad de lluvia mañana: {proba*100:.0f} %\n\n" if proba is not None else ""

    if pred_label == "Sí":
        intro  = "El modelo considera que mañana es muy probable que llueva. "
        cierre = "En síntesis, el exceso de vapor de agua supera cualquier factor de estabilidad."
    else:
        intro  = "El modelo estima que mañana es poco probable que llueva. "
        cierre = "Estas variables favorecen un ambiente estable con escasa formación de nubosidad."

    return f"{prob_txt}{intro}{razones}. {cierre}"


# ───────────── Rutas Flask ─────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rainfall       = float(request.form['var1'])
        humidity3pm    = float(request.form['var2'])
        pressure3pm    = float(request.form['var3'])
        maxtemp        = float(request.form['var4'])
        rain_today_yes = float(request.form['var5'])

        datos = np.array([humidity3pm, rainfall, pressure3pm, maxtemp, rain_today_yes])

        pred        = model.predict(datos.reshape(1, -1))[0]
        etiqueta    = "Sí" if pred == 1 else "No"
        importancia = model.feature_importances_

        texto_just  = build_explanation(etiqueta, datos, importancia)

        filas = "".join(
            f"<tr><td>{n}</td><td>{v}</td><td>{imp:.3f}</td><td>{imp*100:.1f}%</td></tr>"
            for n, v, imp in zip(FEATURES, datos, importancia)
        )

        tabla_html = f"""
        <table class='explicacion-tabla'>
            <thead><tr>
                <th>Variable</th><th>Valor</th>
                <th>Importancia</th><th>Importancia (%)</th>
            </tr></thead>
            <tbody>{filas}</tbody>
        </table>"""

        fondo = "lluvioso" if pred == 1 else "soleado"

        return render_template('index.html',
                               cuadro_texto=texto_just,
                               tabla_html=tabla_html,
                               fondo=fondo)

    except Exception as e:
        return render_template('index.html',
                               cuadro_texto=f"⚠️ Error: {e}",
                               tabla_html="",
                               fondo="")

if __name__ == '__main__':
    app.run(debug=True)
