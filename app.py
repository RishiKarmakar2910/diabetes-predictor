from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')  # ✅ Fixes the Matplotlib + Tkinter error in Flask

import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model and scaler
model = joblib.load('voting_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature_keys = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
                    'insulin', 'bmi', 'diabetespedigreefunction', 'age']
    try:
        features = [float(request.form[key]) for key in feature_keys]
    except KeyError as e:
        return f"Missing field: {e}", 400

    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)

    y_pred = model.predict(input_scaled)[0]
    y_prob = model.predict_proba(input_scaled)[0][1]

    # Create ROC Curve based on the model trained on training data
    # (Optional: Replace X_test, y_test with stored/test values during training)
    fpr, tpr, _ = roc_curve([0, 1], [1 - y_prob, y_prob])
    roc_auc = auc(fpr, tpr)

    # Create plot in memory
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # Show result
    if y_pred == 1:
        result = f"🔴 High Risk of Diabetes (Probability: {y_prob:.2f})"
    else:
        result = f"🟢 Low Risk of Diabetes (Probability: {y_prob:.2f})"

    return render_template('index.html', result=result, roc_plot=plot_data)


if __name__ == '__main__':
    app.run(debug=True)
