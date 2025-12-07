from flask import Flask, render_template, request, jsonify
from model import CarPricePredictor

app = Flask(__name__)

predictor = CarPricePredictor()
try:
    predictor.load_and_train('true_car_listings.csv')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(predictor.limits)

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        'makes': predictor.unique_makes,
        'states': predictor.unique_states
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model_type', 'hgb')
    
    # Mapping nama model untuk ditampilkan di UI
    model_names = {
        'ridge': 'Ridge Regression (Linear)',
        'dt': 'Decision Tree Regressor',
        'rf': 'Random Forest Regressor',
        'hgb': 'HistGradientBoosting (Best)'
    }

    price = predictor.predict(
        data.get('year'),
        data.get('mileage'),
        data.get('make'),
        data.get('state'),
        model_type
    )
    
    return jsonify({
        'price': round(price, 2),
        'formatted': f"${price:,.2f}",
        'model_used': model_names.get(model_type, 'Unknown Model')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)