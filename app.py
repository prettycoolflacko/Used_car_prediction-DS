from flask import Flask, render_template, request, jsonify
from model import CarPricePredictor

app = Flask(__name__)

# Inisialisasi
predictor = CarPricePredictor()
try:
    # Proses ini mungkin memakan waktu 10-20 detik karena melatih 2 model
    predictor.load_and_train('true_car_listings.csv')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Kirim batas min/max ke UI"""
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
    
    # Ambil tipe model dari request, default ke 'rf' jika kosong
    model_selected = data.get('model_type', 'rf') 
    
    price = predictor.predict(
        data.get('year'),
        data.get('mileage'),
        data.get('make'),
        data.get('state'),
        model_type=model_selected
    )
    
    return jsonify({
        'price': round(price, 2),
        'formatted': f"${price:,.2f}",
        'used_model': 'Random Forest' if model_selected == 'rf' else 'HistGradientBoosting'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)