from flask import Flask, render_template, request, jsonify
from model import CarPricePredictor

app = Flask(__name__)

# Inisialisasi Model saat aplikasi start
predictor = CarPricePredictor()
try:
    # Pastikan file csv ada di folder yang sama
    predictor.load_and_train('true_car_listings.csv')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/options', methods=['GET'])
def get_options():
    """API untuk mengirim daftar Merek dan State ke Dropdown"""
    return jsonify({
        'makes': predictor.unique_makes,
        'states': predictor.unique_states
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API untuk memproses prediksi harga"""
    data = request.json
    price = predictor.predict(
        data.get('year'),
        data.get('mileage'),
        data.get('make'),
        data.get('state')
    )
    return jsonify({
        'price': price,
        'formatted': f"${price:,.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)