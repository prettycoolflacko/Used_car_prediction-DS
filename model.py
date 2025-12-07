import os
# SOLUSI ERROR WINDOWS: Set jumlah core manual agar tidak mencari 'wmic'
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Agar tidak error saat generate plot di server tanpa GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

class CarPricePredictor:
    def __init__(self):
        self.models = {} # Dictionary untuk menyimpan dua model
        self.le_make = LabelEncoder()
        self.le_state = LabelEncoder()
        self.unique_makes = []
        self.unique_states = []
        
        # Batas data untuk UI (Validation)
        self.limits = {
            'year_min': 1997, 'year_max': 2018,
            'mileage_min': 5, 'mileage_max': 2856196,
            'price_min': 1500, 'price_max': 499500
        }
        
        if not os.path.exists('static'):
            os.makedirs('static')

    def load_and_train(self, data_path='true_car_listings.csv'):
        print("--- Memulai Proses Loading Data ---")
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("File CSV tidak ditemukan.")
            return

        # 1. Strict Cleaning sesuai Range
        df = df.drop_duplicates().dropna()
        df = df[
            (df['Year'] >= self.limits['year_min']) & 
            (df['Year'] <= self.limits['year_max']) &
            (df['Mileage'] >= self.limits['mileage_min']) & 
            (df['Mileage'] <= self.limits['mileage_max']) &
            (df['Price'] >= self.limits['price_min']) & 
            (df['Price'] <= self.limits['price_max'])
        ]

        # Membersihkan nama model yang aneh
        problematic = (df['Model'].str.contains('\*', na=False, regex=False) | 
                       (df['Model'].str.len() <= 2) | 
                       df['Model'].str.match(r'^\d+$', na=False))
        df = df[~problematic]

        print(f"Data bersih: {len(df)} baris. Memulai Training...")

        # 2. Feature Engineering
        df['Car_Age'] = 2025 - df['Year']
        df['Make_Encoded'] = self.le_make.fit_transform(df['Make'])
        df['State_Encoded'] = self.le_state.fit_transform(df['State'])
        
        self.unique_makes = sorted(df['Make'].unique().tolist())
        self.unique_states = sorted(df['State'].unique().tolist())

        X = df[['Car_Age', 'Mileage', 'Make_Encoded', 'State_Encoded']]
        y = df['Price']
        
        # 3. Training Model A: Random Forest (Original)
        # n_jobs=-1 diganti jadi n_jobs=4 agar lebih aman di Windows
        print("-> Melatih Random Forest (Model A)...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=4)
        rf_model.fit(X, y)
        self.models['rf'] = rf_model
        
        # 4. Training Model B: HistGradientBoosting (Recommended)
        print("-> Melatih HistGradientBoosting (Model B)...")
        hgb_model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=None, learning_rate=0.1, random_state=42, early_stopping=True
        )
        hgb_model.fit(X, y)
        self.models['hgb'] = hgb_model
        
        print("--- Semua Model Selesai Dilatih ---")
        self._generate_plots(df, rf_model)

    def _generate_plots(self, df, rf_model):
        # Plot 1: Feature Importance (Khusus RF)
        plt.figure(figsize=(10, 6))
        features = ['Car_Age', 'Mileage', 'Make', 'State']
        importances = rf_model.feature_importances_
        
        # PERBAIKAN SEABORN WARNING DISINI (tambah hue dan legend=False)
        sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
        
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')
        plt.close()

        # Plot 2: Scatter (Sampled)
        sample = df.sample(min(2000, len(df)), random_state=42)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Car_Age', y='Price', data=sample, alpha=0.5, color='#4F46E5')
        plt.title('Usia Mobil vs Harga')
        plt.xlabel('Usia (Tahun)')
        plt.ylabel('Harga ($)')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Mileage', y='Price', data=sample, alpha=0.5, color='#10B981')
        plt.title('Jarak Tempuh vs Harga')
        plt.xlabel('Mileage')
        
        plt.tight_layout()
        plt.savefig('static/eda_scatters.png')
        plt.close()

    def predict(self, year, mileage, make, state, model_type='hgb'):
        # Pilih model berdasarkan input user
        model = self.models.get(model_type)
        if not model:
            print(f"Model {model_type} tidak ditemukan, menggunakan default (HGB).")
            model = self.models.get('hgb')
            
        try:
            year = int(year)
            mileage = float(mileage)
            
            # Clip input agar aman
            if year < self.limits['year_min']: year = self.limits['year_min']
            if year > self.limits['year_max']: year = self.limits['year_max']
            
            car_age = 2025 - year
            
            # Handle unknown labels
            make_enc = self.le_make.transform([make])[0] if make in self.le_make.classes_ else self.le_make.transform([self.le_make.classes_[0]])[0]
            state_enc = self.le_state.transform([state])[0] if state in self.le_state.classes_ else self.le_state.transform([self.le_state.classes_[0]])[0]
            
            features = np.array([[car_age, mileage, make_enc, state_enc]])
            prediction = model.predict(features)[0]
            
            return max(self.limits['price_min'], min(prediction, self.limits['price_max']))
            
        except Exception as e:
            print(f"Error Prediction: {e}")
            return 0