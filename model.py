import os
# SOLUSI ERROR WINDOWS: Set jumlah core manual (misal 4) agar ringan & tidak error
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

class CarPricePredictor:
    def __init__(self):
        self.models = {} 
        self.le_make = LabelEncoder()
        self.le_state = LabelEncoder()
        self.unique_makes = []
        self.unique_states = []
        
        # Batas data (Validation)
        self.limits = {
            'year_min': 1997, 'year_max': 2018,
            'mileage_min': 5, 'mileage_max': 2856196,
            'price_min': 1500, 'price_max': 499500
        }
        
        if not os.path.exists('static'):
            os.makedirs('static')

    def load_and_train(self, data_path='true_car_listings.csv'):
        print("--- Memulai Loading Data ---")
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("File CSV tidak ditemukan.")
            return

        # 1. Cleaning & Filtering
        df = df.drop_duplicates().dropna()
        df = df[
            (df['Year'] >= self.limits['year_min']) & 
            (df['Year'] <= self.limits['year_max']) &
            (df['Mileage'] >= self.limits['mileage_min']) & 
            (df['Mileage'] <= self.limits['mileage_max']) &
            (df['Price'] >= self.limits['price_min']) & 
            (df['Price'] <= self.limits['price_max'])
        ]

        problematic = (df['Model'].str.contains('\*', na=False, regex=False) | 
                       (df['Model'].str.len() <= 2) | 
                       df['Model'].str.match(r'^\d+$', na=False))
        df = df[~problematic]

        print(f"Data Training: {len(df)} baris.")

        # 2. Feature Engineering
        df['Car_Age'] = 2025 - df['Year']
        df['Make_Encoded'] = self.le_make.fit_transform(df['Make'])
        df['State_Encoded'] = self.le_state.fit_transform(df['State'])
        
        self.unique_makes = sorted(df['Make'].unique().tolist())
        self.unique_states = sorted(df['State'].unique().tolist())

        X = df[['Car_Age', 'Mileage', 'Make_Encoded', 'State_Encoded']]
        y = df['Price']
        
        # --- TRAINING 4 MODEL (Diurutkan dari yang tercepat) ---
        
        # 1. Ridge Regression (Linear - Sangat Cepat)
        print("[1/4] Training Ridge Regression (Baseline)...")
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X, y)

        # 2. Decision Tree (Cepat, Single Tree)
        print("[2/4] Training Decision Tree (Simple)...")
        self.models['dt'] = DecisionTreeRegressor(max_depth=15, random_state=42)
        self.models['dt'].fit(X, y)

        # 3. HistGradientBoosting (Modern, Efisien untuk Data Besar)
        print("[3/4] Training HistGradientBoosting (Recommended)...")
        self.models['hgb'] = HistGradientBoostingRegressor(
            max_iter=150, max_depth=None, learning_rate=0.1, random_state=42, early_stopping=True
        )
        self.models['hgb'].fit(X, y)

        # 4. Random Forest (Berat, dikurangi estimators jadi 50 agar laptop aman)
        print("[4/4] Training Random Forest (Heavy)...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=50, # Dikurangi dari 100 agar hemat RAM/CPU
            max_depth=10, 
            random_state=42, 
            n_jobs=4 # Menggunakan 4 thread
        )
        self.models['rf'].fit(X, y)
        
        print("--- Semua Model Selesai Dilatih ---")
        self._generate_plots(df, self.models['rf'])

    def _generate_plots(self, df, rf_model):
        # Plot Feature Importance (Ambil dari RF)
        plt.figure(figsize=(10, 6))
        features = ['Car_Age', 'Mileage', 'Make', 'State']
        importances = rf_model.feature_importances_
        sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')
        plt.close()

        # Plot Scatter
        sample = df.sample(min(2000, len(df)), random_state=42)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Car_Age', y='Price', data=sample, alpha=0.5, color='#4F46E5')
        plt.title('Usia vs Harga')
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Mileage', y='Price', data=sample, alpha=0.5, color='#10B981')
        plt.title('Jarak Tempuh vs Harga')
        plt.tight_layout()
        plt.savefig('static/eda_scatters.png')
        plt.close()

    def predict(self, year, mileage, make, state, model_type='hgb'):
        model = self.models.get(model_type, self.models.get('hgb'))
        
        try:
            year = int(year)
            mileage = float(mileage)
            
            # Clip
            year = max(self.limits['year_min'], min(year, self.limits['year_max']))
            car_age = 2025 - year
            
            # Encode
            make_enc = self.le_make.transform([make])[0] if make in self.le_make.classes_ else self.le_make.transform([self.le_make.classes_[0]])[0]
            state_enc = self.le_state.transform([state])[0] if state in self.le_state.classes_ else self.le_state.transform([self.le_state.classes_[0]])[0]
            
            prediction = model.predict([[car_age, mileage, make_enc, state_enc]])[0]
            return max(self.limits['price_min'], min(prediction, self.limits['price_max']))
            
        except Exception as e:
            print(f"Error: {e}")
            return 0