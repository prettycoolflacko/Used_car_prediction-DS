import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Agar tidak error saat generate plot di server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.le_make = LabelEncoder()
        self.le_state = LabelEncoder()
        self.unique_makes = []
        self.unique_states = []
        
        # Pastikan folder static ada untuk menyimpan gambar
        if not os.path.exists('static'):
            os.makedirs('static')

    def load_and_train(self, data_path='true_car_listings.csv'):
        print("--- Memulai Proses Training ---")
        df = pd.read_csv(data_path)
        
        # 1. Cleaning
        df = df.drop_duplicates().dropna()
        problematic = (df['Model'].str.contains('\*', na=False, regex=False) | 
                       (df['Model'].str.len() <= 2) | 
                       df['Model'].str.match(r'^\d+$', na=False))
        df = df[~problematic]

        # 2. Feature Engineering
        df['Car_Age'] = 2025 - df['Year']
        df['Make_Encoded'] = self.le_make.fit_transform(df['Make'])
        df['State_Encoded'] = self.le_state.fit_transform(df['State'])
        
        self.unique_makes = sorted(df['Make'].unique().tolist())
        self.unique_states = sorted(df['State'].unique().tolist())

        # 3. Training
        X = df[['Car_Age', 'Mileage', 'Make_Encoded', 'State_Encoded']]
        y = df['Price']
        
        # Tuned parameters
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        print("--- Model Selesai Dilatih ---")
        
        # 4. Generate EDA Images untuk UI
        self._generate_plots(df, self.model)

    def _generate_plots(self, df, model):
        # Plot 1: Feature Importance
        plt.figure(figsize=(10, 6))
        features = ['Car_Age', 'Mileage', 'Make', 'State']
        importances = model.feature_importances_
        sns.barplot(x=importances, y=features, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')
        plt.close()

        # Plot 2: Scatter (Sampled)
        sample = df.sample(min(2000, len(df)), random_state=42)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Car_Age', y='Price', data=sample, alpha=0.5, color='blue')
        plt.title('Usia vs Harga')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Mileage', y='Price', data=sample, alpha=0.5, color='green')
        plt.title('Jarak Tempuh vs Harga')
        plt.tight_layout()
        plt.savefig('static/eda_scatters.png')
        plt.close()

    def predict(self, year, mileage, make, state):
        if not self.model: return None
        try:
            car_age = 2025 - int(year)
            
            # Handle unknown labels gracefully
            make_enc = self.le_make.transform([make])[0] if make in self.le_make.classes_ else self.le_make.transform([self.le_make.classes_[0]])[0]
            state_enc = self.le_state.transform([state])[0] if state in self.le_state.classes_ else self.le_state.transform([self.le_state.classes_[0]])[0]
            
            features = np.array([[car_age, float(mileage), make_enc, state_enc]])
            prediction = self.model.predict(features)[0]
            return round(prediction, 2)
        except Exception as e:
            print(f"Error: {e}")
            return 0