import pandas as pd
import joblib
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

class CropRecommendationModel:
    def __init__(self, dataset_dir="dataset/"):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained = False
        self.dataset_dir = dataset_dir  # Directory where CSV files are stored
        self.weather_api_key = "654231e8f6affa69988bab426f8fd7da"

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df["label"] = self.label_encoder.fit_transform(df["label"])
        X = df.drop("label", axis=1)
        y = df["label"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, file_path):
        X_train, X_test, y_train, y_test = self.load_data(file_path)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.model.fit(X_train, y_train)
        self.trained = True
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"Model Accuracy: {accuracy:.2f}%")
        
        return accuracy

    def save_model(self, model_path="model/crop_model.pkl", scaler_path="model/scaler.pkl", encoder_path="model/label_encoder.pkl"):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        print("Model, scaler, and label encoder saved successfully.")

    def load_model(self, model_path="model/crop_model.pkl", scaler_path="model/scaler.pkl", encoder_path="model/label_encoder.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.trained = True
        print("Model loaded successfully.")

    def get_soil_data(self, district, block=None):
        """Fetch soil quality data for the given district/block."""
        district_file = os.path.join(f"{self.dataset_dir}/region_soil/", "maharashtra.csv")
        
        if not os.path.exists(district_file):
            raise FileNotFoundError("Maharashtra data file not found.")

        df = pd.read_csv(district_file)
        
        if block:
            block_file = os.path.join(f"{self.dataset_dir}/region_soil/", f"{district.lower()}.csv")
            if os.path.exists(block_file):
                df = pd.read_csv(block_file)
                df = df[df["Block"] == block]
            else:
                print(f"Block data file for {district} not found. Searching in all districts.")
                df = df[df["Block"] == block]  # Search in maharashtra.csv
        else:
            df = df[df["District"] == district]

        if df.empty:
            raise ValueError("No matching district/block found in the dataset.")
        
        return df[["N_low", "N_mid", "N_high", "P_low", "P_mid", "P_high", "K_low", "K_mid", "K_high", "pH_low", "pH_mid", "pH_high"]].mean().tolist()
    
    def get_weather_data(self, district):
        """Fetch temperature, humidity, and rainfall data from OpenWeatherMap API."""
        url = f"https://api.openweathermap.org/data/2.5/weather?q={district},IN&appid={self.weather_api_key}&units=metric"
        response = requests.get(url)
        if response.status_code != 200:
            print("Warning: Could not fetch weather data.")
            return None
        data = response.json()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)
        return [temperature, humidity, rainfall]

    def predict(self, district, block=None):
        """Predict top 3 crop recommendations based on location (district/block)."""
        if not self.trained:
            raise Exception("Model is not trained or loaded.")

        try:
            soil_data = self.get_soil_data(district, block)
            logging.info(f"Soil data for {district}, {block}: {soil_data}")
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Error fetching soil data: {e}")
            return str(e)

        weather_data = self.get_weather_data(district)
        logging.info(f"Weather data for {district}: {weather_data}")
        if weather_data is None:
            logging.error("Weather data unavailable. Cannot proceed with prediction.")
            return "Weather data unavailable. Cannot proceed with prediction."
        
        input_data = soil_data + weather_data
        logging.info(f"Input data for prediction: {input_data}")
        input_data = self.scaler.transform([input_data])
        probabilities = self.model.predict_proba(input_data)[0]
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices] * 100

        logging.info(f"Top 3 crops: {top_3_crops} with confidences: {top_3_confidences}")
        return [(crop, round(conf, 2)) for crop, conf in zip(top_3_crops, top_3_confidences)]
    
    def get_available_districts(self):
        df = pd.read_csv(os.path.join(f"{self.dataset_dir}/region_soil/", "maharashtra.csv"))
        return df["District"].unique().tolist()

    def get_blocks_in_district(self, district):
        df = pd.read_csv(os.path.join(f"{self.dataset_dir}/region_soil/", "maharashtra.csv"))
        return df[df["District"] == district]["Block"].unique().tolist()

    def get_all_blocks(self):
        df = pd.read_csv(os.path.join(f"{self.dataset_dir}/region_soil/", "maharashtra.csv"))
        return df["Block"].unique().tolist()

    def get_crop_names(self):
        #crop_path = os.path.join(f"{self.dataset_dir}/region_soil/", "Crop_detection")
        #return [name for name in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, name))]
        return ['almond', 'banana', 'cardamom', 'Cherry', 'chilli', 'clove', 'coconut', 'Coffee-plant', 'cotton', 'Cucumber', 'Fox_nut(Makhana)', 'gram', 'jowar', 'jute', 'Lemon', 'maize', 'mustard-oil', 'Olive-tree', 'papaya', 'Pearl_millet(bajra)', 'pineapple', 'rice', 'soyabean', 'sugarcane', 'sunflower', 'tea', 'Tobacco-plant', 'tomato', 'vigna-radiati(Mung)', 'wheat']

# Example Usage
if __name__ == "__main__":
    crop_model = CropRecommendationModel()
    crop_model.train("dataset/Crop_recommendation.csv")
    crop_model.save_model()
    
    # Example Prediction
    result = crop_model.predict(district="Nagpur", block="SomeBlock")
    print("Top 3 Recommended Crops:", result)
    
    # Fetch available data
    print("Available Districts:", crop_model.get_available_districts())
    print("Blocks in Nagpur:", crop_model.get_blocks_in_district("Nagpur"))
    print("All Available Blocks:", crop_model.get_all_blocks())
    print("Available Crop Names:", crop_model.get_crop_names())