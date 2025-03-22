import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class CropRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained = False

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df["label"] = self.label_encoder.fit_transform(df["label"])  # Encode labels
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
        print(f"Model Accuracy: {accuracy:.2f}")
        
        return accuracy

    def save_model(self, model_path="crop_model.pkl", scaler_path="scaler.pkl", encoder_path="label_encoder.pkl"):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        print("Model, scaler, and label encoder saved successfully.")

    def load_model(self, model_path="crop_model.pkl", scaler_path="scaler.pkl", encoder_path="label_encoder.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.trained = True
        print("Model loaded successfully.")

    def predict(self, input_data):
        if not self.trained:
            raise Exception("Model is not trained or loaded.")
        
        input_data = self.scaler.transform([input_data])  # Scale input
        prediction = self.model.predict(input_data)
        return self.label_encoder.inverse_transform(prediction)[0]  # Convert back to crop name

def main():
    crop_model = CropRecommendationModel()
    crop_model.train("dataset/Crop_recommendation.csv")
    crop_model.save_model(model_path="model/crop_model.pkl", 
                          scaler_path="model/scaler.pkl", 
                          encoder_path="model/label_encoder.pkl")

# Example Usage
if __name__ == "__main__":
    crop_model = CropRecommendationModel()
    crop_model.train("dataset/Crop_recommendation.csv")
    crop_model.save_model(model_path="model/crop_model.pkl", 
                          scaler_path="model/scaler.pkl", 
                          encoder_path="model/label_encoder.pkl")
