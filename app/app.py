from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from model.recommendation import CropRecommendationModel
import os

app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending suitable crops based on soil and climate conditions",
    version="1.0.0"
)

# Initialize the crop recommendation model
model = CropRecommendationModel()

# Load the pre-trained model
try:
    model.load_model(
        model_path="model/crop_model.pkl",
        scaler_path="model/scaler.pkl",
        encoder_path="model/label_encoder.pkl"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model will be trained on first request")

class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropResponse(BaseModel):
    recommended_crop: str

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint that returns API information."""
    return {
        "message": "Welcome to the Crop Recommendation API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.post("/predict/", response_model=CropResponse, tags=["Prediction"])
async def predict_crop(soil_data: SoilData):
    """
    Predict the most suitable crop based on soil and climate conditions.
    
    Parameters:
    - N: Nitrogen content in soil (mg/kg)
    - P: Phosphorus content in soil (mg/kg)
    - K: Potassium content in soil (mg/kg)
    - temperature: Temperature in Celsius
    - humidity: Relative humidity in %
    - ph: pH value of the soil
    - rainfall: Rainfall in mm
    """
    try:
        # If model not loaded, train it
        if not model.trained:
            if os.path.exists("dataset/Crop_recommendation.csv"):
                model.train("dataset/Crop_recommendation.csv")
                model.save_model(
                    model_path="model/crop_model.pkl",
                    scaler_path="model/scaler.pkl",
                    encoder_path="model/label_encoder.pkl"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Model not trained and dataset not found"
                )
        
        # Convert input data to list
        input_data = [
            soil_data.N,
            soil_data.P,
            soil_data.K,
            soil_data.temperature,
            soil_data.humidity,
            soil_data.ph,
            soil_data.rainfall
        ]
        
        # Get prediction
        recommended_crop = model.predict(input_data)
        
        return {"recommended_crop": recommended_crop}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health/", tags=["Health"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy", 
        "model_loaded": model.trained
    }

if __name__ == "__main__":
    # Configuration for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)