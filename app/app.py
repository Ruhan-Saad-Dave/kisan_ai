from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from app.recommendation import CropRecommendationModel
from app.detection import CropClassifier
import os
import shutil
from typing import Optional
import uuid

app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending suitable crops based on soil and climate conditions",
    version="1.0.0"
)

# Initialize the crop recommendation model
model = CropRecommendationModel()

# Initialize the crop classifier
crop_classifier = CropClassifier()

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

class CropDetectionResponse(BaseModel):
    detected_crop: str
    confidence: float

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

@app.post("/detect-crop/", response_model=CropDetectionResponse, tags=["Detection"])
async def detect_crop(file: UploadFile = File(...)):
    """
    Detect crop type from an uploaded image.
    
    Parameters:
    - file: Image file to analyze
    
    Returns:
    - detected_crop: The type of crop detected in the image
    - confidence: Confidence level of the prediction (0-100%)
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "/tmp/uploads"  # Changed to use /tmp which is usually writable
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        result = crop_classifier.predict_crop(temp_file_path)
        
        # Clean up
        os.remove(temp_file_path)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to make prediction. Model may not be trained."
            )
        
        return {
            "detected_crop": result["crop"],
            "confidence": result["confidence"]
        }
    
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting crop: {str(e)}"
        )

@app.get("/health/", tags=["Health"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy", 
        "model_loaded": model.trained,
        "classifier_loaded": crop_classifier.model is not None
    }

if __name__ == "__main__":
    # Configuration for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.app:app", host="0.0.0.0", port=port, reload=False)