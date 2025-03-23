from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import uvicorn
from app.recommendation import CropRecommendationModel
from app.detection import CropClassifier
import os
import shutil
from typing import Optional, List 
import uuid
from app.prediction import CropPriceFinder

app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending suitable crops based on soil and climate conditions",
    version="1.0.0"
)

# Initialize the crop recommendation model
model = CropRecommendationModel()

# Initialize the crop classifier
crop_classifier = CropClassifier()

# Initialize the crop price finder
price_finder = CropPriceFinder("dataset/cleaned_crop_prices.csv")

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
    # Train the model if not loaded
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

class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define response models
class CropPrediction(BaseModel):
    crop: str
    confidence: float

class CropResponse(BaseModel):
    recommended_crops: List[CropPrediction]
    message: Optional[str] = None

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
async def predict_crop(district: str, block: Optional[str] = None):
    """
    Predict the top 3 suitable crops based on district and optional block location.
    
    Parameters:
    - district: District name (e.g., "Nagpur")
    - block: Optional block name within the district
    
    Returns:
    - List of top 3 recommended crops with confidence scores
    """
    try:
        # If model not loaded, try to load it
        if not model.trained:
            try:
                model.load_model(
                    model_path="model/crop_model.pkl",
                    scaler_path="model/scaler.pkl",
                    encoder_path="model/label_encoder.pkl"
                )
            except FileNotFoundError:
                # If models don't exist, train the model
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
        
        # Get prediction using district and block parameters
        results = model.predict(district=district, block=block)
        
        # Handle error messages returned as strings
        if isinstance(results, str):
            return {"recommended_crops": [], "message": results}
        
        # Format the response
        recommendations = [
            CropPrediction(crop=crop, confidence=confidence)
            for crop, confidence in results
        ]
        
        return {"recommended_crops": recommendations}
    
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
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create a file path in the temporary directory
            file_extension = Path(file.filename).suffix or ".jpg"
            temp_file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}{file_extension}")
            
            # Save the uploaded file to the temporary location
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Make prediction using the file path
            result = crop_classifier.predict(temp_file_path)
            
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to make prediction. Model may not be trained."
                )
            
            # Check if there was an error in the prediction
            if "error" in result and result["error"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error in prediction: {result['message']}"
                )
            
            return {
                "detected_crop": result["crop_name"],
                "confidence": result["confidence"]
            }
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error detecting crop: {str(e)}"
            )

@app.get("/districts/", tags=["Metadata"])
async def get_districts():
    """Returns a list of available districts."""
    try:
        districts = model.get_available_districts()
        return {"districts": districts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blocks/{district}/", tags=["Metadata"])
async def get_blocks(district: str):
    """Returns a list of blocks in the given district."""
    try:
        blocks = model.get_blocks_in_district(district)
        if not blocks:
            raise HTTPException(status_code=404, detail="No blocks found for this district")
        return {"district": district, "blocks": blocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blocks/", tags=["Metadata"])
async def get_all_blocks():
    """Returns a list of all available blocks."""
    try:
        blocks = model.get_all_blocks()
        return {"blocks": blocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crops/", tags=["Metadata"])
async def get_crop_names():
    """Returns a list of crop names available in the dataset for detection."""
    try:
        crops = crop_classifier.get_names()
        return {"crops": crops}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crop-price/", tags=["Price"])
async def get_crop_price(
    crop_name: str = Query(..., description="Name of the crop"),
    district: Optional[str] = Query(None, description="Optional district to narrow search"),
    market: Optional[str] = Query(None, description="Optional market to narrow search")
):
    """
    Retrieve min and max price for a given crop.
    
    Parameters:
    - crop_name: Name of the crop (e.g., "Brinjal")
    - district: Optional district to narrow search
    - market: Optional market to narrow search
    
    Returns:
    - Dictionary containing min and max price
    """
    try:
        result = price_finder.get_crop_price(crop_name, district, market)
        
        if "message" in result:
            raise HTTPException(status_code=404, detail=result["message"])
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving crop price: {str(e)}")

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