import requests
import json
import time
import sys
import os

# Base URL for the deployed API
BASE_URL = "https://evoprox-kisan-ai.hf.space"

def test_root_endpoint():
    """Test the root endpoint of the API."""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint test passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint test failed with error: {str(e)}")
        return False

def test_health_endpoint():
    """Test the health endpoint of the API."""
    try:
        response = requests.get(f"{BASE_URL}/health/")
        if response.status_code == 200:
            print("✅ Health endpoint test passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health endpoint test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health endpoint test failed with error: {str(e)}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data."""
    # Sample soil data for testing
    sample_data = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.5,
        "rainfall": 202.93
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/", 
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Prediction endpoint test passed")
            print(f"Recommended crop: {response.json()['recommended_crop']}")
        else:
            print(f"❌ Prediction endpoint test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Prediction endpoint test failed with error: {str(e)}")
        return False

def test_crop_detection_endpoint():
    """Test the crop detection endpoint with a sample image."""
    # Path to a sample image for testing
    sample_image_path = "dataset/Crop_detection/banana/image (1).jpeg"
    
    # Check if the test image exists
    if not os.path.exists(sample_image_path):
        print(f"❌ Crop detection test skipped: Test image not found at {sample_image_path}")
        print("Please place a sample crop image at this location to test the detection endpoint")
        return False
    
    try:
        with open(sample_image_path, "rb") as image_file:
            files = {"file": (os.path.basename(sample_image_path), image_file, "image/jpeg")}
            
            response = requests.post(
                f"{BASE_URL}/detect-crop/",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Crop detection endpoint test passed")
                print(f"Detected crop: {result['detected_crop']}")
                print(f"Confidence: {result['confidence']:.2f}%")
            else:
                print(f"❌ Crop detection endpoint test failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Crop detection endpoint test failed with error: {str(e)}")
        return False

def run_all_tests():
    """Run all API tests and return overall status."""
    print("🔍 Starting API tests for Kisan AI Crop Recommendation API")
    print(f"🌐 Testing API at: {BASE_URL}")
    print("-" * 50)
    
    # Wait for the API to be fully initialized
    print("⏳ Waiting for API to initialize...")
    time.sleep(5)
    
    # Run all tests
    root_test = test_root_endpoint()
    print("-" * 50)
    
    health_test = test_health_endpoint()
    print("-" * 50)
    
    prediction_test = test_prediction_endpoint()
    print("-" * 50)
    
    detection_test = test_crop_detection_endpoint()
    print("-" * 50)
    
    # Overall test results
    all_passed = root_test and health_test and prediction_test and detection_test
    if all_passed:
        print("🎉 All tests passed successfully!")
    else:
        print("❌ Some tests failed. Please check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    # Exit with appropriate status code
    sys.exit(0 if success else 1)