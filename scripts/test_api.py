"""
Simple test script for the FastAPI server.

Usage:
    python scripts/test_api.py
"""

import requests
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000"
# Note: Run with: uvicorn app.api.main:app --reload

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_search(image_path: str):
    """Test search endpoint with an image."""
    print(f"Testing /search endpoint with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"topk": 5}
        response = requests.post(f"{API_URL}/search", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['topk']} similar artworks")
        for i, artwork in enumerate(result['results'][:3], 1):
            print(f"  {i}. {artwork.get('title', 'Unknown')} by {artwork.get('artist', 'Unknown')}")
            print(f"     Score: {artwork['similarity_score']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()
    return response.status_code == 200

def test_artwork(artwork_id: int):
    """Test artwork endpoint."""
    print(f"Testing /artwork/{artwork_id} endpoint...")
    response = requests.get(f"{API_URL}/artwork/{artwork_id}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        artwork = response.json()
        print(f"Title: {artwork.get('title', 'Unknown')}")
        print(f"Artist: {artwork.get('artist', 'Unknown')}")
    else:
        print(f"Error: {response.text}")
    print()
    return response.status_code == 200

if __name__ == "__main__":
    # Find a test image
    test_image = None
    for path in [ROOT / "data/images", ROOT / "data/samples"]:
        if path.exists():
            images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        print("Error: No test image found. Please provide an image path.")
        sys.exit(1)
    
    print(f"Using test image: {test_image}\n")
    
    # Run tests
    if not test_health():
        print("Health check failed. Is the server running?")
        sys.exit(1)
    
    test_search(str(test_image))
    test_artwork(0)  # Test first artwork

