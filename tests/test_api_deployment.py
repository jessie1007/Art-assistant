#!/usr/bin/env python3
"""
Test script for Art Assistant API deployment.

Tests:
- Health endpoint
- Image search endpoint
- Batch search endpoint
- Error handling
- Response times

Usage:
    # Test local deployment
    python tests/test_api_deployment.py http://localhost:8000

    # Test Render deployment
    python tests/test_api_deployment.py https://your-app.onrender.com

    # Test with custom image
    python tests/test_api_deployment.py http://localhost:8000 --image path/to/image.jpg

    # Skip batch test (faster)
    python tests/test_api_deployment.py http://localhost:8000 --skip-batch
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
import io

# Add project root to path (tests/ is one level down)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False


def test_docs(base_url: str) -> bool:
    """Test API docs endpoint."""
    print("🔍 Testing /docs endpoint...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API docs accessible")
            return True
        else:
            print(f"   ⚠️  API docs returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ⚠️  API docs error: {e}")
        return False


def test_search(base_url: str, image_path: Optional[str] = None) -> bool:
    """Test image search endpoint."""
    print("🔍 Testing /search endpoint...")
    
    # Get test image
    if image_path:
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"   ❌ Image not found: {image_path}")
            return False
        img = Image.open(img_path)
    else:
        # Try to find a sample image in data/images
        img_dir = ROOT / "data" / "images"
        if img_dir.exists():
            sample_images = list(img_dir.glob("*.jpg"))[:1]
            if sample_images:
                img = Image.open(sample_images[0])
                print(f"   📷 Using sample image: {sample_images[0].name}")
            else:
                print("   ⚠️  No sample images found, skipping search test")
                return False
        else:
            print("   ⚠️  No data/images directory, skipping search test")
            return False
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.convert("RGB").save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    try:
        start_time = time.time()
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(
            f"{base_url}/search",
            files=files,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"   ✅ Search successful ({elapsed:.2f}s)")
            print(f"   📊 Found {len(results)} similar artworks")
            if results:
                top = results[0]
                print(f"   🎨 Top match: {top.get('title', 'N/A')}")
                print(f"   📈 Similarity: {top.get('similarity', 0):.3f}")
            return True
        else:
            print(f"   ❌ Search failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Search error: {e}")
        return False


def test_batch_search(base_url: str, image_path: Optional[str] = None) -> bool:
    """Test batch search endpoint."""
    print("🔍 Testing /search/batch endpoint...")
    
    # Get test images (need at least 2)
    if image_path:
        img_path = Path(image_path)
        if not img_path.exists():
            print(f"   ⚠️  Image not found, skipping batch test")
            return False
        images = [Image.open(img_path), Image.open(img_path)]  # Use same image twice
    else:
        img_dir = ROOT / "data" / "images"
        if img_dir.exists():
            sample_images = list(img_dir.glob("*.jpg"))[:2]
            if len(sample_images) >= 2:
                images = [Image.open(p) for p in sample_images]
                print(f"   📷 Using {len(images)} sample images")
            else:
                print("   ⚠️  Need 2+ images for batch test, skipping")
                return False
        else:
            print("   ⚠️  No data/images directory, skipping batch test")
            return False
    
    # Convert to bytes
    files = []
    for i, img in enumerate(images):
        img_bytes = io.BytesIO()
        img.convert("RGB").save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        files.append(('files', (f'test{i}.jpg', img_bytes, 'image/jpeg')))
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/search/batch",
            files=files,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"   ✅ Batch search successful ({elapsed:.2f}s)")
            print(f"   📊 Processed {len(results)} images")
            return True
        else:
            print(f"   ❌ Batch search failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Batch search error: {e}")
        return False


def test_artwork_endpoint(base_url: str) -> bool:
    """Test artwork metadata endpoint."""
    print("🔍 Testing /artwork/{id} endpoint...")
    try:
        # Test with ID 0 (first artwork)
        response = requests.get(f"{base_url}/artwork/0", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Artwork endpoint works")
            print(f"   🎨 Title: {data.get('title', 'N/A')}")
            return True
        elif response.status_code == 404:
            print(f"   ⚠️  Artwork 0 not found (might be empty index)")
            return True  # Not a failure, just empty
        else:
            print(f"   ⚠️  Artwork endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ⚠️  Artwork endpoint error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Art Assistant API deployment")
    parser.add_argument(
        "base_url",
        help="Base URL of the API (e.g., http://localhost:8000 or https://your-app.onrender.com)"
    )
    parser.add_argument(
        "--image",
        help="Path to test image (optional, will use sample if not provided)"
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch search test"
    )
    args = parser.parse_args()
    
    base_url = args.base_url.rstrip('/')
    
    print(f"\n🧪 Testing Art Assistant API at {base_url}\n")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health(base_url)))
    results.append(("API Docs", test_docs(base_url)))
    results.append(("Image Search", test_search(base_url, args.image)))
    results.append(("Artwork Metadata", test_artwork_endpoint(base_url)))
    
    if not args.skip_batch:
        results.append(("Batch Search", test_batch_search(base_url, args.image)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print("=" * 60)
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

