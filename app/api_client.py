"""
API Client for Art Assistant FastAPI backend.

Best practices:
- Environment variable for API URL (defaults to localhost)
- Error handling with retries
- Session state caching
- Type hints for better IDE support
"""

import os
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import requests
from PIL import Image
import io


class ArtAssistantAPIClient:
    """
    Client for Art Assistant FastAPI backend.
    
    Usage:
        client = ArtAssistantAPIClient(api_url="http://localhost:8000")
        results = client.search_image(image_file, topk=5)
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize API client.
        
        Args:
            api_url: Base URL of FastAPI server (defaults to env var or localhost:8000)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_url = api_url or os.getenv("ART_ASSISTANT_API_URL", "http://localhost:8000")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Remove trailing slash
        self.api_url = self.api_url.rstrip("/")
        
    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/search")
            **kwargs: Additional arguments for requests.request()
        
        Returns:
            Response object
        
        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.api_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise requests.RequestException("Max retries exceeded")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status dictionary with index_size and metadata_count
        """
        try:
            response = self._make_request("GET", "/health")
            return response.json()
        except requests.RequestException as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "index_size": 0,
                "metadata_count": 0
            }
    
    def search_image(
        self,
        image: Image.Image,
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar artworks by image.
        
        Args:
            image: PIL Image object
            topk: Number of similar artworks to return (1-50)
        
        Returns:
            Dictionary with query_image_size, topk, and results list
        
        Raises:
            requests.RequestException: If API request fails
            ValueError: If topk is out of range
        """
        if topk < 1 or topk > 50:
            raise ValueError("topk must be between 1 and 50")
        
        # Convert PIL Image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # Prepare file upload
        files = {
            "file": ("image.png", img_bytes, "image/png")
        }
        data = {
            "topk": topk
        }
        
        response = self._make_request(
            "POST",
            "/search",
            files=files,
            data=data
        )
        
        return response.json()
    
    def search_image_from_file(
        self,
        file_path: Path,
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar artworks from image file path.
        
        Args:
            file_path: Path to image file
            topk: Number of similar artworks to return
        
        Returns:
            Dictionary with query_image_size, topk, and results list
        """
        image = Image.open(file_path).convert("RGB")
        return self.search_image(image, topk=topk)
    
    def search_batch(
        self,
        images: List[Image.Image],
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        Batch search: Process multiple images in one request.
        
        Why use batch search?
        - More efficient: Single HTTP request instead of N requests
        - Better for bulk processing: Upload gallery, get all results at once
        - Reduced overhead: Less network round-trips
        - Atomic operation: All succeed or all fail (easier error handling)
        
        Args:
            images: List of PIL Image objects (max 10)
            topk: Number of similar artworks per image (1-50)
        
        Returns:
            Dictionary with total_images, successful count, and results list
        
        Raises:
            requests.RequestException: If API request fails
            ValueError: If too many images or topk out of range
        """
        if len(images) > 10:
            raise ValueError("Maximum 10 images per batch request")
        
        if topk < 1 or topk > 50:
            raise ValueError("topk must be between 1 and 50")
        
        # Convert all images to bytes
        # FastAPI expects multiple files with the same field name "files"
        files_list = []
        for idx, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            files_list.append(
                ("files", (f"image_{idx}.png", img_bytes, "image/png"))
            )
        
        data = {
            "topk": topk
        }
        
        response = self._make_request(
            "POST",
            "/search/batch",
            files=files_list,
            data=data
        )
        
        return response.json()
    
    def search_batch_from_files(
        self,
        file_paths: List[Path],
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        Batch search from file paths.
        
        Args:
            file_paths: List of image file paths (max 10)
            topk: Number of similar artworks per image
        
        Returns:
            Dictionary with total_images, successful count, and results list
        """
        images = [Image.open(path).convert("RGB") for path in file_paths]
        return self.search_batch(images, topk=topk)
    
    def get_artwork(self, artwork_id: int) -> Dict[str, Any]:
        """
        Get artwork metadata by ID.
        
        Args:
            artwork_id: Artwork index ID
        
        Returns:
            Artwork metadata dictionary
        
        Raises:
            requests.RequestException: If API request fails
        """
        response = self._make_request("GET", f"/artwork/{artwork_id}")
        return response.json()
    
    def is_available(self) -> bool:
        """
        Check if API is available and responding.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False

