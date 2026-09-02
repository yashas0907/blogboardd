from typing import Type, Optional

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from blogboard.config.settings import app_settings


class UnsplashCoverInput(BaseModel):
    """Input schema for the UnsplashCoverTool."""
    query: str = Field(description="Search phrase for a cover image (e.g. 'neural network abstract').")


class UnsplashCoverTool(BaseTool):
    """Fetches a high-quality, license-free cover image URL from Unsplash."""
    name: str = "unsplash_cover_image"
    description: str = (
        "Fetch a fitting cover image URL from Unsplash for a given topic. "
        "Returns the image URL and photographer credit."
    )
    args_schema: Type[BaseModel] = UnsplashCoverInput
    model_config = {"extra": "ignore"}

    def _run(self, query: str) -> str:
        api_key = app_settings.content.UNSPLASH_API_KEY
        if not api_key:
            return "Error: UNSPLASH_API_KEY is not configured."

        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {api_key}"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return f"No Unsplash results for query: '{query}'."

            photo = results[0]
            urls = photo.get("urls", {})
            user = photo.get("user", {})
            return (
                f"Image URL: {urls.get('regular', '')}\n"
                f"Photographer: {user.get('name', 'Unknown')} (@{user.get('username', 'unknown')})\n"
                f"Unsplash page: {photo.get('links', {}).get('html', '')}"
            )
        except requests.exceptions.RequestException as e:
            return f"Unsplash API request failed: {str(e)}"
        except Exception as e:
            return f"Unexpected error during Unsplash search: {str(e)}"


def fetch_cover_image(topic: str, domain: str) -> Optional[str]:
    """
    Best-effort synchronous helper used by the validator when saving articles.
    Returns a raw Unsplash 'regular' image URL or None on any failure.
    """
    tool = UnsplashCoverTool()
    try:
        result = tool._run(query=f"{topic} {domain} abstract technology")
        if result.startswith("Image URL: "):
            return result.splitlines()[0].replace("Image URL: ", "").strip()
    except Exception:
        pass
    return None
