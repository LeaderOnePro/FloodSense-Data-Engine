"""Crawler modules for web scraping."""

from floodsense.crawlers.base import BaseCrawler
from floodsense.crawlers.image_spider import ImageSpider
from floodsense.crawlers.video_crawler import VideoCrawler

# API-based crawlers
from floodsense.crawlers.api_crawlers import (
    BaseAPICrawler,
    RateLimiter,
    UnsplashCrawler,
    PexelsCrawler,
    FlickrCrawler,
    WikimediaCrawler,
    MultiSourceCrawler,
)

# Satellite crawlers
from floodsense.crawlers.satellite_crawlers import NASACrawler

__all__ = [
    # Base crawlers
    "BaseCrawler",
    "ImageSpider",
    "VideoCrawler",
    # API crawlers
    "BaseAPICrawler",
    "RateLimiter",
    "UnsplashCrawler",
    "PexelsCrawler",
    "FlickrCrawler",
    "WikimediaCrawler",
    "MultiSourceCrawler",
    # Satellite crawlers
    "NASACrawler",
]
