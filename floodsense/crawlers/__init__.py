"""Crawler modules for web scraping."""

from floodsense.crawlers.base import BaseCrawler
from floodsense.crawlers.image_spider import ImageSpider
from floodsense.crawlers.video_crawler import VideoCrawler

__all__ = ["BaseCrawler", "ImageSpider", "VideoCrawler"]
