"""
Multi-source crawler that aggregates results from multiple image sources.

Provides unified interface for querying multiple APIs with priority-based
source ordering, auto-failover, and URL deduplication.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import requests
from loguru import logger

from floodsense.crawlers.base import BaseCrawler
from floodsense.crawlers.api_crawlers.unsplash_crawler import UnsplashCrawler
from floodsense.crawlers.api_crawlers.pexels_crawler import PexelsCrawler
from floodsense.crawlers.api_crawlers.flickr_crawler import FlickrCrawler
from floodsense.crawlers.api_crawlers.wikimedia_crawler import WikimediaCrawler
from floodsense.crawlers.satellite_crawlers.nasa_crawler import NASACrawler
from floodsense.utils.config import Config, CrawlerConfig
from floodsense.utils.file_utils import FileUtils
from floodsense.utils.proxy import ProxyManager


class MultiSourceCrawler(BaseCrawler):
    """
    Aggregator crawler that combines results from multiple image sources.

    Features:
    - Priority-based source ordering
    - Configurable results distribution across sources
    - Auto-failover on source errors
    - URL deduplication across sources
    """

    # Source registry with crawler classes and their priorities
    SOURCE_REGISTRY: Dict[str, Type[BaseCrawler]] = {
        "unsplash": UnsplashCrawler,
        "pexels": PexelsCrawler,
        "nasa": NASACrawler,
        "wikimedia": WikimediaCrawler,
        "flickr": FlickrCrawler,
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        crawler_config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
        sources: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize MultiSourceCrawler.

        Args:
            config: Full application config (contains api_sources).
            crawler_config: Crawler-specific configuration.
            proxy_manager: Optional proxy manager.
            output_dir: Directory to save downloaded files.
            sources: List of source names to use (defaults to all enabled).
        """
        super().__init__(
            config=crawler_config or (config.crawler if config else None),
            proxy_manager=proxy_manager,
            output_dir=output_dir,
        )
        self.app_config = config
        self.enabled_sources: List[str] = []
        self.crawlers: Dict[str, BaseCrawler] = {}
        self.source_priorities: Dict[str, int] = {}

        self._initialize_sources(sources)

    def _initialize_sources(self, sources: Optional[List[str]] = None) -> None:
        """
        Initialize enabled source crawlers.

        Args:
            sources: Optional list of source names to enable.
        """
        if self.app_config and hasattr(self.app_config, "api_sources"):
            api_sources = self.app_config.api_sources
        else:
            api_sources = None

        for source_name, crawler_class in self.SOURCE_REGISTRY.items():
            # Skip if specific sources requested and this isn't one
            if sources and source_name not in sources:
                continue

            # Get source config if available
            source_config = None
            if api_sources:
                source_config = getattr(api_sources, source_name, None)

            # Check if source is enabled
            if source_config and not source_config.enabled:
                continue

            # Get API key if needed
            api_key = None
            if source_config:
                api_key = source_config.api_key
                priority = source_config.priority
            else:
                priority = 99  # Default low priority

            # Initialize crawler
            try:
                if source_name in ["nasa", "wikimedia"]:
                    # No API key needed
                    crawler = crawler_class(
                        config=self.config,
                        proxy_manager=self.proxy_manager,
                        output_dir=self.output_dir,
                    )
                else:
                    # API key required
                    if not api_key:
                        logger.debug(
                            f"Skipping {source_name}: no API key configured"
                        )
                        continue
                    crawler = crawler_class(
                        config=self.config,
                        proxy_manager=self.proxy_manager,
                        output_dir=self.output_dir,
                        api_key=api_key,
                    )

                self.crawlers[source_name] = crawler
                self.source_priorities[source_name] = priority
                self.enabled_sources.append(source_name)
                logger.info(f"Enabled source: {source_name} (priority: {priority})")

            except (TypeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to initialize {source_name}: {e}")

        # Sort enabled sources by priority
        self.enabled_sources.sort(key=lambda s: self.source_priorities.get(s, 99))
        logger.info(f"Enabled sources (by priority): {self.enabled_sources}")

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs from all enabled sources.

        Args:
            keyword: Search keyword.
            max_results: Maximum total URLs to return.

        Returns:
            List of deduplicated image URLs.
        """
        all_urls: List[str] = []
        seen_urls: Set[str] = set()

        if not self.enabled_sources:
            logger.warning("No sources enabled. Check API keys and configuration.")
            return []

        # Distribute results across sources
        results_per_source = self._distribute_results(max_results)

        logger.info(f"Searching {len(self.enabled_sources)} sources for: {keyword}")

        for source_name in self.enabled_sources:
            if len(all_urls) >= max_results:
                break

            crawler = self.crawlers.get(source_name)
            if not crawler:
                continue

            # Determine how many results to request from this source
            remaining = max_results - len(all_urls)
            source_limit = min(
                results_per_source.get(source_name, remaining),
                remaining,
            )

            if source_limit <= 0:
                continue

            try:
                logger.debug(
                    f"Requesting {source_limit} images from {source_name}"
                )
                urls = crawler.get_image_urls(keyword, source_limit)

                # Deduplicate
                for url in urls:
                    if url not in seen_urls and len(all_urls) < max_results:
                        seen_urls.add(url)
                        all_urls.append(url)

                logger.info(
                    f"Got {len(urls)} URLs from {source_name} "
                    f"({len(all_urls)} total so far)"
                )

            except (requests.exceptions.RequestException, ValueError, OSError) as e:
                logger.exception(f"Error fetching from {source_name}: {e}")
                # Auto-failover: continue to next source
                continue

        logger.info(
            f"Found {len(all_urls)} total images from {len(self.enabled_sources)} sources"
        )
        return all_urls

    def _distribute_results(self, max_results: int) -> Dict[str, int]:
        """
        Distribute max_results across enabled sources.

        Higher priority sources get more results.

        Args:
            max_results: Total maximum results.

        Returns:
            Dictionary mapping source names to result limits.
        """
        distribution: Dict[str, int] = {}

        if not self.enabled_sources:
            return distribution

        num_sources = len(self.enabled_sources)

        # Weighted distribution based on priority
        # Priority 1 gets more than priority 5
        weights: Dict[str, float] = {}
        total_weight = 0.0

        for source_name in self.enabled_sources:
            priority = self.source_priorities.get(source_name, 5)
            # Higher weight for lower priority number
            weight = 1.0 / (priority * 0.5 + 0.5)
            weights[source_name] = weight
            total_weight += weight

        # Distribute based on weights
        for source_name, weight in weights.items():
            proportion = weight / total_weight
            allocation = int(max_results * proportion)
            # Ensure at least some results from each source
            distribution[source_name] = max(allocation, max_results // num_sources)

        return distribution

    def crawl(self, keywords: List[str], max_results: int = 100) -> List[Path]:
        """
        Crawl images for given keywords from all sources.

        Args:
            keywords: List of search keywords.
            max_results: Maximum images per keyword.

        Returns:
            List of paths to downloaded files.
        """
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"[MultiSource] Crawling for: {keyword}")
            urls = self.get_image_urls(keyword, max_results)

            if not urls:
                logger.warning(f"No images found for: {keyword}")
                continue

            keyword_dir = self.output_dir / FileUtils.sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            for idx, url in enumerate(urls):
                # Determine source from URL for naming
                source = self._identify_source(url)
                filepath = keyword_dir / f"{source}_{idx:05d}.jpg"

                if self.download_file(url, filepath):
                    downloaded_paths.append(filepath)

        return downloaded_paths

    def _identify_source(self, url: str) -> str:
        """
        Identify the source of an image URL.

        Args:
            url: Image URL.

        Returns:
            Source name.
        """
        url_lower = url.lower()
        if "unsplash" in url_lower:
            return "unsplash"
        elif "pexels" in url_lower:
            return "pexels"
        elif "flickr" in url_lower or "staticflickr" in url_lower:
            return "flickr"
        elif "wikimedia" in url_lower or "wikipedia" in url_lower:
            return "wikimedia"
        elif "nasa" in url_lower or "earthobservatory" in url_lower:
            return "nasa"
        return "unknown"

    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all sources.

        Returns:
            Dictionary with source status information.
        """
        status = {}
        for source_name in self.SOURCE_REGISTRY:
            is_enabled = source_name in self.enabled_sources
            status[source_name] = {
                "enabled": is_enabled,
                "priority": self.source_priorities.get(source_name),
                "crawler_type": self.SOURCE_REGISTRY[source_name].__name__,
            }

            # Add rate limiter info if available
            crawler = self.crawlers.get(source_name)
            if crawler and hasattr(crawler, "rate_limiter"):
                status[source_name]["remaining_requests"] = (
                    crawler.rate_limiter.get_remaining_requests()
                )

        return status
