"""
Base API crawler with rate limiting functionality.

Provides foundation for API-based image crawlers.
"""

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from floodsense.crawlers.base import BaseCrawler
from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


class RateLimiter:
    """
    Rate limiter that tracks requests per hour and auto-waits when limit is reached.
    """

    def __init__(self, requests_per_hour: int) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_hour: Maximum requests allowed per hour.
        """
        self.requests_per_hour = requests_per_hour
        self.requests: List[float] = []
        self.window_seconds = 3600  # 1 hour window

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        current_time = time.time()

        # Remove requests outside the window
        self.requests = [
            t for t in self.requests if current_time - t < self.window_seconds
        ]

        if len(self.requests) >= self.requests_per_hour:
            # Calculate wait time until oldest request falls out of window
            oldest_request = min(self.requests)
            wait_time = self.window_seconds - (current_time - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                # Clean up after waiting
                self.requests = [
                    t
                    for t in self.requests
                    if time.time() - t < self.window_seconds
                ]

    def record_request(self) -> None:
        """Record a request timestamp."""
        self.requests.append(time.time())

    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        current_time = time.time()
        self.requests = [
            t for t in self.requests if current_time - t < self.window_seconds
        ]
        return max(0, self.requests_per_hour - len(self.requests))


class BaseAPICrawler(BaseCrawler):
    """
    Abstract base class for API-based image crawlers.

    Extends BaseCrawler with rate limiting and API-specific functionality.
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        rate_limit: int = 100,
    ) -> None:
        """
        Initialize BaseAPICrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager for requests.
            output_dir: Directory to save downloaded files.
            api_key: API key for the service.
            rate_limit: Requests per hour limit.
        """
        super().__init__(config, proxy_manager, output_dir)
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit)

    @property
    @abstractmethod
    def api_base_url(self) -> str:
        """Return the base URL for the API."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this source for logging."""
        pass

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.

        Returns:
            Dictionary of authentication headers.
        """
        pass

    @abstractmethod
    def _build_search_params(self, keyword: str, page: int, per_page: int) -> Dict[str, Any]:
        """
        Build search query parameters.

        Args:
            keyword: Search keyword.
            page: Page number for pagination.
            per_page: Results per page.

        Returns:
            Dictionary of query parameters.
        """
        pass

    @abstractmethod
    def _parse_search_response(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse search response to extract image data.

        Args:
            response_data: Raw API response data.

        Returns:
            List of image dictionaries with 'url', 'id', and optional metadata.
        """
        pass

    @abstractmethod
    def _get_total_pages(self, response_data: Dict[str, Any], per_page: int) -> int:
        """
        Get total number of pages from response.

        Args:
            response_data: Raw API response data.
            per_page: Results per page.

        Returns:
            Total number of pages.
        """
        pass

    def _make_api_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request with rate limiting.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            JSON response data or None if failed.
        """
        self.rate_limiter.wait_if_needed()

        url = f"{self.api_base_url}{endpoint}"
        headers = {**self._get_headers(), **self._get_auth_headers()}
        headers["Accept"] = "application/json"

        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.timeout,
                proxies=self._get_proxies(),
            )
            self.rate_limiter.record_request()

            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(
                    f"Rate limited by {self.source_name}. Waiting {retry_after} seconds..."
                )
                time.sleep(retry_after)
                return self._make_api_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"API request failed for {self.source_name}: {e}")
            return None

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs for a keyword using the API.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of image URLs.
        """
        urls: List[str] = []
        per_page = min(30, max_results)
        page = 1
        total_pages = None

        logger.info(f"Searching {self.source_name} for: {keyword}")

        while len(urls) < max_results:
            params = self._build_search_params(keyword, page, per_page)
            response_data = self._make_api_request("/search/photos", params)

            if response_data is None:
                logger.warning(f"Failed to get results from {self.source_name}")
                break

            images = self._parse_search_response(response_data)

            if not images:
                logger.debug(f"No more results from {self.source_name}")
                break

            for image in images:
                if len(urls) >= max_results:
                    break
                if image.get("url"):
                    urls.append(image["url"])

            if total_pages is None:
                total_pages = self._get_total_pages(response_data, per_page)

            if total_pages and page >= total_pages:
                break

            page += 1

        logger.info(f"Found {len(urls)} images from {self.source_name}")
        return urls

    def crawl(self, keywords: List[str], max_results: int = 100) -> List[Path]:
        """
        Crawl images for given keywords.

        Args:
            keywords: List of search keywords.
            max_results: Maximum images per keyword.

        Returns:
            List of paths to downloaded files.
        """
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"[{self.source_name}] Crawling for: {keyword}")
            urls = self.get_image_urls(keyword, max_results)

            if not urls:
                continue

            keyword_dir = self.output_dir / self._sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            for idx, url in enumerate(urls):
                filepath = keyword_dir / f"{self.source_name.lower()}_{idx:05d}.jpg"
                if self.download_file(url, filepath):
                    downloaded_paths.append(filepath)

        return downloaded_paths

    @staticmethod
    def _sanitize_keyword(keyword: str) -> str:
        """
        Sanitize keyword for use as directory name.

        Args:
            keyword: Raw keyword string.

        Returns:
            Sanitized keyword.
        """
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', "", keyword)
        sanitized = sanitized.strip().replace(" ", "_")
        return sanitized
