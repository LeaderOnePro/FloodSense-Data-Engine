"""
Base crawler class with common functionality.

Provides foundation for specific crawler implementations.
"""

import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from floodsense.utils.config import CrawlerConfig
from floodsense.utils.proxy import ProxyManager


class BaseCrawler(ABC):
    """
    Abstract base class for all crawlers.

    Provides common functionality like session management,
    retry logic, and header spoofing.
    """

    DEFAULT_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize BaseCrawler.

        Args:
            config: Crawler configuration.
            proxy_manager: Optional proxy manager for requests.
            output_dir: Directory to save downloaded files.
        """
        self.config = config or CrawlerConfig()
        self.proxy_manager = proxy_manager
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = self._create_session()
        self.user_agents = self.config.user_agents or self.DEFAULT_USER_AGENTS

    def _create_session(self) -> requests.Session:
        """
        Create a configured requests session.

        Returns:
            Configured Session object.
        """
        session = requests.Session()
        session.headers.update(self._get_headers())
        return session

    def _get_headers(self) -> Dict[str, str]:
        """
        Get randomized request headers.

        Returns:
            Dictionary of HTTP headers.
        """
        user_agent = random.choice(self.user_agents)
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """
        Get proxy configuration.

        Returns:
            Proxy dictionary or None.
        """
        if self.proxy_manager:
            return self.proxy_manager.get_proxy()
        return None

    def request_with_retry(
        self,
        url: str,
        method: str = "GET",
        **kwargs: Any,
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic.

        Args:
            url: Target URL.
            method: HTTP method (GET, POST, etc.).
            **kwargs: Additional arguments for requests.

        Returns:
            Response object or None if all retries failed.
        """
        # Update headers for each request
        self.session.headers.update(self._get_headers())

        for attempt in range(self.config.retry_count):
            try:
                proxies = self._get_proxies()
                response = self.session.request(
                    method,
                    url,
                    timeout=self.config.timeout,
                    proxies=proxies,
                    **kwargs,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.retry_count}): "
                    f"{url} - {e}"
                )
                if self.proxy_manager and proxies:
                    proxy_url = proxies.get("http") or proxies.get("https")
                    if proxy_url:
                        self.proxy_manager.mark_failed(proxy_url)

                if attempt < self.config.retry_count - 1:
                    delay = self.config.retry_delay * (2**attempt)  # Exponential backoff
                    delay += random.uniform(0, 1)  # Add jitter
                    time.sleep(delay)

        logger.error(f"All retry attempts failed for: {url}")
        return None

    def download_file(
        self,
        url: str,
        filepath: Path,
        chunk_size: int = 8192,
    ) -> bool:
        """
        Download a file from URL.

        Args:
            url: File URL.
            filepath: Destination path.
            chunk_size: Download chunk size.

        Returns:
            True if download successful, False otherwise.
        """
        try:
            response = self.request_with_retry(url, stream=True)
            if response is None:
                return False

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            logger.debug(f"Downloaded: {filepath}")
            return True

        except IOError as e:
            logger.error(f"Failed to save file {filepath}: {e}")
            return False

    @abstractmethod
    def crawl(self, keywords: List[str], max_results: int = 100) -> List[Path]:
        """
        Execute crawl operation.

        Args:
            keywords: List of search keywords.
            max_results: Maximum number of results per keyword.

        Returns:
            List of paths to downloaded files.
        """
        pass

    @abstractmethod
    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs for a keyword.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of image URLs.
        """
        pass
