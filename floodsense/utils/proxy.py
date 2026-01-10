"""
Proxy management utilities.

Provides proxy rotation and management for web crawling.
"""

import random
from typing import Dict, List, Optional

from loguru import logger


class ProxyManager:
    """
    Manages proxy rotation for web requests.

    Supports HTTP/HTTPS proxies with optional rotation.
    """

    def __init__(
        self,
        proxy_list: Optional[List[str]] = None,
        http_proxy: Optional[str] = None,
        https_proxy: Optional[str] = None,
        rotation_enabled: bool = False,
    ) -> None:
        """
        Initialize ProxyManager.

        Args:
            proxy_list: List of proxy URLs for rotation.
            http_proxy: Default HTTP proxy.
            https_proxy: Default HTTPS proxy.
            rotation_enabled: Whether to rotate proxies.
        """
        self.proxy_list = proxy_list or []
        self.http_proxy = http_proxy
        self.https_proxy = https_proxy
        self.rotation_enabled = rotation_enabled
        self._current_index = 0
        self._failed_proxies: set[str] = set()

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """
        Get proxy configuration for requests.

        Returns:
            Dictionary with 'http' and 'https' proxy URLs, or None.
        """
        if self.rotation_enabled and self.proxy_list:
            return self._get_rotated_proxy()

        if self.http_proxy or self.https_proxy:
            return {
                "http": self.http_proxy,
                "https": self.https_proxy or self.http_proxy,
            }

        return None

    def _get_rotated_proxy(self) -> Optional[Dict[str, str]]:
        """
        Get next proxy from rotation list.

        Returns:
            Dictionary with proxy URLs or None if all failed.
        """
        available = [p for p in self.proxy_list if p not in self._failed_proxies]

        if not available:
            logger.warning("All proxies have failed, resetting failed list")
            self._failed_proxies.clear()
            available = self.proxy_list

        if not available:
            return None

        proxy = random.choice(available)
        return {"http": proxy, "https": proxy}

    def mark_failed(self, proxy_url: str) -> None:
        """
        Mark a proxy as failed.

        Args:
            proxy_url: The failed proxy URL.
        """
        self._failed_proxies.add(proxy_url)
        logger.warning(f"Proxy marked as failed: {proxy_url}")

    def reset_failed(self) -> None:
        """Reset all failed proxy markers."""
        self._failed_proxies.clear()
        logger.info("Failed proxy list has been reset")

    @classmethod
    def from_file(cls, filepath: str, rotation_enabled: bool = True) -> "ProxyManager":
        """
        Load proxies from a text file (one per line).

        Args:
            filepath: Path to proxy list file.
            rotation_enabled: Whether to enable rotation.

        Returns:
            ProxyManager instance.
        """
        proxy_list = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        proxy_list.append(line)
            logger.info(f"Loaded {len(proxy_list)} proxies from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Proxy file not found: {filepath}")

        return cls(proxy_list=proxy_list, rotation_enabled=rotation_enabled)
