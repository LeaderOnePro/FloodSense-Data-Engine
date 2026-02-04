"""
Video crawler for downloading flood-related videos.

Uses yt-dlp for downloading videos from YouTube and other platforms.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

import yt_dlp
from loguru import logger
from tqdm import tqdm

from floodsense.crawlers.base import BaseCrawler


class VideoCrawler(BaseCrawler):
    """
    Video crawler using yt-dlp for downloading videos.

    Supports YouTube and other video platforms.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize VideoCrawler."""
        super().__init__(*args, **kwargs)

    def crawl(
        self,
        keywords: List[str],
        max_results: int = 10,
        platform: str = "youtube",
    ) -> List[Path]:
        """
        Crawl videos for given keywords.

        Args:
            keywords: List of search keywords.
            max_results: Maximum videos to download per keyword.
            platform: Video platform (youtube, vimeo, etc.).

        Returns:
            List of paths to downloaded videos.
        """
        downloaded_paths: List[Path] = []

        for keyword in keywords:
            logger.info(f"Starting video crawl for keyword: {keyword}")
            urls = self._search_videos(keyword, max_results, platform)
            logger.info(f"Found {len(urls)} video URLs for '{keyword}'")

            keyword_dir = self.output_dir / self._sanitize_keyword(keyword)
            keyword_dir.mkdir(parents=True, exist_ok=True)

            paths = self._download_videos(urls, keyword_dir, keyword)
            downloaded_paths.extend(paths)

        return downloaded_paths

    def _search_videos(
        self,
        keyword: str,
        max_results: int,
        platform: str,
    ) -> List[str]:
        """
        Search for videos using yt-dlp.

        Args:
            keyword: Search keyword.
            max_results: Maximum results to return.
            platform: Video platform.

        Returns:
            List of video URLs.
        """
        urls: List[str] = []

        # yt-dlp search format: ytsearch{N}:keyword for YouTube
        if platform == "youtube":
            search_query = f"ytsearch{max_results}:{keyword}"
        else:
            search_query = f"{platform}search{max_results}:{keyword}"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)

                if info and "entries" in info:
                    for entry in info["entries"]:
                        if entry and entry.get("url"):
                            urls.append(entry["url"])
                        elif entry and entry.get("id"):
                            # Build YouTube URL from video ID
                            urls.append(f"https://www.youtube.com/watch?v={entry['id']}")

        except Exception as e:
            logger.error(f"Failed to search videos for '{keyword}': {e}")

        logger.info(f"Found {len(urls)} videos for '{keyword}'")
        return urls

    def _download_videos(
        self,
        urls: List[str],
        output_dir: Path,
        keyword: str,
    ) -> List[Path]:
        """
        Download videos using yt-dlp.

        Args:
            urls: List of video URLs.
            output_dir: Directory to save videos.
            keyword: Keyword for progress display.

        Returns:
            List of paths to downloaded videos.
        """
        downloaded_paths: List[Path] = []

        with tqdm(
            total=len(urls),
            desc=f"Downloading videos: {keyword}",
            unit="vid",
        ) as pbar:
            for url in urls:
                try:
                    filepath = self._download_single_video(url, output_dir)
                    if filepath:
                        downloaded_paths.append(filepath)
                except Exception as e:
                    logger.error(f"Failed to download video {url}: {e}")
                pbar.update(1)

        return downloaded_paths

    def _download_single_video(
        self,
        url: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """
        Download a single video.

        Args:
            url: Video URL.
            output_dir: Directory to save video.

        Returns:
            Path to downloaded file or None if failed.
        """
        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "restrictfilenames": True,
            "retries": 5,
            "fragment_retries": 5,
            "socket_timeout": 60,
            "merge_output_format": "mp4",
        }

        # Add proxy if configured
        if self.proxy_manager:
            proxy = self.proxy_manager.get_proxy()
            if proxy:
                ydl_opts["proxy"] = proxy.get("http") or proxy.get("https")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    filename = ydl.prepare_filename(info)
                    return Path(filename)
        except Exception as e:
            logger.error(f"Failed to download video {url}: {e}")

        return None

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

    def get_video_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get video URLs for a keyword.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of video URLs.
        """
        return self._search_videos(keyword, max_results, "youtube")

    def get_image_urls(self, keyword: str, max_results: int = 100) -> List[str]:
        """
        Get image URLs (not applicable for video crawler).

        This method is required by BaseCrawler but not used for videos.
        Returns video URLs instead.

        Args:
            keyword: Search keyword.
            max_results: Maximum number of URLs to return.

        Returns:
            List of video URLs.
        """
        return self.get_video_urls(keyword, max_results)
