# FloodSense Data Engine

[English](README.md) | [中文](README_CN.md)

A comprehensive pipeline for collecting, generating, and cleaning flood disaster imagery data for Vision Language Model (VLM) fine-tuning.

## Project Overview

This project is part of a research initiative to fine-tune Qwen3-VL-235B (a multimodal large model) on the Tinker platform for identifying flood-affected personnel, vehicles, and buildings after disasters.

## Features

- **Web Crawling** — Multi-source image crawling (Bing, Unsplash, Pexels, Flickr, Wikimedia, NASA) and video downloading via yt-dlp
- **Smart Frame Extraction** — Scene detection-based keyframe extraction from videos
- **Quality Control** — Blur detection, resolution standardization, pHash deduplication
- **Content Validation** — Heuristic filtering + CLIP-based semantic validation
- **Synthetic Data Generation** — AI-powered image generation using Google Gemini API
- **Cleaning Pipeline** — End-to-end data processing with comprehensive JSON reporting
- **TUI** — Unified Textual terminal interface (`python main.py`) to operate all modules interactively

## Quick Start

```bash
# Clone
git clone https://github.com/LeaderOnePro/FloodSense-Data-Engine.git
cd FloodSense-Data-Engine

# Install
pip install -e ".[dev]"
pip install textual          # for TUI

# Launch TUI
python main.py
```

## TUI

The built-in TUI provides a sidebar + 7 panes accessible via keyboard shortcuts:

| Key | Pane | Description |
|-----|------|-------------|
| `d` | Dashboard | Data statistics, registered crawler list |
| `c` | Crawl | Image / video crawling with source selection |
| `p` | Process | Cleaning pipeline (blur, dedup, frames) |
| `s` | Synthesize | Gemini-based synthetic image generation |
| `v` | Validate | Heuristic + CLIP content validation |
| `o` | Config | YAML config editor with reload / apply / save |
| `l` | Logs | Global loguru log viewer |
| `q` | — | Quit |

All long-running operations execute in background worker threads; the UI stays responsive.

## Configuration

Edit `config/config.yaml` or use the Config pane in the TUI:

```yaml
crawler:
  max_workers: 4
  timeout: 30
  retry_count: 3

processor:
  target_resolution: [1920, 1080]
  min_resolution: [1280, 720]
  blur_threshold: 100.0
  scene_threshold: 30.0
  phash_threshold: 8

synthesizer:
  api_key: null  # or set FLOODSENSE_API_KEY env var
  model: "gemini-3-pro-image-preview"

validator:
  enable_heuristic: true
  enable_clip: true
  clip_threshold: 0.25

api_sources:
  unsplash:
    api_key: null  # or set FLOODSENSE_UNSPLASH_API_KEY
  pexels:
    api_key: null  # or set FLOODSENSE_PEXELS_API_KEY
```

## Project Structure

```
FloodSense-Data-Engine/
├── main.py                        # Textual TUI entry point
├── floodsense/
│   ├── crawlers/
│   │   ├── base.py                # BaseCrawler + registry
│   │   ├── image_spider.py        # Bing image spider (Playwright)
│   │   ├── video_crawler.py       # yt-dlp video crawler
│   │   ├── api_crawlers/          # API-based crawlers
│   │   │   ├── unsplash_crawler.py
│   │   │   ├── pexels_crawler.py
│   │   │   ├── flickr_crawler.py
│   │   │   ├── wikimedia_crawler.py
│   │   │   └── multi_source_crawler.py
│   │   └── satellite_crawlers/
│   │       └── nasa_crawler.py
│   ├── processors/
│   │   ├── image_processor.py     # Resolution, blur, dedup
│   │   ├── video_processor.py     # Scene-based frame extraction
│   │   └── cleaning_pipeline.py   # Orchestrator
│   ├── synthesizers/
│   │   └── img_gen_models_client.py  # Gemini API client
│   ├── validators/
│   │   ├── base_validator.py      # Strategy interface
│   │   └── image_validator.py     # Heuristic + CLIP validator
│   └── utils/
│       ├── config.py              # Pydantic config models
│       ├── file_utils.py          # FileUtils, CheckpointManager
│       ├── logger.py              # Logging setup
│       └── proxy.py               # Proxy rotation
├── config/
│   ├── config.yaml
│   └── prompts.json
└── data/
    ├── raw/                       # Crawled data
    ├── processed/                 # Cleaned data
    └── synthetic/                 # AI-generated data
```

## Tech Stack

- **Python** 3.10+
- **TUI**: [Textual](https://textual.textualize.io/)
- **Web Scraping**: Playwright, Beautiful Soup, yt-dlp, requests
- **Image Processing**: Pillow, OpenCV, imagehash
- **AI/ML**: Google Gemini API, CLIP (transformers + torch)
- **Concurrency**: ThreadPoolExecutor
- **Logging**: loguru
- **Configuration**: Pydantic, PyYAML

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.