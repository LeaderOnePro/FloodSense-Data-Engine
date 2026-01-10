# FloodSense Data Engine

A comprehensive pipeline for collecting, generating, and cleaning flood disaster imagery data for Vision Language Model (VLM) fine-tuning.

## Project Overview

This project is part of a research initiative to fine-tune Qwen3-VL-235B (a multimodal large model) on the Tinker platform for identifying flood-affected personnel, vehicles, and buildings after disasters.

## Features

- **Web Crawling**: Multi-threaded image and video crawling from various sources
- **Smart Frame Extraction**: Scene detection-based keyframe extraction from videos
- **Quality Control**: Blur detection, resolution standardization, and deduplication
- **Synthetic Data Generation**: AI-powered image generation using Google Gemini API
- **Complete Pipeline**: End-to-end data processing with comprehensive reporting

## Installation

```bash
# Clone the repository
git clone https://github.com/LeaderOnePro/FloodSense-Data-Engine.git
cd FloodSense-Data-Engine

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Configuration

Edit `config/config.yaml` to customize settings:

```yaml
# Crawler settings
crawler:
  max_workers: 4
  timeout: 30
  retry_count: 3

# Processor settings
processor:
  target_resolution: [1920, 1080]  # 1080p
  min_resolution: [1280, 720]      # 720p minimum
  blur_threshold: 100.0
  scene_threshold: 30.0
  phash_threshold: 8

# Synthesizer settings
synthesizer:
  api_key: null  # Set via FLOODSENSE_API_KEY environment variable
  model: "gemini-2.0-flash-preview-image-generation"
  max_retries: 5
```

## Usage

### Crawl Images

```bash
python scripts/crawl_images.py \
  --keywords "flood trapped people" "submerged car" "flood aerial view" \
  --max-results 100 \
  --output-dir data/raw
```

### Crawl Videos

```bash
python scripts/crawl_videos.py \
  --keywords "flood disaster" "flash flood" \
  --max-results 10 \
  --output-dir data/raw
```

### Extract Video Frames

```bash
python scripts/extract_frames.py \
  --input-dir data/raw \
  --output-dir data/processed/video_frames \
  --scene-threshold 30.0 \
  --blur-threshold 100.0
```

### Clean Data

```bash
python scripts/clean_data.py \
  --input-dir data/raw \
  --output-dir data/processed
```

### Generate Synthetic Data

```bash
# Set API key
export FLOODSENSE_API_KEY="your-api-key-here"

# Generate images
python scripts/generate_synthetic.py \
  --prompts config/prompts.json \
  --output-dir data/synthetic
```

## Project Structure

```
FloodSense-Data-Engine/
├── floodsense/
│   ├── crawlers/          # Web scraping modules
│   │   ├── base.py        # Base crawler class
│   │   ├── image_spider.py    # Image crawler
│   │   └── video_crawler.py   # Video crawler
│   ├── processors/        # Data processing modules
│   │   ├── image_processor.py     # Image quality control
│   │   ├── video_processor.py     # Video frame extraction
│   │   └── cleaning_pipeline.py   # Complete cleaning pipeline
│   ├── synthesizers/      # AI data generation
│   │   └── nano_banana_client.py  # Gemini API client
│   └── utils/             # Utility modules
│       ├── config.py      # Configuration management
│       ├── file_utils.py  # File operations
│       ├── logger.py      # Logging setup
│       └── proxy.py       # Proxy management
├── scripts/               # Executable scripts
│   ├── crawl_images.py
│   ├── crawl_videos.py
│   ├── extract_frames.py
│   ├── clean_data.py
│   └── generate_synthetic.py
├── config/                # Configuration files
│   ├── config.yaml
│   └── prompts.json
├── data/                  # Data directories
│   ├── raw/               # Raw crawled data
│   ├── processed/         # Cleaned data
│   └── synthetic/         # AI-generated data
└── tests/                 # Unit tests
```

## Tech Stack

- **Python**: 3.10+
- **Web Scraping**: Selenium, Playwright, Beautiful Soup, yt-dlp
- **Image Processing**: Pillow, OpenCV, imagehash
- **AI/ML**: Google Gemini API (google-genai)
- **Concurrency**: asyncio, ThreadPoolExecutor
- **Logging**: loguru
- **Configuration**: Pydantic, PyYAML

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.