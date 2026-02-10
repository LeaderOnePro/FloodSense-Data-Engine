# FloodSense Data Engine

[English](README.md) | [中文](README_CN.md)

面向视觉语言模型 (VLM) 微调的洪灾影像数据采集、生成与清洗一体化流水线。

## 项目概述

本项目是一项研究计划的组成部分，旨在基于 Tinker 平台对 Qwen3-VL-235B（多模态大模型）进行微调，使其能够识别灾后洪水中受困的人员、车辆和建筑物。

## 功能特性

- **网络爬虫** — 多源图片抓取（Bing、Unsplash、Pexels、Flickr、Wikimedia、NASA）及 yt-dlp 视频下载
- **智能帧提取** — 基于场景检测的视频关键帧提取
- **质量控制** — 模糊检测、分辨率标准化、pHash 去重
- **内容验证** — 启发式过滤 + CLIP 语义验证
- **合成数据生成** — 基于 Google Gemini API 的 AI 图像生成
- **清洗流水线** — 端到端数据处理，输出 JSON 统计报告
- **TUI 终端界面** — 基于 Textual 的统一终端界面 (`python main.py`)，交互式操作全部模块，支持中英文实时切换

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/LeaderOnePro/FloodSense-Data-Engine.git
cd FloodSense-Data-Engine

# 安装
pip install -e ".[dev]"
pip install textual          # TUI 依赖

# 启动 TUI
python main.py
```

## TUI 终端界面

内置 TUI 提供侧栏 + 7 个面板，支持快捷键切换：

| 快捷键 | 面板 | 说明 |
|--------|------|------|
| `d` | Dashboard | 数据统计、已注册爬虫列表 |
| `c` | Crawl | 图片/视频爬取，支持多源选择 |
| `p` | Process | 清洗流水线（去模糊、去重、帧提取） |
| `s` | Synthesize | 基于 Gemini 的合成图像生成 |
| `v` | Validate | 启发式 + CLIP 内容验证 |
| `o` | Config | YAML 配置编辑器（加载/应用/保存） |
| `l` | Logs | 全局 loguru 日志查看器 |
| `i` | — | 切换语言（EN/中文） |
| `q` | — | 退出 |

所有耗时操作在后台 Worker 线程中执行，UI 始终保持响应。

## 配置

编辑 `config/config.yaml` 或在 TUI 的 Config 面板中操作：

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
  api_key: null  # 或设置环境变量 FLOODSENSE_API_KEY
  model: "gemini-3-pro-image-preview"

validator:
  enable_heuristic: true
  enable_clip: true
  clip_threshold: 0.25

api_sources:
  unsplash:
    api_key: null  # 或设置 FLOODSENSE_UNSPLASH_API_KEY
  pexels:
    api_key: null  # 或设置 FLOODSENSE_PEXELS_API_KEY
```

## 项目结构

```
FloodSense-Data-Engine/
├── main.py                        # Textual TUI 入口
├── floodsense/
│   ├── crawlers/
│   │   ├── base.py                # BaseCrawler + 爬虫注册表
│   │   ├── image_spider.py        # Bing 图片爬虫（Playwright）
│   │   ├── video_crawler.py       # yt-dlp 视频爬虫
│   │   ├── api_crawlers/          # API 爬虫
│   │   │   ├── unsplash_crawler.py
│   │   │   ├── pexels_crawler.py
│   │   │   ├── flickr_crawler.py
│   │   │   ├── wikimedia_crawler.py
│   │   │   └── multi_source_crawler.py
│   │   └── satellite_crawlers/
│   │       └── nasa_crawler.py
│   ├── processors/
│   │   ├── image_processor.py     # 分辨率、模糊、去重
│   │   ├── video_processor.py     # 场景检测帧提取
│   │   └── cleaning_pipeline.py   # 流水线调度器
│   ├── synthesizers/
│   │   └── img_gen_models_client.py  # Gemini API 客户端
│   ├── validators/
│   │   ├── base_validator.py      # 策略接口
│   │   └── image_validator.py     # 启发式 + CLIP 验证器
│   └── utils/
│       ├── config.py              # Pydantic 配置模型
│       ├── file_utils.py          # 文件工具、断点续传
│       ├── i18n.py                # 中英文国际化
│       ├── logger.py              # 日志配置
│       └── proxy.py               # 代理轮换
├── config/
│   ├── config.yaml
│   └── prompts.json
└── data/
    ├── raw/                       # 爬取的原始数据
    ├── processed/                 # 清洗后的数据
    └── synthetic/                 # AI 生成的数据
```

## 技术栈

- **Python** 3.10+
- **TUI**: [Textual](https://textual.textualize.io/)
- **网络爬虫**: Playwright, Beautiful Soup, yt-dlp, requests
- **图像处理**: Pillow, OpenCV, imagehash
- **AI/ML**: Google Gemini API, CLIP (transformers + torch)
- **并发**: ThreadPoolExecutor
- **日志**: loguru
- **配置管理**: Pydantic, PyYAML

## 许可证

Apache License 2.0

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 联系方式

如有问题或需要支持，请在 GitHub 上创建 Issue。
