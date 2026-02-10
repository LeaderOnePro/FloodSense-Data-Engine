"""
Internationalisation helpers for the FloodSense TUI.

Usage::

    from floodsense.utils.i18n import t, set_lang, get_lang

    set_lang("zh")
    print(t("dashboard"))  # => "仪表盘"
"""

from __future__ import annotations

_lang: str = "en"

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # nav / pane titles
        "dashboard": "Dashboard",
        "crawl": "Crawl",
        "process": "Process",
        "synthesize": "Synthesize",
        "validate": "Validate",
        "config": "Config",
        "logs": "Logs",
        "quit": "Quit",
        # dashboard
        "raw_data": "Raw Data",
        "processed_data": "Processed Data",
        "synthetic_data": "Synthetic Data",
        "refresh": "Refresh",
        "registered_crawlers": "Registered Crawlers",
        "col_name": "Name",
        "col_class": "Class",
        "col_type": "Type",
        # crawl
        "keywords_comma": "Keywords (comma)",
        "max_results": "Max results",
        "output_dir": "Output dir",
        "crawler_type": "Crawler type",
        "opt_image_spider": "Image Spider",
        "opt_multi_source": "Multi-Source API",
        "opt_video": "Video (yt-dlp)",
        "select_api_sources": "Select API sources:",
        "start_crawl": "Start Crawl",
        "output": "Output",
        "msg_enter_keywords": "Please enter keywords.",
        "msg_starting_spider": "Starting ImageSpider …",
        "msg_starting_multi": "Starting MultiSourceCrawler (sources={sources}) …",
        "msg_starting_video": "Starting VideoCrawler …",
        "msg_done_images": "Done — {n} images downloaded.",
        "msg_done_videos": "Done — {n} videos downloaded.",
        "msg_playwright_missing": "Playwright browsers not installed.\nRun: playwright install chromium",
        # process
        "input_dir": "Input dir",
        "extract_video_frames": "Extract video frames",
        "remove_blurry": "Remove blurry images",
        "deduplicate": "Deduplicate",
        "scene_threshold": "Scene threshold",
        "blur_threshold": "Blur threshold",
        "run_pipeline": "Run Pipeline",
        "msg_running_pipeline": "Running cleaning pipeline …",
        "msg_pipeline_complete": "Pipeline complete.",
        # synthesize
        "api_key": "API Key",
        "prompts_per_line": "Prompts (one per line):",
        "prompts_file": "Prompts file",
        "enhance_prompts": "Enhance prompts",
        "generate": "Generate",
        "msg_loading_prompts": "Loading prompts from {path} …",
        "msg_generating": "Generating {n} images …",
        "msg_no_prompts": "No prompts provided.",
        "msg_done_generated": "Done — {n} images generated.",
        # validate
        "image_dir": "Image dir",
        "enable_heuristic": "Enable heuristic filter",
        "enable_clip": "Enable CLIP filter",
        "delete_invalid": "Delete invalid images",
        "btn_validate": "Validate",
        "msg_no_images": "No images found.",
        "msg_validating": "Validating {n} images …",
        "msg_validate_done": "Done — {valid} valid, {invalid} invalid out of {total} images ({rate:.1f}%)",
        "msg_deleted": "Deleted {n} invalid images.",
        # config
        "config_file": "Config file",
        "reload": "Reload",
        "apply": "Apply",
        "save": "Save",
        "msg_config_reloaded": "Config reloaded from {path}.",
        "msg_config_applied": "Config applied.",
        "msg_config_saved": "Config saved to {path}.",
        "msg_apply_failed": "Apply failed: {err}",
        # logs
        "clear": "Clear",
    },
    "zh": {
        # nav
        "dashboard": "仪表盘",
        "crawl": "爬取",
        "process": "处理",
        "synthesize": "合成",
        "validate": "验证",
        "config": "配置",
        "logs": "日志",
        "quit": "退出",
        # dashboard
        "raw_data": "原始数据",
        "processed_data": "已处理数据",
        "synthetic_data": "合成数据",
        "refresh": "刷新",
        "registered_crawlers": "已注册爬虫",
        "col_name": "名称",
        "col_class": "类",
        "col_type": "类型",
        # crawl
        "keywords_comma": "关键词（逗号分隔）",
        "max_results": "最大数量",
        "output_dir": "输出目录",
        "crawler_type": "爬虫类型",
        "opt_image_spider": "图片爬虫",
        "opt_multi_source": "多源 API",
        "opt_video": "视频 (yt-dlp)",
        "select_api_sources": "选择 API 源：",
        "start_crawl": "开始爬取",
        "output": "输出",
        "msg_enter_keywords": "请输入关键词。",
        "msg_starting_spider": "正在启动 ImageSpider …",
        "msg_starting_multi": "正在启动 MultiSourceCrawler (源={sources}) …",
        "msg_starting_video": "正在启动 VideoCrawler …",
        "msg_done_images": "完成 — 已下载 {n} 张图片。",
        "msg_done_videos": "完成 — 已下载 {n} 个视频。",
        "msg_playwright_missing": "Playwright 浏览器未安装。\n请运行: playwright install chromium",
        # process
        "input_dir": "输入目录",
        "extract_video_frames": "提取视频帧",
        "remove_blurry": "去除模糊图片",
        "deduplicate": "去重",
        "scene_threshold": "场景阈值",
        "blur_threshold": "模糊阈值",
        "run_pipeline": "运行管线",
        "msg_running_pipeline": "正在运行清洗管线 …",
        "msg_pipeline_complete": "管线完成。",
        # synthesize
        "api_key": "API 密钥",
        "prompts_per_line": "提示词（每行一条）：",
        "prompts_file": "提示词文件",
        "enhance_prompts": "增强提示词",
        "generate": "生成",
        "msg_loading_prompts": "正在加载提示词：{path} …",
        "msg_generating": "正在生成 {n} 张图片 …",
        "msg_no_prompts": "未提供提示词。",
        "msg_done_generated": "完成 — 已生成 {n} 张图片。",
        # validate
        "image_dir": "图片目录",
        "enable_heuristic": "启用启发式过滤",
        "enable_clip": "启用 CLIP 过滤",
        "delete_invalid": "删除无效图片",
        "btn_validate": "验证",
        "msg_no_images": "未找到图片。",
        "msg_validating": "正在验证 {n} 张图片 …",
        "msg_validate_done": "完成 — {valid} 有效，{invalid} 无效，共 {total} 张（{rate:.1f}%）",
        "msg_deleted": "已删除 {n} 张无效图片。",
        # config
        "config_file": "配置文件",
        "reload": "重载",
        "apply": "应用",
        "save": "保存",
        "msg_config_reloaded": "已从 {path} 重载配置。",
        "msg_config_applied": "配置已应用。",
        "msg_config_saved": "配置已保存到 {path}。",
        "msg_apply_failed": "应用失败：{err}",
        # logs
        "clear": "清空",
    },
}


def t(key: str, **kwargs: object) -> str:
    """Look up a translated string by *key*, with optional format arguments."""
    text = _STRINGS.get(_lang, _STRINGS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def set_lang(lang: str) -> None:
    """Set the active language (``"en"`` or ``"zh"``)."""
    global _lang
    _lang = lang


def get_lang() -> str:
    """Return the active language code."""
    return _lang
