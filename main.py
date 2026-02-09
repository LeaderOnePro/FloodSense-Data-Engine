"""
FloodSense Data Engine — Textual TUI.

Unified terminal interface for crawling, processing, synthesising and
validating flood imagery.

Launch:
    python main.py
"""

from __future__ import annotations

import os

# Disable tqdm progress bars so they don't corrupt the TUI
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    SelectionList,
    Static,
    Switch,
    TextArea,
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from floodsense.utils.config import Config
from floodsense.utils.file_utils import FileUtils

# Importing the crawlers package triggers @BaseCrawler.register decorators
from floodsense.crawlers import (  # noqa: F401
    BaseCrawler,
    ImageSpider,
    MultiSourceCrawler,
    VideoCrawler,
)
from floodsense.processors.cleaning_pipeline import CleaningPipeline
from floodsense.validators.image_validator import ImageValidator

# ImageGenClient depends on google.generativeai which may not be installed.
# Import lazily inside SynthesizePane._run_synth() instead.

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
Screen {
    layout: horizontal;
}

#sidebar {
    width: 24;
    dock: left;
    background: $surface;
    border-right: thick $primary;
    padding: 1;
}

#sidebar Button {
    width: 100%;
    margin-bottom: 1;
}

#sidebar .nav-active {
    background: $primary;
    color: $text;
}

#content-area {
    width: 1fr;
}

ContentSwitcher {
    width: 1fr;
    height: 1fr;
}

.pane {
    padding: 1 2;
}

.card {
    border: round $primary;
    padding: 1 2;
    margin: 0 1 1 0;
    width: 1fr;
    height: auto;
    min-width: 20;
}

.card-title {
    text-style: bold;
    color: $text;
    margin-bottom: 1;
}

.card-value {
    text-style: bold;
    color: $success;
    text-align: center;
}

.section-title {
    text-style: bold;
    margin: 1 0;
    color: $primary;
}

.form-row {
    height: auto;
    margin-bottom: 1;
}

.form-row Label {
    width: 18;
    padding: 1 1 0 0;
}

.form-row Input {
    width: 1fr;
}

.form-row Select {
    width: 1fr;
}

.switch-row {
    height: auto;
    margin-bottom: 1;
}

.switch-row Label {
    width: 30;
    padding: 0 1 0 0;
}

.switch-row Switch {
    width: auto;
}

.action-bar {
    height: auto;
    margin: 1 0;
}

.action-bar Button {
    margin-right: 1;
}

RichLog {
    height: 1fr;
    border: round $surface-lighten-2;
    margin-top: 1;
}

TextArea {
    height: 1fr;
}

DataTable {
    height: 1fr;
}

SelectionList {
    height: auto;
    max-height: 10;
    margin-bottom: 1;
}

#multi-source-group {
    height: auto;
    display: none;
}

#multi-source-group.visible {
    display: block;
}
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PANES                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝


class DashboardPane(VerticalScroll):
    """Overview statistics and crawler registry."""

    DEFAULT_CSS = """
    DashboardPane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Dashboard", classes="section-title")

        with Horizontal():
            with Vertical(classes="card", id="card-raw"):
                yield Static("Raw Data", classes="card-title")
                yield Static("—", id="raw-count", classes="card-value")
            with Vertical(classes="card", id="card-processed"):
                yield Static("Processed Data", classes="card-title")
                yield Static("—", id="processed-count", classes="card-value")
            with Vertical(classes="card", id="card-synthetic"):
                yield Static("Synthetic Data", classes="card-title")
                yield Static("—", id="synthetic-count", classes="card-value")

        with Horizontal(classes="action-bar"):
            yield Button("Refresh", id="btn-refresh-dash", variant="primary")

        yield Static("Registered Crawlers", classes="section-title")
        yield DataTable(id="crawler-table")

    def on_mount(self) -> None:
        table = self.query_one("#crawler-table", DataTable)
        table.add_columns("Name", "Class", "Type")
        self._refresh_table()
        self.action_refresh()

    def _refresh_table(self) -> None:
        table = self.query_one("#crawler-table", DataTable)
        table.clear()
        registry = BaseCrawler.get_registry()
        for name, cls in registry.items():
            table.add_row(name, cls.__name__, "API")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh-dash":
            self.action_refresh()

    @work(thread=True, group="dashboard")
    def action_refresh(self) -> None:
        cfg: Config = self.app.config  # type: ignore[attr-defined]
        raw = self._count_dir(cfg.data.raw_dir)
        processed = self._count_dir(cfg.data.processed_dir)
        synthetic = self._count_dir(cfg.data.synthetic_dir)
        self.app.call_from_thread(self._update_counts, raw, processed, synthetic)

    def _update_counts(self, raw: str, processed: str, synthetic: str) -> None:
        self.query_one("#raw-count", Static).update(raw)
        self.query_one("#processed-count", Static).update(processed)
        self.query_one("#synthetic-count", Static).update(synthetic)

    @staticmethod
    def _count_dir(d: Path) -> str:
        if not d.exists():
            return "0 imgs / 0 vids"
        imgs = sum(1 for _ in FileUtils.iter_images(d))
        vids = sum(1 for _ in FileUtils.iter_videos(d))
        return f"{imgs} imgs / {vids} vids"


class CrawlPane(VerticalScroll):
    """Crawl images / videos from the web."""

    DEFAULT_CSS = """
    CrawlPane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Crawl", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Keywords (comma)")
            yield Input(placeholder="flood, urban flood, ...", id="crawl-keywords")
        with Horizontal(classes="form-row"):
            yield Label("Max results")
            yield Input(value="100", id="crawl-max")
        with Horizontal(classes="form-row"):
            yield Label("Output dir")
            yield Input(value="data/raw", id="crawl-output")
        with Horizontal(classes="form-row"):
            yield Label("Crawler type")
            yield Select(
                [
                    ("Image Spider", "image_spider"),
                    ("Multi-Source API", "multi_source"),
                    ("Video (yt-dlp)", "video"),
                ],
                id="crawl-type",
                value="image_spider",
            )

        with Vertical(id="multi-source-group"):
            yield Static("Select API sources:", classes="section-title")
            yield SelectionList[str](
                ("Unsplash", "unsplash", True),
                ("Pexels", "pexels", True),
                ("Flickr", "flickr", True),
                ("Wikimedia", "wikimedia", True),
                ("NASA", "nasa", True),
                id="source-selection",
            )

        with Horizontal(classes="action-bar"):
            yield Button("Start Crawl", id="btn-crawl", variant="primary")

        yield Static("Output", classes="section-title")
        yield RichLog(id="crawl-log", highlight=True, markup=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        group = self.query_one("#multi-source-group")
        if event.value == "multi_source":
            group.add_class("visible")
        else:
            group.remove_class("visible")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-crawl":
            self._run_crawl()

    @work(thread=True, exclusive=True, group="crawl")
    def _run_crawl(self) -> None:
        log = self.query_one("#crawl-log", RichLog)
        btn = self.query_one("#btn-crawl", Button)
        self.app.call_from_thread(btn.set_class, True, "-disabled")

        kw_raw = self.query_one("#crawl-keywords", Input).value.strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        if not keywords:
            self.app.call_from_thread(log.write, "[red]Please enter keywords.[/red]")
            self.app.call_from_thread(btn.set_class, False, "-disabled")
            return

        max_results = int(self.query_one("#crawl-max", Input).value or "100")
        output_dir = Path(self.query_one("#crawl-output", Input).value or "data/raw")
        crawler_type = self.query_one("#crawl-type", Select).value

        cfg: Config = self.app.config  # type: ignore[attr-defined]

        try:
            if crawler_type == "image_spider":
                self.app.call_from_thread(log.write, "Starting ImageSpider …")
                crawler = ImageSpider(config=cfg.crawler, output_dir=output_dir)
                paths = crawler.crawl(keywords, max_results=max_results)
                self.app.call_from_thread(
                    log.write, f"[green]Done — {len(paths)} images downloaded.[/green]"
                )

            elif crawler_type == "multi_source":
                sel_list = self.query_one("#source-selection", SelectionList)
                sources = [str(v) for v in sel_list.selected]
                self.app.call_from_thread(
                    log.write, f"Starting MultiSourceCrawler (sources={sources}) …"
                )
                crawler = MultiSourceCrawler(
                    config=cfg, output_dir=output_dir, sources=sources
                )
                paths = crawler.crawl(keywords, max_results=max_results)
                self.app.call_from_thread(
                    log.write, f"[green]Done — {len(paths)} images downloaded.[/green]"
                )

            elif crawler_type == "video":
                self.app.call_from_thread(log.write, "Starting VideoCrawler …")
                crawler = VideoCrawler(config=cfg.crawler, output_dir=output_dir)
                paths = crawler.crawl(keywords, max_results=max_results)
                self.app.call_from_thread(
                    log.write, f"[green]Done — {len(paths)} videos downloaded.[/green]"
                )

        except Exception as exc:
            self.app.call_from_thread(log.write, f"[red]Error: {exc}[/red]")
        finally:
            self.app.call_from_thread(btn.set_class, False, "-disabled")


class ProcessPane(VerticalScroll):
    """Image / video cleaning pipeline."""

    DEFAULT_CSS = """
    ProcessPane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Process", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Input dir")
            yield Input(value="data/raw", id="proc-input")
        with Horizontal(classes="form-row"):
            yield Label("Output dir")
            yield Input(value="data/processed", id="proc-output")

        with Horizontal(classes="switch-row"):
            yield Label("Extract video frames")
            yield Switch(value=True, id="proc-frames")
        with Horizontal(classes="switch-row"):
            yield Label("Remove blurry images")
            yield Switch(value=True, id="proc-blur")
        with Horizontal(classes="switch-row"):
            yield Label("Deduplicate")
            yield Switch(value=True, id="proc-dedup")

        with Horizontal(classes="action-bar"):
            yield Button("Run Pipeline", id="btn-process", variant="primary")

        yield Static("Output", classes="section-title")
        yield RichLog(id="process-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-process":
            self._run_pipeline()

    @work(thread=True, exclusive=True, group="process")
    def _run_pipeline(self) -> None:
        log = self.query_one("#process-log", RichLog)
        btn = self.query_one("#btn-process", Button)
        self.app.call_from_thread(btn.set_class, True, "-disabled")

        input_dir = Path(self.query_one("#proc-input", Input).value or "data/raw")
        output_dir = Path(self.query_one("#proc-output", Input).value or "data/processed")
        extract_frames = self.query_one("#proc-frames", Switch).value
        remove_blur = self.query_one("#proc-blur", Switch).value
        deduplicate = self.query_one("#proc-dedup", Switch).value

        cfg: Config = self.app.config  # type: ignore[attr-defined]

        try:
            self.app.call_from_thread(log.write, "Running cleaning pipeline …")
            pipeline = CleaningPipeline(config=cfg.processor)
            stats = pipeline.run(
                input_dir=input_dir,
                output_dir=output_dir,
                extract_video_frames=extract_frames,
                remove_blur=remove_blur,
                deduplicate=deduplicate,
            )
            self.app.call_from_thread(
                log.write, f"[green]Pipeline complete.[/green]"
            )
            for section in ("images", "videos", "total"):
                if section in stats:
                    self.app.call_from_thread(
                        log.write, f"  {section}: {stats[section]}"
                    )
        except Exception as exc:
            self.app.call_from_thread(log.write, f"[red]Error: {exc}[/red]")
        finally:
            self.app.call_from_thread(btn.set_class, False, "-disabled")


class SynthesizePane(VerticalScroll):
    """Generate synthetic flood images via Gemini API."""

    DEFAULT_CSS = """
    SynthesizePane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Synthesize", classes="section-title")

        yield Static("Prompts (one per line):")
        yield TextArea(
            "a flooded street in a tropical city\n"
            "aerial view of river overflowing into farmland\n",
            id="synth-prompts",
        )

        with Horizontal(classes="form-row"):
            yield Label("Prompts file")
            yield Input(placeholder="(optional) path/to/prompts.json", id="synth-file")
        with Horizontal(classes="form-row"):
            yield Label("Output dir")
            yield Input(value="data/synthetic", id="synth-output")

        with Horizontal(classes="switch-row"):
            yield Label("Enhance prompts")
            yield Switch(value=True, id="synth-enhance")

        with Horizontal(classes="action-bar"):
            yield Button("Generate", id="btn-synth", variant="primary")

        yield Static("Output", classes="section-title")
        yield RichLog(id="synth-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-synth":
            self._run_synth()

    @work(thread=True, exclusive=True, group="synthesize")
    def _run_synth(self) -> None:
        log = self.query_one("#synth-log", RichLog)
        btn = self.query_one("#btn-synth", Button)
        self.app.call_from_thread(btn.set_class, True, "-disabled")

        output_dir = Path(self.query_one("#synth-output", Input).value or "data/synthetic")
        enhance = self.query_one("#synth-enhance", Switch).value
        prompts_file = self.query_one("#synth-file", Input).value.strip()

        cfg: Config = self.app.config  # type: ignore[attr-defined]

        try:
            from floodsense.synthesizers.img_gen_models_client import ImageGenClient
            client = ImageGenClient(config=cfg.synthesizer)

            if prompts_file:
                self.app.call_from_thread(
                    log.write, f"Loading prompts from {prompts_file} …"
                )
                paths = client.generate_from_file(
                    Path(prompts_file), output_dir, enhance_prompt=enhance
                )
            else:
                text = self.query_one("#synth-prompts", TextArea).text
                prompts = [l.strip() for l in text.splitlines() if l.strip()]
                if not prompts:
                    self.app.call_from_thread(
                        log.write, "[red]No prompts provided.[/red]"
                    )
                    self.app.call_from_thread(btn.set_class, False, "-disabled")
                    return
                self.app.call_from_thread(
                    log.write, f"Generating {len(prompts)} images …"
                )
                paths = client.generate_batch(
                    prompts, output_dir, enhance_prompt=enhance
                )

            self.app.call_from_thread(
                log.write, f"[green]Done — {len(paths)} images generated.[/green]"
            )
        except ImportError as exc:
            self.app.call_from_thread(
                log.write,
                f"[red]Missing dependency: {exc}. "
                f"Install with: pip install google-generativeai[/red]",
            )
        except ValueError as exc:
            self.app.call_from_thread(log.write, f"[red]{exc}[/red]")
        except Exception as exc:
            self.app.call_from_thread(log.write, f"[red]Error: {exc}[/red]")
        finally:
            self.app.call_from_thread(btn.set_class, False, "-disabled")


class ValidatePane(VerticalScroll):
    """Content-validate images against keywords."""

    DEFAULT_CSS = """
    ValidatePane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Validate", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Image dir")
            yield Input(value="data/processed", id="val-dir")
        with Horizontal(classes="form-row"):
            yield Label("Keywords (comma)")
            yield Input(placeholder="flood, water, ...", id="val-keywords")

        with Horizontal(classes="switch-row"):
            yield Label("Enable heuristic filter")
            yield Switch(value=True, id="val-heuristic")
        with Horizontal(classes="switch-row"):
            yield Label("Enable CLIP filter")
            yield Switch(value=True, id="val-clip")

        with Horizontal(classes="action-bar"):
            yield Button("Validate", id="btn-validate", variant="primary")

        yield Static("Output", classes="section-title")
        yield RichLog(id="validate-log", highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-validate":
            self._run_validate()

    @work(thread=True, exclusive=True, group="validate")
    def _run_validate(self) -> None:
        log = self.query_one("#validate-log", RichLog)
        btn = self.query_one("#btn-validate", Button)
        self.app.call_from_thread(btn.set_class, True, "-disabled")

        image_dir = Path(self.query_one("#val-dir", Input).value or "data/processed")
        kw_raw = self.query_one("#val-keywords", Input).value.strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        enable_heuristic = self.query_one("#val-heuristic", Switch).value
        enable_clip = self.query_one("#val-clip", Switch).value

        try:
            validator = ImageValidator(
                enable_heuristic=enable_heuristic,
                enable_clip=enable_clip,
            )

            images = list(FileUtils.iter_images(image_dir))
            if not images:
                self.app.call_from_thread(
                    log.write, "[yellow]No images found.[/yellow]"
                )
                self.app.call_from_thread(btn.set_class, False, "-disabled")
                return

            self.app.call_from_thread(
                log.write, f"Validating {len(images)} images …"
            )

            valid_count = 0
            invalid_count = 0
            for img_path in images:
                result = validator.validate(img_path, keywords=keywords)
                if result:
                    valid_count += 1
                else:
                    invalid_count += 1

            self.app.call_from_thread(
                log.write,
                f"[green]Done — {valid_count} valid, {invalid_count} invalid "
                f"out of {len(images)} images.[/green]",
            )
        except Exception as exc:
            self.app.call_from_thread(log.write, f"[red]Error: {exc}[/red]")
        finally:
            self.app.call_from_thread(btn.set_class, False, "-disabled")


class ConfigPane(VerticalScroll):
    """View and edit YAML configuration."""

    DEFAULT_CSS = """
    ConfigPane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Config", classes="section-title")

        with Horizontal(classes="form-row"):
            yield Label("Config file")
            yield Input(value="config.yaml", id="cfg-path")

        yield TextArea("", language="yaml", id="cfg-editor")

        with Horizontal(classes="action-bar"):
            yield Button("Reload", id="btn-cfg-reload", variant="default")
            yield Button("Apply", id="btn-cfg-apply", variant="primary")
            yield Button("Save", id="btn-cfg-save", variant="warning")

        yield RichLog(id="cfg-log", highlight=True, markup=True, max_lines=200)

    def on_mount(self) -> None:
        self._load_config_text()

    def _load_config_text(self) -> None:
        cfg: Config = self.app.config  # type: ignore[attr-defined]
        text = yaml.dump(
            cfg.model_dump(), default_flow_style=False, allow_unicode=True
        )
        self.query_one("#cfg-editor", TextArea).text = text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#cfg-log", RichLog)

        if event.button.id == "btn-cfg-reload":
            cfg_path = self.query_one("#cfg-path", Input).value.strip()
            try:
                p = Path(cfg_path) if cfg_path else None
                self.app.config = Config.load(p)  # type: ignore[attr-defined]
                self._load_config_text()
                log.write("[green]Config reloaded.[/green]")
            except Exception as exc:
                log.write(f"[red]Reload failed: {exc}[/red]")

        elif event.button.id == "btn-cfg-apply":
            raw = self.query_one("#cfg-editor", TextArea).text
            try:
                parsed = yaml.safe_load(raw) or {}
                self.app.config = Config(**parsed)  # type: ignore[attr-defined]
                log.write("[green]Config applied.[/green]")
            except Exception as exc:
                log.write(f"[red]Apply failed: {exc}[/red]")

        elif event.button.id == "btn-cfg-save":
            cfg_path = self.query_one("#cfg-path", Input).value.strip() or "config.yaml"
            try:
                self.app.config.save(Path(cfg_path))  # type: ignore[attr-defined]
                log.write(f"[green]Config saved to {cfg_path}.[/green]")
            except Exception as exc:
                log.write(f"[red]Save failed: {exc}[/red]")


class LogPane(Vertical):
    """Global log viewer (loguru sink)."""

    DEFAULT_CSS = """
    LogPane { padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Logs", classes="section-title")
        with Horizontal(classes="action-bar"):
            yield Button("Clear", id="btn-log-clear", variant="default")
        yield RichLog(id="global-log", highlight=True, markup=True, max_lines=1000)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-log-clear":
            self.query_one("#global-log", RichLog).clear()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  APPLICATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

PANES = [
    ("dashboard", "Dashboard", "d"),
    ("crawl", "Crawl", "c"),
    ("process", "Process", "p"),
    ("synthesize", "Synthesize", "s"),
    ("validate", "Validate", "v"),
    ("config", "Config", "o"),
    ("logs", "Logs", "l"),
]


class FloodSenseApp(App):
    """FloodSense Data Engine TUI."""

    TITLE = "FloodSense Data Engine"
    CSS = CSS

    BINDINGS = [
        Binding("d", "switch_pane('dashboard')", "Dashboard", show=True),
        Binding("c", "switch_pane('crawl')", "Crawl", show=True),
        Binding("p", "switch_pane('process')", "Process", show=True),
        Binding("s", "switch_pane('synthesize')", "Synthesize", show=True),
        Binding("v", "switch_pane('validate')", "Validate", show=True),
        Binding("o", "switch_pane('config')", "Config", show=True),
        Binding("l", "switch_pane('logs')", "Logs", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    current_pane: reactive[str] = reactive("dashboard")

    def __init__(self) -> None:
        super().__init__()
        self.config: Config = Config.load()
        self.config.ensure_directories()
        self._loguru_sink_id: int | None = None

    # ---- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            # Sidebar
            with Vertical(id="sidebar"):
                for pane_id, label, key in PANES:
                    yield Button(
                        f"[{key.upper()}] {label}",
                        id=f"nav-{pane_id}",
                        classes="nav-active" if pane_id == "dashboard" else "",
                    )

            # Main content
            with ContentSwitcher(initial="dashboard", id="content-area"):
                yield DashboardPane(id="dashboard")
                yield CrawlPane(id="crawl")
                yield ProcessPane(id="process")
                yield SynthesizePane(id="synthesize")
                yield ValidatePane(id="validate")
                yield ConfigPane(id="config")
                yield LogPane(id="logs")

        yield Footer()

    # ---- lifecycle -------------------------------------------------------

    def on_mount(self) -> None:
        self._install_loguru_sink()

    def on_unmount(self) -> None:
        if self._loguru_sink_id is not None:
            try:
                logger.remove(self._loguru_sink_id)
            except ValueError:
                pass

    def _install_loguru_sink(self) -> None:
        """Route loguru messages into the Logs pane RichLog widget."""

        def _sink(message: Any) -> None:
            record = message.record
            lvl = record["level"].name
            color_map = {
                "DEBUG": "dim",
                "INFO": "cyan",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold red",
            }
            style = color_map.get(lvl, "")
            text = str(message).rstrip()
            markup = f"[{style}]{text}[/{style}]" if style else text
            try:
                log_widget = self.query_one("#global-log", RichLog)
                self.call_from_thread(log_widget.write, markup)
            except NoMatches:
                pass

        self._loguru_sink_id = logger.add(
            _sink, level="DEBUG", format="{time:HH:mm:ss} | {level:<8} | {message}"
        )

    # ---- navigation ------------------------------------------------------

    def action_switch_pane(self, pane_id: str) -> None:
        self.current_pane = pane_id

    def watch_current_pane(self, pane_id: str) -> None:
        try:
            self.query_one("#content-area", ContentSwitcher).current = pane_id
        except NoMatches:
            return

        # Update sidebar highlight
        for pid, _, _ in PANES:
            try:
                btn = self.query_one(f"#nav-{pid}", Button)
                if pid == pane_id:
                    btn.add_class("nav-active")
                else:
                    btn.remove_class("nav-active")
            except NoMatches:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("nav-"):
            pane_id = btn_id.removeprefix("nav-")
            self.current_pane = pane_id


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    app = FloodSenseApp()
    app.run()
