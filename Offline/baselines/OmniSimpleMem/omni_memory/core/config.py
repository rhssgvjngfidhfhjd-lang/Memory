"""Configuration objects required by the extracted OmniSimpleMem package.

The upstream OmniSimpleMem sources reference this module throughout the package,
but it was absent from the extracted directory.  The dataclasses below restore
the documented public API without introducing a second runtime configuration:
the Offline adapter overwrites model and endpoint values from configs/defaults.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    api_key: str | None = None
    api_base_url: str | None = None
    remote: bool = False
    visual_embedding_model: str = "openai/clip-vit-base-patch32"
    visual_embedding_dim: int = 512


@dataclass
class RetrievalConfig:
    default_top_k: int = 10
    enable_hybrid_search: bool = True
    enable_graph_traversal: bool = True
    auto_expand_threshold: float = 0.8
    max_expanded_items: int = 5


@dataclass
class StorageConfig:
    base_dir: str = "./omni_memory_data"
    cold_storage_dir: str = "./omni_memory_data/cold_storage"
    index_dir: str = "./omni_memory_data/index"
    use_s3: bool = False
    s3_bucket: str = ""
    s3_prefix: str = ""
    organize_by_date: bool = True
    organize_by_modality: bool = True
    auto_cleanup_enabled: bool = False


@dataclass
class LLMConfig:
    summary_model: str = "gpt-4o-mini"
    query_model: str = "gpt-4o-mini"
    caption_model: str = "gpt-4o-mini"
    whisper_model: str = "whisper-1"
    temperature: float = 0.0
    max_tokens: int = 1000
    api_key: str | None = None
    api_base_url: str | None = None

    @property
    def model(self) -> str:
        return self.summary_model

    @model.setter
    def model(self, value: str) -> None:
        self.summary_model = value
        self.query_model = value
        self.caption_model = value


@dataclass
class EventConfig:
    auto_create_events: bool = True
    event_time_window_seconds: float = 300.0
    summarize_on_close: bool = True
    max_maus_for_summary: int = 50


@dataclass
class EntropyTriggerConfig:
    visual_similarity_threshold_high: float = 0.9
    visual_similarity_threshold_low: float = 0.6
    visual_encoder: str = "clip"
    visual_model_name: str = "openai/clip-vit-base-patch32"
    enable_visual_trigger: bool = True
    audio_energy_threshold: float = 0.01
    audio_vad_threshold: float = 0.5
    audio_min_speech_duration_ms: int = 300
    enable_audio_trigger: bool = True


@dataclass
class RouterConfig:
    router_mode: str = "off"
    benchmark_safe: bool = True
    gini_threshold: float = 0.35
    top1_threshold: float = 0.7
    gap_threshold: float = 0.2
    episodic_margin: float = 0.05
    close_margin: float = 0.03
    shadow_mode: bool = True


@dataclass
class OmniMemoryConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    event: EventConfig = field(default_factory=EventConfig)
    entropy_trigger: EntropyTriggerConfig = field(default_factory=EntropyTriggerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_self_evolution: bool = False
    evolution: Any = None

    def __post_init__(self) -> None:
        if self.llm.api_key is None:
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
        if self.llm.api_base_url is None:
            self.llm.api_base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    @classmethod
    def create_default(cls) -> "OmniMemoryConfig":
        return cls()

    def set_unified_model(self, model: str) -> "OmniMemoryConfig":
        self.llm.model = model
        return self

    def enable_evolution(self) -> "OmniMemoryConfig":
        self.enable_self_evolution = True
        if self.evolution is None:
            try:
                from omni_memory.evolution import EvolutionConfig

                self.evolution = EvolutionConfig()
            except ImportError:
                self.evolution = {}
        return self

    def ensure_directories(self) -> None:
        for value in (
            self.storage.base_dir,
            self.storage.cold_storage_dir,
            self.storage.index_dir,
        ):
            Path(value).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OmniMemoryConfig":
        data = dict(value)
        return cls(
            embedding=EmbeddingConfig(**data.pop("embedding", {})),
            retrieval=RetrievalConfig(**data.pop("retrieval", {})),
            storage=StorageConfig(**data.pop("storage", {})),
            llm=LLMConfig(**data.pop("llm", {})),
            event=EventConfig(**data.pop("event", {})),
            entropy_trigger=EntropyTriggerConfig(**data.pop("entropy_trigger", {})),
            router=RouterConfig(**data.pop("router", {})),
            **data,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str) -> "OmniMemoryConfig":
        return cls.from_dict(json.loads(value))

    def save_to_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "OmniMemoryConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
