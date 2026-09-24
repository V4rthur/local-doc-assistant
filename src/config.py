"""Loads config.yaml and validates it with pydantic."""
from pathlib import Path
from typing import List
import yaml
from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    generator: str
    grader: str
    embedder: str
    reranker: str


class IngestionConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    supported_extensions: List[str]
    extract_tables: bool
    script_normalization: str


class ChunkingConfig(BaseModel):
    strategy: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    article_patterns: List[str]


class IndexingConfig(BaseModel):
    chroma_dir: str
    bm25_path: str
    collection_name: str
    incremental: bool


class RetrievalConfig(BaseModel):
    top_k_vector: int = Field(gt=0)
    top_k_bm25: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    top_k_after_fusion: int = Field(gt=0)
    top_k_after_rerank: int = Field(gt=0)


class AgentConfig(BaseModel):
    max_query_rewrites: int = Field(ge=0)
    max_generation_retries: int = Field(ge=0)
    min_relevant_docs: int = Field(gt=0)


class AuditConfig(BaseModel):
    log_path: str
    log_queries: bool
    log_sources: bool


class UIConfig(BaseModel):
    page_title: str
    show_sources: bool
    show_retry_count: bool
    enable_flag_button: bool


class Config(BaseModel):
    models: ModelsConfig
    ingestion: IngestionConfig
    chunking: ChunkingConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    audit: AuditConfig
    ui: UIConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# Singleton — import this everywhere
CFG = load_config()