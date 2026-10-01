from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TreeModelSettings(BaseModel):
    """Model 1 (banana tree classification) configuration.

    Grouped the way the spec wants so Model 2/3 can each get their own
    XxxModelSettings built from Settings the same way, without touching this
    one. See Settings.tree below.
    """

    enabled: bool
    weights_path: str
    model_version: str
    imgsz: int
    positive_class: str
    uncertain_threshold: float
    display_names: dict[str, str]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- app ---
    app_env: Literal["development", "production", "test"] = "development"
    app_name: str = "KeraAI API"
    api_v1_prefix: str = "/api/v1"
    enable_docs: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    trust_cloudflare_headers: bool = False
    signing_secret: str
    signed_url_ttl_seconds: int = 3600

    # --- database ---
    db_host: str = "localhost"
    db_port: int | None = None
    db_name: str = "KeraAI"
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_trusted_connection: bool = True
    db_user: str = "kera_app"
    db_password: str = ""
    db_trust_server_cert: bool = True
    db_pool_size: int = 5
    db_echo: bool = False

    # --- storage & uploads ---
    data_root: str
    max_upload_mb: int = 10
    min_image_side_px: int = 64
    max_image_pixels: int = 40_000_000
    store_max_side_px: int = 2048
    thumbnail_side_px: int = 320
    allowed_image_formats: list[str] = Field(default_factory=lambda: ["JPEG", "PNG", "WEBP"])

    # --- inference ---
    inference_device: str = "auto"
    rate_limit_predict: str = "10/minute"
    rate_limit_default: str = "120/minute"

    # --- model 1: tree classification ---
    tree_enabled: bool = True
    tree_weights_path: str = "weights/tree_cls_v1.pt"
    tree_model_version: str = "tree_cls_v1"
    tree_imgsz: int = 224
    tree_positive_class: str = "banana_tree"
    tree_uncertain_threshold: float = 0.60
    tree_display_names: dict[str, str] = Field(
        default_factory=lambda: {
            "banana_tree": "Banana tree",
            "non_banana": "Not a banana tree",
        }
    )

    @field_validator("db_port", mode="before")
    @classmethod
    def _blank_db_port_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("signing_secret")
    @classmethod
    def _signing_secret_min_length(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError("SIGNING_SECRET must be at least 32 characters")
        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def _prefix_starts_with_slash(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("API_V1_PREFIX must start with '/'")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _trusted_connection_or_credentials(self) -> Settings:
        if not self.db_trusted_connection and not self.db_user:
            raise ValueError("DB_USER is required when DB_TRUSTED_CONNECTION=false")
        return self

    @property
    def tree(self) -> TreeModelSettings:
        return TreeModelSettings(
            enabled=self.tree_enabled,
            weights_path=self.tree_weights_path,
            model_version=self.tree_model_version,
            imgsz=self.tree_imgsz,
            positive_class=self.tree_positive_class,
            uncertain_threshold=self.tree_uncertain_threshold,
            display_names=self.tree_display_names,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
