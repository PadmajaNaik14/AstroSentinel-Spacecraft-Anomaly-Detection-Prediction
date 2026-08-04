"""Backend settings. Data/artifact paths are deliberately NOT duplicated here
-- they come from ml.config (see ASTROSENTINEL_DATA_RAW_DIR / _ARTIFACTS_DIR
env vars), so there is exactly one place that resolves those paths.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_url: str = "sqlite:///./astrosentinel.db"
    replay_tick_ms: int = 150
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
