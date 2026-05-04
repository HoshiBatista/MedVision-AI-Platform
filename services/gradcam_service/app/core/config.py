from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Root of the Triton model repository; ONNX files are at
    # {model_repo_root}/{model_name}/1/model.onnx
    model_repo_root: str = "/models"

    # Where generated heatmap PNGs are written
    heatmap_output_dir: str = "/data/heatmaps"

    # Overlay transparency (0.0 = only original, 1.0 = only heatmap)
    heatmap_alpha: float = 0.5

    # ONNX Runtime providers in priority order
    ort_providers: list[str] = ["CPUExecutionProvider"]

    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True


settings = Settings()
