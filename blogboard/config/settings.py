from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    API_KEY: str = Field(
        default="",
        validation_alias=AliasChoices(
            "API_KEY",
            "api_key",
            "GROQ_API_KEY",
            "groq_api_key",
            "LLM__API_KEY",
            "llm__api_key",
        ),
    )
    MODEL_NAME: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices(
            "MODEL_NAME",
            "model_name",
            "LLM__MODEL_NAME",
            "llm__model_name",
        ),
    )
    TEMPERATURE: float = 1.0


class TagSettings(BaseModel):
    ml: dict[str, str] = {"label": "Machine Learning", "shortLabel": "ML"}
    dl: dict[str, str] = {"label": "Deep Learning", "shortLabel": "DL"}
    statistics: dict[str, str] = {"label": "Statistics for AI", "shortLabel": "Stats"}
    nlp: dict[str, str] = {"label": "Natural Language Processing", "shortLabel": "NLP"}
    cv: dict[str, str] = {"label": "Computer Vision", "shortLabel": "CV"}
    genai: dict[str, str] = {"label": "Generative AI", "shortLabel": "Gen AI"}
    ainews: dict[str, str] = {"label": "AI News", "shortLabel": "AI News"}


class R2Settings(BaseModel):
    ACCOUNT_ID: str = ""
    ACCESS_KEY_ID: str = ""
    SECRET_ACCESS_KEY: str = ""
    BUCKET_NAME: str = ""


class ContentAPISettings(BaseModel):
    TAVILY_API_KEY: str = ""
    GUARDIAN_API_KEY: str = ""
    UNSPLASH_API_KEY: str = ""


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tags: TagSettings = Field(default_factory=TagSettings)
    r2: R2Settings = Field(default_factory=R2Settings)
    content: ContentAPISettings = Field(default_factory=ContentAPISettings)
    site_url: str = "http://localhost:8000"
    admin_token: str = ""
    storage_backend: str = "r2"
    local_storage_root: str = "blogboard/web"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    def is_r2_configured(self) -> bool:
        r2 = self.r2
        return bool(
            r2.ACCOUNT_ID.strip()
            and r2.ACCESS_KEY_ID.strip()
            and r2.SECRET_ACCESS_KEY.strip()
            and r2.BUCKET_NAME.strip(" =\"'")
        )

    def is_llm_configured(self) -> bool:
        return bool(self.llm.API_KEY.strip())


app_settings = Settings()
