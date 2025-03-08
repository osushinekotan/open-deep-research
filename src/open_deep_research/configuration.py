import os
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections
   - Provide a concise summary of the report"""


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"


class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    report_structure: str = DEFAULT_REPORT_STRUCTURE  # Defaults to the default report structure
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations

    planner_provider: PlannerProvider = PlannerProvider.OPENAI
    planner_model: str = "gpt-4o"
    planner_model_config: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "max_tokens": 8192,
            "temperature": 0.0,
        }
    )

    writer_provider: WriterProvider = WriterProvider.OPENAI
    writer_model: str = "gpt-4o"
    writer_model_config: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "max_tokens": 8192,
            "temperature": 0.0,
        }
    )

    search_api: SearchAPI = SearchAPI.TAVILY  # Default to TAVILY
    search_api_config: Optional[dict[str, Any]] = field(
        default_factory=lambda: {
            "max_results": 5,
            "include_raw_content": False,
        }
    )

    language: str = "japanese"

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name)) for f in fields(cls) if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
