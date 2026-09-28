from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from neurapress.config.settings import app_settings
from neurapress.tools import GuardianSearchTool, TavilySearchTool


class LLMAgentService:
    """
    A core service class responsible for managing LLM connections and
    orchestrating tool-binding for multi-agent workflows.

    This class adheres to OOP principles, allowing easy instantiation
    of different base LLMs and assembling specialized agents as needed.
    """

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        """
        Initializes the LLM Service with specific model configurations.

        Args:
            model_name (Optional[str]): The Groq model variant to use. Defaults to settings.
            temperature (Optional[float]): The inference temperature. Defaults to settings.
        """
        self.model_name = model_name or app_settings.llm.MODEL_NAME
        self.temperature = temperature if temperature is not None else app_settings.llm.TEMPERATURE
        self.api_key = app_settings.llm.API_KEY

        # Statically instantiate the main LLM client for reuse across agents created by this service
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> ChatGroq:
        """
        Core logic to instantiate the connection with Groq.

        Returns:
             ChatGroq: An active, authenticated Groq LLM instance.
        """
        return ChatGroq(model=self.model_name, temperature=self.temperature, api_key=self.api_key)

    def invoke(self, prompt: str, attempts: int = 4):
        """
        Invoke the LLM with exponential-backoff retries for transient errors
        (rate limits, timeouts, 5xx). If all attempts on the primary model
        fail, falls back to a lighter model with separate rate-limit buckets
        so scheduled publishing survives provider throttling.
        """
        import time

        models = [self.model_name]
        fallback = "openai/gpt-oss-20b"
        if self.model_name != fallback:
            models.append(fallback)

        last: Exception | None = None
        for model in models:
            for attempt in range(1, attempts + 1):
                try:
                    llm = self.llm if model == self.model_name else self._llm_for(model)
                    return llm.invoke(prompt)
                except Exception as e:  # noqa: BLE001 - retry any transient provider error
                    last = e
                    wait = min(60, 5 * (2 ** (attempt - 1)))
                    print(
                        f"  [RETRY] LLM call failed (model={model}, attempt "
                        f"{attempt}/{attempts}): {type(e).__name__}: {str(e)[:150]} "
                        f"— waiting {wait}s"
                    )
                    time.sleep(wait)
            if len(models) > 1 and model == models[0]:
                print(f"  [FALLBACK] Primary model exhausted — switching to {models[1]}")
        assert last is not None
        raise last

    def _llm_for(self, model_name: str) -> ChatGroq:
        return ChatGroq(model=model_name, temperature=self.temperature, api_key=self.api_key)

    def get_news_agent(self, system_prompt: str | None = None):
        """
        Constructs a specialized News Research Agent equipped with our web-search tools.

        Args:
            system_prompt (Optional[str]): Context or behavioral instructions injected into the StateGraph.

        Returns:
            CompiledGraph: A runnable LangGraph ReAct agent pre-equipped with Tavily and Guardian search tools.
        """
        # Step 1: Initialize the tools designed for the News Agent
        news_tools: list[BaseTool] = [TavilySearchTool(), GuardianSearchTool()]

        # Step 2: Bind the tools and LLM using LangGraph's prebuilt ReAct orchestrator
        # (langgraph 1.x renamed state_modifier -> prompt)
        agent = create_react_agent(model=self.llm, tools=news_tools, prompt=system_prompt)
        return agent

    def get_custom_agent(self, tools: list[BaseTool], system_prompt: str | None = None):
        """
        A flexible builder method to create a custom agent given an arbitrary set of OOP-based tools.

        Args:
            tools (List[BaseTool]): A list of initialized classes inheriting from BaseTool.
            system_prompt (Optional[str]): Context or behavioral instructions injected into the agent.

        Returns:
            CompiledGraph: A customizable runnable LangGraph ReAct agent.
        """
        agent = create_react_agent(model=self.llm, tools=tools, prompt=system_prompt)
        return agent
