"""Building the prompt of a turn: system prompt, memories, history and its summary."""

from collections.abc import Sequence
from typing import Protocol

from zero import clock
from zero.config import Config
from zero.core.attachments import prepare_for_model
from zero.core.compaction import compact
from zero.core.context import build_system_prompt, keep_from, to_history
from zero.core.events import EventBus
from zero.protocol.messages import ClientInfo, TextPart
from zero.providers.base import LLMProvider, Message
from zero.store.memory import Memory
from zero.store.messages import MessagesRepo, Source
from zero.store.summaries import SummariesRepo

HISTORY_FETCH_LIMIT = 500
PROMPT_MEMORIES = 10  # how many memories go into the system prompt
SUMMARY_RESERVE_TOKENS = 1500  # room kept for the summary once history has to be cut


class MemorySource(Protocol):
    """Gives the memories for the system prompt."""

    async def top(self, n: int = 10) -> Sequence[Memory]: ...


class PromptBuilder:
    """Puts together what one turn sends to the model.

    The history that fits the model's budget goes in as messages (the
    provider's ``history_token_budget``, else the agent default). If it
    does not all fit and ``summaries`` is given, the dropped start of the
    conversation is summarized (see ``compact``) and the summary joins the
    system prompt; some of the budget is kept free for it.
    """

    def __init__(
        self,
        *,
        session_id: str,
        client: ClientInfo | None,
        source: Source,
        config: Config,
        provider: LLMProvider,
        messages: MessagesRepo,
        events: EventBus,
        memory: MemorySource | None = None,
        summaries: SummariesRepo | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._client = client
        self._source: Source = source
        self._config = config
        self._provider = provider
        self._messages = messages
        self._events = events
        self._memory = memory
        self._summaries = summaries
        self._agent_name = agent_name

    async def build(self) -> list[Message]:
        """The system prompt followed by the history, ready for the provider."""
        memories = await self._memory.top(PROMPT_MEMORIES) if self._memory is not None else []
        stored = await self._messages.list(self._session_id, limit=HISTORY_FETCH_LIMIT)
        budget = self._config.provider.history_token_budget or (
            self._config.agent.history_token_budget
        )
        start = keep_from(stored, budget)
        summary = None
        if start > 0 and self._summaries is not None:
            start = keep_from(stored, max(budget - SUMMARY_RESERVE_TOKENS, 0))
            summary = await compact(
                self._session_id,
                stored,
                start,
                summaries=self._summaries,
                provider=self._provider,
                model=self._config.provider.model,
                events=self._events,
            )
        system = build_system_prompt(
            self._client,
            self._source,
            [memory.text for memory in memories],
            now=clock.now(),
            summary=summary,
            agent_name=self._agent_name,
        )
        prompt = [
            Message(role="system", content=[TextPart(text=system)]),
            *to_history(stored[start:]),
        ]
        can_see = self._provider.supports_images(self._config.provider.model)
        return await prepare_for_model(prompt, can_see)
