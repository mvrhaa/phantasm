"""Building the prompt of a turn: system prompt, memories, history and its summary."""

from collections.abc import Sequence
from typing import Protocol

from zero import clock
from zero.config import Config
from zero.core.attachments import prepare_for_model
from zero.core.cancellation import CancelToken
from zero.core.compaction import compact
from zero.core.context import (
    Computer,
    build_system_prompt,
    clear_old_results,
    keep_from,
    to_history,
)
from zero.core.events import EventBus
from zero.core.model_call import UsageRecorder
from zero.protocol.messages import ClientInfo, TextPart
from zero.providers.base import LLMProvider, Message
from zero.providers.registry import small_model
from zero.store.journal import JournalRepo, note_time
from zero.store.memory import Memory
from zero.store.messages import MessagesRepo, Source, StoredMessage
from zero.store.session_memory import SessionMemoryRepo
from zero.store.summaries import SummariesRepo

HISTORY_FETCH_LIMIT = 500
PROMPT_MEMORIES = 10
JOURNAL_PROMPT_CHARS = 5000
SUMMARY_RESERVE_TOKENS = 1500


def _first_user(stored: list[StoredMessage]) -> int:
    """Index of the first user message; the length of the list if there is none."""
    return next((i for i, message in enumerate(stored) if message.role == "user"), len(stored))


class MemorySource(Protocol):
    """Gives the memories for the system prompt."""

    async def top(self, n: int = 10) -> Sequence[Memory]: ...


class PromptBuilder:
    """Puts together what one turn sends to the model.

    The history that fits the model's budget goes in as messages (the
    provider's ``history_token_budget``, else the agent default). If it
    does not all fit and ``summaries`` is given, the dropped start of the
    conversation is summarized (see ``compact``) and the summary joins the
    system prompt; some of the budget is kept free for it. Only the latest
    ``HISTORY_FETCH_LIMIT`` messages are read; what lies before them is
    summarized the same way, and the window starts at a user message.
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
        session_memory: SessionMemoryRepo | None = None,
        journal: JournalRepo | None = None,
        summaries: SummariesRepo | None = None,
        agent_name: str | None = None,
        usage: UsageRecorder | None = None,
        cancel: CancelToken | None = None,
        command: str | None = None,
        model: str | None = None,
        tools: bool = True,
    ) -> None:
        self._session_id = session_id
        self._client = client
        self._source: Source = source
        self._config = config
        self._provider = provider
        self._messages = messages
        self._events = events
        self._memory = memory
        self._session_memory = session_memory
        self._journal = journal
        self._summaries = summaries
        self._agent_name = agent_name
        self._usage = usage
        self._cancel = cancel
        self._command = command
        self._tools = tools
        self._model = model or config.provider.model

    async def build(self) -> list[Message]:
        """The system prompt followed by the history, ready for the provider."""
        memories = await self._memory.top(PROMPT_MEMORIES) if self._memory is not None else []
        own = (
            await self._session_memory.list(self._session_id)
            if self._session_memory is not None
            else []
        )
        notes = (
            await self._journal.newest(self._session_id, JOURNAL_PROMPT_CHARS)
            if self._journal is not None and self._agent_name is None
            else []
        )
        stored = await self._messages.list(self._session_id, limit=HISTORY_FETCH_LIMIT)
        older: list[StoredMessage] = []
        if len(stored) >= HISTORY_FETCH_LIMIT:
            first = _first_user(stored)
            if self._summaries is not None:
                before = await self._messages.list(
                    self._session_id, before=stored[0].id, limit=HISTORY_FETCH_LIMIT
                )
                older = [*before, *stored[:first]]
            stored = stored[first:]
        budget = self._config.provider.history_token_budget or (
            self._config.agent.history_token_budget
        )
        shown = clear_old_results(stored, budget)
        start = keep_from(shown, budget)
        summary = None
        if (start > 0 or older) and self._summaries is not None:
            start = keep_from(shown, max(budget - SUMMARY_RESERVE_TOKENS, 0))
            summary = await compact(
                self._session_id,
                [*older, *stored],
                len(older) + start,
                summaries=self._summaries,
                provider=self._provider,
                model=small_model(self._config.provider),
                events=self._events,
                usage=self._usage,
                cancel=self._cancel,
            )
        system = build_system_prompt(
            self._client,
            self._source,
            [memory.text for memory in memories],
            session_memories=[f"[{entry.id}] {entry.text}" for entry in own],
            journal=[f"[{note_time(note.created_at)}] {note.text}" for note in notes],
            now=clock.now(),
            summary=summary,
            agent_name=self._agent_name,
            computer=Computer.detect(self._config.permissions.allowed_roots),
            command=self._command,
            tools=self._tools,
        )
        prompt = [
            Message(role="system", content=[TextPart(text=system)]),
            *to_history(shown[start:]),
        ]
        can_see = self._provider.supports_images(self._model)
        return await prepare_for_model(prompt, can_see)
