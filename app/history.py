"""Completed database context reconstruction and legacy conversation reservations."""

from collections import OrderedDict
from threading import RLock
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.config import Settings


class UnknownConversation(Exception):
    """The supplied conversation ID is absent or has been evicted."""


class InputBudgetExceeded(Exception):
    """System and current user messages exceed the available input budget."""


class ConversationCapacityExceeded(Exception):
    """Every conversation slot is occupied by an active stream."""


class ConversationBusy(Exception):
    """A stream already owns the supplied conversation ID."""


class ConversationStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._conversations: OrderedDict[str, list[BaseMessage]] = OrderedDict()
        self._active: set[str] = set()
        self._lock = RLock()

    def create(self) -> str:
        with self._lock:
            if len(self._conversations) >= self._settings.max_conversations:
                victim = next(
                    (key for key in self._conversations if key not in self._active), None
                )
                if victim is None:
                    raise ConversationCapacityExceeded()
                del self._conversations[victim]
            conversation_id = str(uuid4())
            self._conversations[conversation_id] = []
            return conversation_id

    def create_reserved(self) -> str:
        with self._lock:
            conversation_id = self.create()
            self._active.add(conversation_id)
            return conversation_id

    def reserve(self, conversation_id: str) -> None:
        with self._lock:
            if conversation_id not in self._conversations:
                raise UnknownConversation(conversation_id)
            if conversation_id in self._active:
                raise ConversationBusy(conversation_id)
            self._active.add(conversation_id)

    def release(self, conversation_id: str) -> None:
        with self._lock:
            self._active.discard(conversation_id)

    def get(self, conversation_id: str) -> list[BaseMessage]:
        with self._lock:
            try:
                messages = self._conversations[conversation_id]
            except KeyError as error:
                raise UnknownConversation(conversation_id) from error
            self._conversations.move_to_end(conversation_id)
            return messages.copy()

    def prepare(
        self, conversation_id: str, system: SystemMessage, current: HumanMessage
    ) -> list[BaseMessage]:
        history = self.get(conversation_id)
        limit = (
            self._settings.context_token_budget
            - self._settings.response_token_reserve
        )
        if count_tokens_approximately([system, current]) > limit:
            raise InputBudgetExceeded("System and current message exceed input budget")

        candidate = [system, *history, current]
        while count_tokens_approximately(candidate) > limit:
            del candidate[1:3]
        return candidate

    def commit(
        self, conversation_id: str, user: HumanMessage, assistant: AIMessage
    ) -> None:
        with self._lock:
            self.get(conversation_id)
            messages = self._conversations[conversation_id]
            messages.extend((user, assistant))
            excess = len(messages) - 2 * self._settings.max_turns_per_conversation
            if excess > 0:
                del messages[:excess]


def completed_turns(rows) -> list[list[BaseMessage]]:
    """Rebuild only complete turns with exactly paired tool request/result IDs.

    Failed audit rows stay in the database. A later user row starts a new turn,
    allowing successful requests after a failure to reenter model context.
    """
    from langchain_core.messages import ToolMessage

    groups = []
    for row in rows:
        if row.role == 'user':
            groups.append([row])
        elif groups:
            groups[-1].append(row)

    turns = []
    for group in groups:
        if len(group) < 2 or group[-1].role != 'assistant' or group[-1].tool_calls:
            continue
        user = HumanMessage(content=group[0].content or '')
        final = AIMessage(content=group[-1].content or '')
        if len(group) == 2:
            turns.append([user, final])
            continue
        request = group[1]
        if request.role != 'assistant' or not request.tool_calls:
            continue
        calls = request.tool_calls
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            continue
        ids = [call.get('id') for call in calls]
        if any(not isinstance(id, str) or not id for id in ids) or len(set(ids)) != len(ids):
            continue
        results = group[2:-1]
        if len(results) != len(calls) or any(row.role != 'tool' for row in results):
            continue
        if set(row.tool_call_id for row in results) != set(ids):
            continue
        try:
            valid = [call for call in calls if call.get('type') != 'invalid_tool_call']
            invalid = [call for call in calls if call.get('type') == 'invalid_tool_call']
            assistant = AIMessage(content=request.content or '', tool_calls=valid,
                                  invalid_tool_calls=invalid)
        except (TypeError, ValueError):
            continue
        turns.append([user, assistant,
                      *(ToolMessage(content=row.content or '', tool_call_id=row.tool_call_id)
                        for row in results), final])
    return turns


def prepare_context(rows, system: SystemMessage, current: HumanMessage,
                    settings: Settings, *, suffix: list[BaseMessage] | None = None) -> list[BaseMessage]:
    """Bound prompt context by dropping oldest whole turns; never change rows."""
    limit = settings.context_token_budget - settings.response_token_reserve
    active = [current, *(suffix or [])]
    if count_tokens_approximately([system, *active]) > limit:
        raise InputBudgetExceeded('System and active turn exceed input budget')
    turns = completed_turns(rows)[-settings.max_turns_per_conversation:]
    candidate = [system, *(message for turn in turns for message in turn), *active]
    while count_tokens_approximately(candidate) > limit:
        turns.pop(0)
        candidate = [system, *(message for turn in turns for message in turn), *active]
    return candidate
