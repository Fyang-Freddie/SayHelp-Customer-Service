"""Bounded, in-memory conversation history."""

from collections import OrderedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.config import Settings


class UnknownConversation(Exception):
    """The supplied conversation ID is absent or has been evicted."""


class InputBudgetExceeded(Exception):
    """System and current user messages exceed the available input budget."""


class ConversationStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._conversations: OrderedDict[str, list[BaseMessage]] = OrderedDict()

    def create(self) -> str:
        conversation_id = str(uuid4())
        self._conversations[conversation_id] = []
        if len(self._conversations) > self._settings.max_conversations:
            self._conversations.popitem(last=False)
        return conversation_id

    def get(self, conversation_id: str) -> list[BaseMessage]:
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
        self.get(conversation_id)
        messages = self._conversations[conversation_id]
        messages.extend((user, assistant))
        excess = len(messages) - 2 * self._settings.max_turns_per_conversation
        if excess > 0:
            del messages[:excess]
