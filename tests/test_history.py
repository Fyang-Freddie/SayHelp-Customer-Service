"""Behavior checks for bounded, completed conversation history."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.history import ConversationStore, InputBudgetExceeded, UnknownConversation


def settings(**overrides: int) -> Settings:
    values = dict(
        chat_base_url="https://example.invalid",
        chat_model="test-model",
        chat_api_key="test-key",
        context_token_budget=100,
        response_token_reserve=10,
        max_conversations=3,
        max_turns_per_conversation=3,
    )
    values.update(overrides)
    return Settings(**values)


def test_new_conversation_has_no_history_and_prepares_current_turn() -> None:
    store = ConversationStore(settings())
    conversation_id = store.create()
    system = SystemMessage(content="System")
    current = HumanMessage(content="Current")

    assert conversation_id
    assert store.get(conversation_id) == []
    assert store.prepare(conversation_id, system, current) == [system, current]
    assert store.get(conversation_id) == []


def test_completed_turn_is_returned_on_next_request() -> None:
    store = ConversationStore(settings())
    conversation_id = store.create()
    user = HumanMessage(content="First question")
    assistant = AIMessage(content="First answer")
    store.commit(conversation_id, user, assistant)
    system = SystemMessage(content="System")
    current = HumanMessage(content="Follow-up")

    assert store.get(conversation_id) == [user, assistant]
    assert store.prepare(conversation_id, system, current) == [
        system, user, assistant, current
    ]


def test_prepare_drops_oldest_whole_turn_without_changing_saved_history() -> None:
    # LangChain's approximate counts for these fixed messages are 37 for all
    # six and 25 after removing the first completed turn.
    store = ConversationStore(settings(context_token_budget=35))
    conversation_id = store.create()
    old_user = HumanMessage(content="Old")
    old_assistant = AIMessage(content="Reply")
    recent_user = HumanMessage(content="Recent")
    recent_assistant = AIMessage(content="Answer")
    store.commit(conversation_id, old_user, old_assistant)
    store.commit(conversation_id, recent_user, recent_assistant)
    system = SystemMessage(content="System")
    current = HumanMessage(content="Current")

    assert store.prepare(conversation_id, system, current) == [
        system, recent_user, recent_assistant, current
    ]
    assert store.get(conversation_id) == [
        old_user, old_assistant, recent_user, recent_assistant
    ]


def test_prepare_rejects_system_and_current_when_they_exceed_input_budget() -> None:
    # This pair costs 12 approximate tokens, while the input budget is 11.
    store = ConversationStore(settings(context_token_budget=21))
    conversation_id = store.create()

    with pytest.raises(InputBudgetExceeded):
        store.prepare(
            conversation_id,
            SystemMessage(content="System"),
            HumanMessage(content="Current"),
        )


def test_unknown_ids_are_rejected_by_every_history_operation() -> None:
    store = ConversationStore(settings())

    with pytest.raises(UnknownConversation):
        store.get("missing")
    with pytest.raises(UnknownConversation):
        store.prepare("missing", SystemMessage(content="S"), HumanMessage(content="C"))
    with pytest.raises(UnknownConversation):
        store.commit("missing", HumanMessage(content="U"), AIMessage(content="A"))


def test_commit_caps_completed_turns_without_splitting_a_pair() -> None:
    store = ConversationStore(settings(max_turns_per_conversation=2))
    conversation_id = store.create()
    turns = [
        (HumanMessage(content=f"User {number}"), AIMessage(content=f"AI {number}"))
        for number in range(3)
    ]
    for user, assistant in turns:
        store.commit(conversation_id, user, assistant)

    assert store.get(conversation_id) == [*turns[1], *turns[2]]


def test_conversation_cap_evicts_least_recently_used_id() -> None:
    store = ConversationStore(settings(max_conversations=2))
    oldest = store.create()
    recently_used = store.create()
    store.get(oldest)
    newest = store.create()

    with pytest.raises(UnknownConversation):
        store.get(recently_used)
    assert store.get(oldest) == []
    assert store.get(newest) == []


def test_new_conversation_is_reserved_before_another_creation_can_evict_it() -> None:
    from app.history import ConversationCapacityExceeded

    store = ConversationStore(settings(max_conversations=1))
    first = store.create_reserved()
    with pytest.raises(ConversationCapacityExceeded):
        store.create_reserved()
    assert store.get(first) == []

    store.release(first)
    second = store.create_reserved()
    with pytest.raises(UnknownConversation):
        store.get(first)
    assert store.get(second) == []
