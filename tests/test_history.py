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


@pytest.mark.parametrize('suffix', ['user_only', 'missing_result', 'wrong_id', 'missing_final', 'duplicate_id', 'orphan_result'])
def test_database_context_ignores_incomplete_or_malformed_turns(suffix):
    from types import SimpleNamespace
    from app.history import completed_turns

    def row(role, content='', calls=None, call_id=None):
        return SimpleNamespace(role=role, content=content, tool_calls=calls, tool_call_id=call_id)

    calls = [{'name': 'query_faq', 'args': {'keyword': '退货'}, 'id': 'c1'}]
    good = [row('user', 'complete'), row('assistant', 'answer')]
    bad = [row('user', 'unfinished')]
    if suffix != 'user_only':
        if suffix == 'duplicate_id':
            calls = [*calls, *calls]
        bad.append(row('assistant', calls=calls))
        if suffix != 'missing_result':
            bad.append(row('tool', 'result', call_id='wrong' if suffix == 'wrong_id' else 'c1'))
        if suffix != 'missing_final':
            bad.append(row('assistant', 'untrusted final'))
        if suffix == 'orphan_result':
            bad.insert(-1, row('tool', 'orphan', call_id='unknown'))
    turns = completed_turns([*good, *bad])
    assert [[m.content for m in turn] for turn in turns] == [['complete', 'answer']]


def test_context_recovers_later_complete_turn_after_failed_audit_rows():
    from types import SimpleNamespace
    from app.history import completed_turns
    rows = [SimpleNamespace(role=role, content=content, tool_calls=None, tool_call_id=None)
            for role, content in [('user', 'failed'), ('user', 'recovered'), ('assistant', 'answer')]]
    assert [[m.content for m in turn] for turn in completed_turns(rows)] == [['recovered', 'answer']]
