"""Stable MySQL message identities and atomic, side-effect-free reply suggestions."""
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import undefer

from app.ch05_db import WorkflowAction, WorkflowMessageKey, WorkflowTurn
from app.db import Conversation, Message
from app.workflow_types import ActionSuggestion


def _active_conversation(session, conversation_id):
    row = session.get(Conversation, conversation_id, options=[undefer('*')], with_for_update=True)
    if row is None or row.deleted_at is not None:
        raise KeyError('Conversation not found')
    return row


def _identity(turn_id, position):
    if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 64:
        raise ValueError('Invalid turn identity')
    if type(position) is not int or not 0 <= position < 2**64:
        raise ValueError('Invalid message position')


def _suggestions(suggestions):
    result = []
    for item in suggestions:
        if item.get('kind') == 'handoff' and set(item) == {'kind'}:
            result.append({'kind': 'handoff', 'description': None, 'ticket_type': None})
        elif (item.get('kind') == 'create_ticket' and set(item) == {'kind', 'description', 'ticket_type'}
              and isinstance(item['description'], str) and item['description'].strip()
              and item['ticket_type'] in ('售后', '投诉', '咨询')):
            result.append(dict(item))
        else:
            raise ValueError('Invalid action suggestion')
    return result


def _action_dict(row):
    return {'action_id': row.action_id, 'kind': row.kind,
            'description': row.description, 'ticket_type': row.ticket_type}


class WorkflowRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def _write(self, operation):
        # A failed flush ends this session; only a new transaction may retry.
        try:
            with self.session_factory.begin() as session:
                return operation(session)
        except IntegrityError as error:
            if getattr(error.orig, 'args', (None,))[0] != 1062:
                raise
            with self.session_factory.begin() as session:
                return operation(session)

    @staticmethod
    def _turn(session, conversation_id, turn_id):
        _active_conversation(session, conversation_id)
        row = session.get(WorkflowTurn, turn_id)
        if row is None:
            row = WorkflowTurn(turn_id=turn_id, conversation_id=conversation_id)
            session.add(row)
            session.flush()
        elif row.conversation_id != conversation_id:
            raise ValueError('Turn identity conflict')
        return row

    @staticmethod
    def _message(session, conversation_id, turn_id, position, role, content, tool_calls, tool_call_id, citations=None):
        key = session.get(WorkflowMessageKey, (turn_id, position))
        if key is not None:
            row = session.get(Message, key.message_id, options=[undefer(Message.citations)])
            if (row.conversation_id, row.role, row.content, row.tool_calls, row.tool_call_id, row.citations) != (conversation_id, role, content, tool_calls, tool_call_id, citations):
                raise ValueError('Message payload conflict')
            return row, False
        row = Message(conversation_id=conversation_id, role=role, content=content,
                      tool_calls=tool_calls, tool_call_id=tool_call_id, citations=citations)
        session.add(row)
        session.flush()
        session.add(WorkflowMessageKey(turn_id=turn_id, position=position, message_id=row.id))
        session.flush()
        return row, True

    def append_message_once(self, conversation_id: int, turn_id: str, position: int, role: str,
                            content: str | None, *, tool_calls=None, tool_call_id=None) -> str:
        _identity(turn_id, position)
        def operation(session):
            self._turn(session, conversation_id, turn_id)
            row, _ = self._message(session, conversation_id, turn_id, position, role, content, tool_calls, tool_call_id)
            return str(row.id)
        return self._write(operation)

    def commit_reply(self, conversation_id: int, turn_id: str, position: int, answer: str,
                     citations: list, suggestions: list[ActionSuggestion]) -> tuple[str, list[dict]]:
        _identity(turn_id, position)
        normalized = _suggestions(suggestions)
        def operation(session):
            turn = self._turn(session, conversation_id, turn_id)
            row, created = self._message(session, conversation_id, turn_id, position, 'assistant', answer, None, None, citations)
            if turn.final_message_id is not None and turn.final_message_id != row.id:
                raise ValueError('Final reply identity conflict')
            if created:
                prefix = uuid5(NAMESPACE_URL, f'{turn_id}:{position}').hex
                for index, suggestion in enumerate(normalized):
                    session.add(WorkflowAction(action_id=f'{prefix}:{index:04d}', conversation_id=conversation_id,
                                               turn_id=turn_id, message_id=row.id, **suggestion))
                session.flush()
            actions = list(session.scalars(select(WorkflowAction).where(WorkflowAction.message_id == row.id).order_by(WorkflowAction.action_id)))
            if [{'kind': a.kind, 'description': a.description, 'ticket_type': a.ticket_type} for a in actions] != normalized:
                raise ValueError('Reply suggestions conflict')
            turn.final_message_id = row.id
            turn.status = 'complete'
            return str(row.id), [_action_dict(a) for a in actions]
        return self._write(operation)

    def load_actions(self, conversation_id: int) -> dict[str, list[dict]]:
        with self.session_factory() as session:
            rows = session.scalars(select(WorkflowAction).join(Conversation, Conversation.id == WorkflowAction.conversation_id)
                                    .where(WorkflowAction.conversation_id == conversation_id, Conversation.deleted_at.is_(None))
                                    .order_by(WorkflowAction.message_id, WorkflowAction.action_id))
            result = {}
            for row in rows:
                result.setdefault(str(row.message_id), []).append(_action_dict(row))
            return result

    def set_turn_status(self, turn_id: str, status: str) -> None:
        if not isinstance(status, str) or not status.strip() or len(status) > 32:
            raise ValueError('Invalid turn status')
        with self.session_factory.begin() as session:
            turn = session.get(WorkflowTurn, turn_id)
            if turn is None:
                raise KeyError('Turn not found')
            _active_conversation(session, turn.conversation_id)
            turn.status = status
