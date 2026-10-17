"""Authorize explicit persisted actions and invoke the original ticket tool."""
from app.schemas import TicketConfirmationRequest
from app.tools import build_tools
from app.workflow_service import _settled


class TicketActions:
    def __init__(self, repository, workflow_repository):
        self.repository = repository
        self.workflow_repository = workflow_repository

    async def confirm(self, conversation_id: int, action_id: str,
                      description: str, ticket_type: str) -> dict:
        request = TicketConfirmationRequest(action_id=action_id,
            description=description, ticket_type=ticket_type)
        # The request reservation must outlive cancellation of this worker.
        return await _settled(self._confirm, conversation_id, request)

    def _confirm(self, conversation_id, request):
        if self.repository.get_conversation(conversation_id) is None:
            raise KeyError('Conversation not found')
        action = next((item for group in self.workflow_repository.load_actions(conversation_id).values()
                       for item in group if item['action_id'] == request.action_id), None)
        if action is None:
            raise KeyError('Action not found')
        if action['kind'] != 'create_ticket':
            raise ValueError('Action does not create a ticket')
        tool = build_tools(self.repository, conversation_id, None,
            ticket_request_key=request.action_id)['create_ticket']
        return tool.invoke({'description': request.description, 'ticket_type': request.ticket_type})
