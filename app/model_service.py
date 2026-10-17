"""Configured chat model operations."""

from collections.abc import AsyncIterator
import json

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.prompts import render_extraction_prompt
from app.schemas import AfterSalesExtraction


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings=settings
        self._response_token_reserve = settings.response_token_reserve
        self._model = ChatOpenAI(
            base_url=settings.chat_base_url,
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            timeout=180,
            max_retries=0,
        )

        from app.intent import IntentClassifier
        self._intent_classifier = IntentClassifier(settings)

        self._agent_model = ChatOpenAI(
            base_url=settings.chat_base_url, model=settings.chat_model,
            api_key=settings.chat_api_key, timeout=180, max_retries=0,
        )
        from app.ch07_evaluation import configured_meter, MeteredModel
        meter=configured_meter()
        if meter is not None:
            self._model=MeteredModel(self._model,meter)
            self._agent_model=MeteredModel(self._agent_model,meter)

    async def classify_intent(self, text: str):
        """Compatibility facade for the independent Jev decision node."""
        return await self._intent_classifier.classify(text)

    def _validate_context(self,messages,tools,output):
        if getattr(getattr(self,'_settings',None),'layered_context',False):
            from app.context_budget import validate_request
            validate_request(messages,tools,output,self._settings.context_token_budget)

    async def summarize(self, *, background, turns):
        from app.summary_prompts import summary_messages
        from app.context_budget import validate_request
        messages=summary_messages(background,turns)
        validate_request(messages,[],512,self._settings.context_token_budget)
        response=await self._agent_model.ainvoke(messages,max_tokens=512,temperature=0)
        return response.content

    async def recap_reply(self,messages,*,max_tokens=512):
        from app.context_recap import RecapAnswer,RecapResponseError
        from app.jev_classifier import strict_json
        self._validate_context(messages,[],max_tokens)
        structured=self._agent_model.with_structured_output(RecapAnswer,
            method='json_mode',include_raw=True,temperature=0,max_tokens=max_tokens)
        result=await structured.ainvoke(messages)
        raw=result['raw']
        if not isinstance(raw,AIMessage) or not isinstance(raw.content,str):
            raise RecapResponseError('Invalid recap model output',getattr(raw,'usage_metadata',None))
        try:
            answer=RecapAnswer.model_validate(strict_json(raw.content)).answer.strip()
            if not answer or 'DSML' in answer or 'invoke name=' in answer:
                raise ValueError('Invalid recap model output')
        except ValueError as error:
            raise RecapResponseError('Invalid recap model output',raw.usage_metadata) from error
        return AIMessage(content=answer,usage_metadata=raw.usage_metadata)

    async def resolve_query(self, messages: list) -> AIMessage:
        self._validate_context(messages,[],1024)
        from app.ch06_types import Resolution
        structured = self._agent_model.with_structured_output(
            Resolution, method='json_mode', include_raw=True, temperature=0, max_tokens=1024)
        result = await structured.ainvoke(messages)
        # The resolver revalidates raw JSON, preserving usage even on parsing failure.
        return result['raw']

    async def expand_query(self, messages: list) -> AIMessage:
        self._validate_context(messages,[],1024)
        from app.ch06_types import Expansion
        structured = self._agent_model.with_structured_output(
            Expansion, method='json_mode', include_raw=True, temperature=0, max_tokens=1024)
        result = await structured.ainvoke(messages)
        return result['raw']

    async def _ch06_structured(self,schema,messages,max_tokens,on_request=None):
        self._validate_context(messages,[],max_tokens)
        from app.jev_classifier import strict_json
        from app.query_resolver import QueryResolutionError
        from app.workflow_types import Usage
        from app.agent_runtime import _estimate
        structured=self._agent_model.with_structured_output(schema,method='json_mode',include_raw=True,temperature=0,max_tokens=max_tokens)
        if on_request is not None:on_request(messages)
        result=await structured.ainvoke(messages)
        raw=result['raw']
        usage=(Usage(raw.usage_metadata['input_tokens'],raw.usage_metadata['output_tokens']) if isinstance(raw,AIMessage) and raw.usage_metadata else Usage(estimated=_estimate(messages)+max_tokens))
        try:
            if not isinstance(raw,AIMessage) or not isinstance(raw.content,str) or len(raw.content)>16000 or raw.tool_calls or raw.invalid_tool_calls:raise ValueError()
            return schema.model_validate(strict_json(raw.content)),usage
        except (TypeError,ValueError):raise QueryResolutionError('agent_invalid_response',usage) from None

    async def ask_customer(self,query,*,max_tokens=512,on_request=None):
        from app.ch06_types import CustomerQuestion
        from app.ch06_prompts import CUSTOMER_SYSTEM
        from langchain_core.messages import SystemMessage,HumanMessage
        result,usage=await self._ch06_structured(CustomerQuestion,[SystemMessage(content=CUSTOMER_SYSTEM),HumanMessage(content=query)],max_tokens,on_request)
        return result.question,usage

    async def judge_eligibility(self,query,item,evidence,facts,*,max_tokens=512,on_request=None):
        from datetime import datetime
        from app.business_data import TZ
        from app.ch06_types import Eligibility
        from app.ch06_prompts import ELIGIBILITY_SYSTEM
        from langchain_core.messages import SystemMessage,HumanMessage
        payload={'query':query,'item':item.to_dict(),'evidence':evidence,'facts':facts,'now':datetime.now(TZ).isoformat()}
        return await self._ch06_structured(Eligibility,[SystemMessage(content=ELIGIBILITY_SYSTEM),HumanMessage(content=json.dumps(payload,ensure_ascii=False))],max_tokens,on_request)

    async def select(self, messages: list, tools: list, *, max_tokens: int) -> AIMessage:
        """Chapter 5 internal selection, with a per-request output budget."""
        self._validate_context(messages,tools,max_tokens)
        selection = await self._agent_model.bind_tools(tools).ainvoke(messages, max_tokens=max_tokens)
        if not isinstance(selection, AIMessage):
            raise TypeError("Tool selection must return an AIMessage")
        return selection

    async def stream_reply(self, messages: list, *, max_tokens: int) -> AsyncIterator[AIMessageChunk]:
        """Unbound terminal generation retaining provider usage chunks."""
        self._validate_context(messages,[],max_tokens)
        async for chunk in self._agent_model.astream(messages, max_tokens=max_tokens, stream_usage=True):
            yield chunk

    async def choose_tool(self, messages: list[BaseMessage], tools: list[BaseTool]) -> AIMessage:
        """Make one async selection request, leaving the final model unbound."""
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        if not isinstance(selection, AIMessage):
            raise TypeError("Tool selection must return an AIMessage")
        return selection

    async def stream_chat(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        async for chunk in self._model.astream(messages):
            text = str(chunk.text)
            if text:
                yield text

    async def extract_after_sales(self, description: str) -> AfterSalesExtraction:
        structured_model = self._model.with_structured_output(
            AfterSalesExtraction, method="json_mode", include_raw=True
        )
        result = await structured_model.ainvoke(render_extraction_prompt(description))
        if result["parsing_error"] is not None:
            raise ValueError("Upstream extraction could not be parsed")
        raw = result["raw"].content

        def reject_non_json_constant(value: str) -> None:
            raise ValueError(f"Invalid JSON constant: {value}")

        payload = json.loads(raw, parse_constant=reject_non_json_constant)
        return AfterSalesExtraction.model_validate(payload)


    async def understand_query(self, raw_question: str):
        from app.query_understanding import QueryDecision
        from app.query_prompts import query_messages
        structured = self._model.with_structured_output(QueryDecision, method='json_mode', include_raw=True)
        result = await structured.ainvoke(query_messages(raw_question))
        if result['parsing_error'] is not None:
            raise ValueError('Query understanding returned invalid JSON')
        raw = result['raw'].content
        if not isinstance(raw,str) or len(raw)>50000:
            raise ValueError('Query understanding must return bounded JSON text')
        def reject_constant(value): raise ValueError('Invalid JSON constant')
        def unique_keys(items):
            value={}
            for key,item in items:
                if key in value: raise ValueError('Duplicate JSON key')
                value[key]=item
            return value
        try:
            return QueryDecision.model_validate(json.loads(raw,parse_constant=reject_constant,object_pairs_hook=unique_keys))
        except (ValueError,TypeError):
            raise ValueError('Query understanding returned invalid structured JSON') from None


    async def generate_knowledge(self, messages):
        from app.rag_generation import GenerationDecision,GenerationFormatError
        structured=self._model.with_structured_output(GenerationDecision,method='json_mode',include_raw=True,
            temperature=0,max_tokens=self._response_token_reserve)
        # Transport failures propagate, distinct from schema/parser failure.
        result=await structured.ainvoke(messages)
        try:
            if result['parsing_error'] is not None: raise ValueError()
            raw=result['raw'].content
            if not isinstance(raw,str) or len(raw)>16000: raise ValueError()
            def reject_constant(value): raise ValueError()
            def unique_keys(items):
                payload={}
                for key,value in items:
                    if key in payload: raise ValueError()
                    payload[key]=value
                return payload
            return GenerationDecision.model_validate(json.loads(raw,parse_constant=reject_constant,object_pairs_hook=unique_keys))
        except (ValueError,TypeError,KeyError,AttributeError):
            raise GenerationFormatError('Knowledge generation returned invalid structured JSON') from None
