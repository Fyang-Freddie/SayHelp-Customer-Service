"""Dedicated daily conversation mining/indexing process; --once runs immediately."""
import argparse
from functools import partial
from apscheduler.schedulers.blocking import BlockingScheduler
from langchain_openai import ChatOpenAI
from app.config import Settings
from app.db import make_session_factory
from app.embedding import BgeM3Embedder
from app.init_knowledge_db import initialize_knowledge_database
from app.knowledge_index import count_pending, index_pending
from app.qa_mining import mine_conversations
from app.vector_store import MilvusKnowledgeStore


def run_once(session_factory, llm, embedder, store, *, batch_size=20):
    summary = mine_conversations(session_factory, llm, batch_size=batch_size, embedder=embedder)
    store.ensure_collection()
    while index_pending(session_factory, embedder, store, 100):
        pass
    if count_pending(session_factory):
        raise RuntimeError('QA indexing made no progress; pending rows remain')
    return summary


def run_scheduled_mining(session_factory, llm, embedder, store, *, batch_size=20, scheduler=None):
    scheduler = scheduler or BlockingScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(partial(run_once, session_factory, llm, embedder, store, batch_size=batch_size),
        'cron', hour=2, minute=0, timezone='Asia/Shanghai', id='sayhelp_daily_qa',
        max_instances=1, coalesce=True, replace_existing=True)
    scheduler.start()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    sessions = None
    try:
        settings = Settings.from_env()
        initialize_knowledge_database(settings.database_url)
        sessions = make_session_factory(settings.database_url)
        llm = ChatOpenAI(base_url=settings.chat_base_url, model=settings.chat_model,
            api_key=settings.chat_api_key, temperature=0, timeout=60, max_retries=0,
            max_tokens=4096).bind(response_format={'type':'json_object'})
        embedder = BgeM3Embedder(cache_dir=settings.bge_cache_dir)
        store = MilvusKnowledgeStore(uri=settings.milvus_uri)
        if args.once:
            result = run_once(sessions, llm, embedder, store, batch_size=args.batch_size)
            print(f'QA mining complete: staged={result.staged}, kept={result.kept}, discarded={result.discarded}.')
        else:
            run_scheduled_mining(sessions, llm, embedder, store, batch_size=args.batch_size)
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit('QA mining failed; staged and pending rows can be retried. Check configured services and model cache.') from None
    finally:
        if sessions is not None:
            sessions.kw['bind'].dispose()


if __name__ == '__main__':
    main()
