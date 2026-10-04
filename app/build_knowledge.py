"""Initialize, ingest and resume offline dense knowledge indexing."""
import argparse
import os
from pathlib import Path
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.embedding import BgeM3Embedder
from app.init_knowledge_db import initialize_knowledge_database
from app.knowledge_ingest import ingest_faq, ingest_markdown
from app.knowledge_index import count_pending, index_pending
from app.vector_store import MilvusKnowledgeStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ('policy', 'manual'):
        parser.add_argument('--' + kind, action='append', default=[], metavar='MARKDOWN')
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    file_values = dotenv_values(Path.cwd() / '.env')
    def value(name, default=None):
        return os.environ.get(name, file_values.get(name)) or default
    url = value('DATABASE_URL')
    if not url:
        parser.error('DATABASE_URL is required')
    engine = None
    try:
        initialize_knowledge_database(url)
        engine = create_engine(url, pool_pre_ping=True)
        sessions = sessionmaker(engine)
        ingest_faq(sessions)
        for kind in ('policy', 'manual'):
            for path in getattr(args, kind):
                ingest_markdown(sessions, path, kind)
        embedder = BgeM3Embedder(cache_dir=value('BGE_CACHE_DIR'))
        store = MilvusKnowledgeStore(uri=value('MILVUS_URI', 'http://127.0.0.1:19530'))
        store.ensure_collection()
        total = 0
        while True:
            count = index_pending(sessions, embedder, store, args.batch_size)
            if count == 0:
                if count_pending(sessions):
                    raise RuntimeError('Knowledge indexing made no progress; pending rows remain')
                break
            total += count
        print(f'Knowledge indexing completed: {total} pending rows marked done.')
    except Exception:
        # Raw library errors may include credential-bearing connection URLs.
        raise SystemExit('Knowledge build failed; pending rows can be retried. Check MySQL, Milvus, model cache and input files.') from None
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == '__main__':
    main()
