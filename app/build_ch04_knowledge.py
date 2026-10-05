"""Build the explicit real Chapter 4 corpus without replacing the legacy collection."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from app.ch04_ingest import ingest_corpus, load_documents
from app.knowledge_chunking import chunk_markdown

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'eval/ch04/corpus.json')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.batch_size<=128:
        parser.error('--batch-size must be within [1,128]')
    if args.dry_run:
        try:
            sources=[]
            for name,kind,body,digest in load_documents(args.manifest):
                chunks=chunk_markdown(body,content_type=kind)
                if not chunks:
                    raise ValueError('Corpus source is empty')
                sources.append({'source_file':name,'sha256':digest,'chunks':len(chunks),
                                'sample_range':[chunks[0].source_start_line,chunks[0].source_end_line]})
        except Exception:
            raise SystemExit('Corpus dry-run failed; check explicit manifest and source files') from None
        print(json.dumps({'mode':'dry_run','documents':len(sources),'chunks':sum(s['chunks'] for s in sources),'sources':sources},ensure_ascii=False))
        return
    values=dotenv_values(Path.cwd()/'.env')
    def value(name,default=None):
        return os.environ.get(name,values.get(name)) or default
    url=value('DATABASE_URL')
    if not url:
        parser.error('DATABASE_URL is required')
    factory,store=None,None
    try:
        from app.init_ch04_db import initialize_ch04_database
        from app.db import make_session_factory
        from app.embedding import BgeM3Embedder
        from app.hybrid_store import HybridMilvusStore
        from app.ch04_index import build_index
        initialize_ch04_database(url)
        factory=make_session_factory(url)
        snapshot=ingest_corpus(factory,args.manifest)
        store=HybridMilvusStore(uri=value('MILVUS_URI','http://127.0.0.1:19530'),collection_name=value('KNOWLEDGE_COLLECTION','knowledge_ch04'))
        summary=build_index(factory,BgeM3Embedder(cache_dir=value('BGE_CACHE_DIR')),store,snapshot,args.batch_size)
        print(json.dumps({'mode':'built',**asdict(summary)},ensure_ascii=False))
    except Exception:
        raise SystemExit('Chapter 4 build failed; verify schema, corpus, model cache and Milvus. Independent pending state can be retried.') from None
    finally:
        if store is not None:
            store.close()
        if factory is not None:
            factory.kw['bind'].dispose()


if __name__=='__main__':
    main()
