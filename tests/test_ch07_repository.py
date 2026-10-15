"""Real MySQL checks for additive anchors and fenced summary commits."""
from datetime import datetime, timedelta, UTC
import pytest
from sqlalchemy import text
from test_db import database_url, sessions
from app.repository import Repository


@pytest.fixture
def context(sessions, database_url):
    from app.init_ch07_db import initialize_ch07_database
    from app.context_repository import ContextRepository
    initialize_ch07_database(database_url)
    initialize_ch07_database(database_url)
    repo=Repository(sessions)
    cid=repo.create_conversation('test-context')
    ids=[]
    for i in range(3):
        ids.append(int(repo.append_message(cid,'user',f'订单{i}')))
        ids.append(int(repo.append_message(cid,'assistant',f'答复{i}')))
    return ContextRepository(sessions),cid,ids


def test_complete_is_atomic_and_idempotent(context):
    repo,cid,ids=context
    assert repo.advance_layer1(cid,0,ids[4])
    claim=repo.claim(cid,ids[0],ids[3],'a'*64)
    assert claim and claim.ordinal==1
    assert repo.claim(cid,ids[0],ids[3],'a'*64) is None
    assert repo.complete(claim,'用户查询订单，等待核实。',30)
    assert not repo.complete(claim,'重复',3)
    state=repo.load(cid)
    assert state.summary_upto_msg_id==ids[3]
    assert len(state.summaries)==1


def test_late_expired_claim_cannot_commit(context):
    repo,cid,ids=context
    repo.advance_layer1(cid,0,ids[4])
    claim=repo.claim(cid,ids[0],ids[3],'a'*64)
    with repo.session_factory.begin() as session:
        session.execute(text('UPDATE conversations SET summary_lease_until=:at WHERE id=:id'),
                        {'at':datetime.now(UTC).replace(tzinfo=None)-timedelta(seconds=5),'id':cid})
    replacement=repo.claim(cid,ids[0],ids[3],'a'*64)
    assert replacement.lease_token!=claim.lease_token
    assert not repo.complete(claim,'过时',2)
    assert repo.complete(replacement,'当前有效摘要',8)


def test_downgrade_during_summary_does_not_skip_range(context):
    repo,cid,ids=context
    repo.advance_layer1(cid,0,ids[2])
    claim=repo.claim(cid,ids[0],ids[1],'a'*64)
    state=repo.load(cid)
    assert repo.advance_layer1(cid,state.version,ids[4])
    assert repo.complete(claim,'第一轮',4)
    assert repo.load(cid).summary_upto_msg_id==ids[1]
    assert repo.load(cid).layer1_from_msg_id==ids[4]


def test_cross_conversation_and_backward_boundary_are_rejected(context):
    repo,cid,ids=context
    with pytest.raises(ValueError):
        repo.advance_layer1(cid,0,ids[-1])  # assistant cannot begin a layer
    repo.advance_layer1(cid,0,ids[4])
    with pytest.raises(ValueError):
        repo.advance_layer1(cid,repo.load(cid).version,ids[0])


def test_failed_summary_preserves_boundary(context):
    repo,cid,ids=context
    repo.advance_layer1(cid,0,ids[4])
    claim=repo.claim(cid,ids[0],ids[3],'a'*64)
    repo.fail(claim,'timeout')
    assert repo.load(cid).summary_upto_msg_id is None
    assert not repo.load(cid).summaries
