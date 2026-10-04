from pathlib import Path
import subprocess
import sys

import pytest

from app.knowledge_chunking import chunk_markdown


FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.mark.parametrize('body, source_line', [('# \nBody.', 1), ('Before.\n# \nBody.', 2)])
def test_empty_heading_rejected_without_hanging(body, source_line):
    # A separate process bounds this regression even if parsing stops advancing.
    script = (
        'from app.knowledge_chunking import chunk_markdown\n'
        'try:\n'
        f'    chunk_markdown({body!r}, content_type="manual")\n'
        'except ValueError as error:\n'
        '    print(error)\n'
        'else:\n'
        '    raise AssertionError("empty heading was accepted")\n'
    )
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=3)
    assert completed.returncode == 0, completed.stderr
    assert f'line {source_line}' in completed.stdout
    assert 'empty heading' in completed.stdout


@pytest.mark.parametrize('sentence, limit', [
    ('Price .99 USD.', 14), ('Price -.99 USD.', 15),
    ('Price +.99 USD.', 15), ('.99 USD.', 8),
])
def test_leading_and_signed_decimal_is_indivisible(sentence, limit):
    with pytest.raises(ValueError, match='line 2.*indivisible sentence'):
        chunk_markdown('# Rate\n' + sentence, content_type='manual', max_chars=limit - 1, overlap_chars=0)
    chunks = chunk_markdown(sentence + ' Next.', content_type='manual', max_chars=limit, overlap_chars=0)
    assert [c.answer for c in chunks] == [sentence, 'Next.']


def test_full_stop_after_decimal_sentence_remains_boundary():
    chunks = chunk_markdown('Done. Next.', content_type='manual', max_chars=6, overlap_chars=0)
    assert [c.answer for c in chunks] == ['Done.', 'Next.']


def test_heading_path_and_policy_fields():
    chunks = chunk_markdown((FIXTURES / 'ch03_policy.md').read_text(encoding='utf-8'), content_type='policy')
    assert [(c.category, c.questions, c.section_path) for c in chunks] == [
        ('售后政策 / 退货', '期限', '售后政策 / 退货 / 期限'),
        ('售后政策 / 退货', '期限', '售后政策 / 退货 / 期限'),
        ('售后政策 / 退货', '期限', '售后政策 / 退货 / 期限'),
        ('售后政策', '运费', '售后政策 / 运费'),
    ]
    assert [c.is_key_clause for c in chunks] == [False, True, False, False]
    assert [c.source_order for c in chunks] == [0, 1, 2, 3]
    assert [c.source_line for c in chunks] == [4, 7, 9, 13]
    assert all(c.content_type == 'policy' for c in chunks)
    assert chunks[1].answer == '商品必须保留原包装。请附上订单号。'
    assert chunks[2].answer == '[!WARNING]\n请勿寄送到付款包裹。'


def test_sentence_overlap_preserves_decimal_and_full_stops():
    chunks = chunk_markdown('# 费率\n价格为3.14元。Ready! Next? Done.结束。', content_type='manual', max_chars=20, overlap_chars=7)
    assert [c.answer for c in chunks] == ['价格为3.14元。Ready!', 'Ready! Next? Done.', 'Done.结束。']
    assert all(len(c.answer) <= 20 for c in chunks)
    assert [c.source_order for c in chunks] == [0, 1, 2]
    assert chunk_markdown('', content_type='manual') == []
    # Removing the "new content" requirement would produce a duplicate tail.
    assert [c.answer for c in chunk_markdown('甲。乙。', content_type='manual', max_chars=4, overlap_chars=3)] == ['甲。乙。']
    assert [c.answer for c in chunk_markdown('第一句很长。第二句。', content_type='manual', max_chars=7, overlap_chars=6)] == ['第一句很长。', '第二句。']


def test_table_groups_repeat_header():
    chunks = chunk_markdown((FIXTURES / 'ch03_manual.md').read_text(encoding='utf-8'), content_type='manual', max_chars=70, overlap_chars=10)
    header = '| 操作 | 说明 |\n| --- | --- |'
    assert [c.answer for c in chunks] == [
        '先确认设备已经断电。',
        header + '\n| 开机 | 按住电源键三秒后松开。 |\n| 关机 | 保存设置后关闭电源。 |',
        header + '\n| 复位 | 长按复位键五秒后等待重启。 |',
        '完成后检查指示灯。',
    ]
    assert all(len(c.answer) <= 70 for c in chunks)
    assert [c.source_line for c in chunks] == [3, 7, 9, 11]
    assert all(c.section_path == '设备手册 / 使用' for c in chunks)


def test_table_preserves_empty_and_escaped_pipe_cells():
    text = '| A | B | C |\n| --- | --- | --- |\n|| x\\|y ||'
    chunks = chunk_markdown(text, content_type='manual')
    assert [c.answer for c in chunks] == [text]


def test_sentence_keeps_closing_quote_with_punctuation():
    chunks = chunk_markdown('“第一句。”下一句。', content_type='policy', max_chars=8, overlap_chars=0)
    assert [c.answer for c in chunks] == ['“第一句。”', '下一句。']


@pytest.mark.parametrize('body, limit, location, kind', [
    ('# 标题\n' + '长' * 21 + '。', 20, 2, 'sentence'),
    ('# 表格\n| A | B |\n| --- | --- |\n| x | ' + '长' * 30 + ' |', 35, 4, 'row'),
    ('# 表格\n| A | B |\n| --- | --- |\n| x |', 80, 4, 'row'),
])
def test_indivisible_input_rejected_with_location(body, limit, location, kind):
    with pytest.raises(ValueError, match=rf'line {location}.*{kind}'):
        chunk_markdown(body, content_type='policy', max_chars=limit)


@pytest.mark.parametrize('kwargs', [{'max_chars': 0}, {'overlap_chars': -1}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        chunk_markdown('正文。', content_type='policy', **kwargs)


@pytest.mark.parametrize('fence', ['```', '~~~'])
def test_fenced_code_is_verbatim_and_does_not_change_heading_context(fence):
    block = (fence + 'sh\n# reboot system\n\n  reboot --now  \n'
             '| A | B |\n| --- | --- |\n| malformed |\n'
             '> [!IMPORTANT]\n> literal code\n' + fence)
    chunks = chunk_markdown('# Manual\n## Restart\nBefore.\n' + block + '\nAfter.',
                            content_type='manual')
    assert [c.answer for c in chunks] == ['Before.', block, 'After.']
    assert [c.source_line for c in chunks] == [3, 4, 14]
    assert all(c.section_path == 'Manual / Restart' for c in chunks)
    assert all(c.questions == 'Restart' and not c.is_key_clause for c in chunks)


def test_fence_requires_matching_character_and_at_least_opening_length():
    block = '````text\n```\n~~~\n# literal\n````'
    chunks = chunk_markdown('# Manual\n' + block + '\n## Next\nDone.', content_type='manual')
    assert [c.answer for c in chunks] == [block, 'Done.']
    assert [c.section_path for c in chunks] == ['Manual', 'Manual / Next']


def test_unclosed_fence_preserves_remaining_lines_verbatim():
    block = '~~~\n# literal\n\n> [!IMPORTANT]\n  code  '
    chunks = chunk_markdown('# Manual\n' + block, content_type='manual')
    assert [c.answer for c in chunks] == [block]
    assert chunks[0].section_path == 'Manual' and not chunks[0].is_key_clause


def test_oversized_fence_rejected_as_indivisible_with_source_line():
    with pytest.raises(ValueError, match='line 3.*indivisible fenced code'):
        chunk_markdown('# Manual\n\n```\nlong code\n```',
                       content_type='manual', max_chars=12)
