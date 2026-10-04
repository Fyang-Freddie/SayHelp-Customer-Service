"""Markdown chunks with intact sentences/rows and deterministic source order.

Limits count characters in ``answer`` including repeated table headers. Overlap
is a best-effort suffix of whole sentences, reduced when new text needs space.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class KnowledgeDraft:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool
    source_order: int  # Zero-based position in this document's returned list.
    source_line: int  # One-based first body sentence/row line, including overlap.


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith('|'):
        body = body[1:]
    if body.endswith('|') and not body.endswith('\\|'):
        body = body[:-1]
    return [part.strip() for part in re.split(r'(?<!\\)\|', body)]


def _separator(line: str) -> bool:
    return '|' in line and all(re.fullmatch(r':?-{3,}:?', cell) for cell in _cells(line))


def _heading(line: str):
    return re.fullmatch(r'[ \t]{0,3}(#{1,6})(?:[ \t]+(.*))?', line)


def _fence_open(line: str):
    match = re.fullmatch(r'[ \t]{0,3}(`{3,}|~{3,})(.*)', line)
    if match and (match[1][0] != '`' or '`' not in match[2]):
        return match[1]
    return None


def _fence_close(line: str, opening: str) -> bool:
    return re.fullmatch(r'[ \t]{0,3}' + re.escape(opening[0]) +
                        '{' + str(len(opening)) + r',}[ \t]*', line) is not None


def _sentences(text: str, line: int) -> list[tuple[str, int]]:
    units = []
    start = 0
    for index, char in enumerate(text):
        if char not in '。！？.!?':
            continue
        # A fractional point can follow a digit, whitespace, or a sign (.99).
        if char == '.' and index + 1 < len(text) and text[index + 1].isdigit():
            continue
        # Keep consecutive punctuation with its sentence (e.g. "Really?!").
        if index + 1 < len(text) and text[index + 1] in '。！？.!?':
            continue
        end = index + 1
        while end < len(text) and text[end] in '\"\'”’）)]】》」』':
            end += 1
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        units.append((text[start:end], line + text[:start + leading].count('\n')))
        start = end
    if text[start:].strip():
        leading = len(text[start:]) - len(text[start:].lstrip())
        units.append((text[start:], line + text[:start + leading].count('\n')))
    return units


def _prose(text: str, line: int, limit: int, overlap: int) -> list[tuple[str, int]]:
    units = _sentences(text, line)
    for sentence, source_line in units:
        if len(sentence.strip()) > limit:
            raise ValueError(f'line {source_line}: oversized indivisible sentence (limit {limit})')
    result = []
    current = []
    for unit in units:
        if current and len(''.join(item[0] for item in current + [unit]).strip()) > limit:
            result.append((''.join(item[0] for item in current).strip(), current[0][1]))
            suffix = []
            for old in reversed(current):
                candidate = [old] + suffix
                if len(''.join(item[0] for item in candidate).strip()) > overlap:
                    break
                suffix = candidate
            while suffix and len(''.join(item[0] for item in suffix + [unit]).strip()) > limit:
                suffix.pop(0)
            current = suffix
        current.append(unit)
    if current:
        result.append((''.join(item[0] for item in current).strip(), current[0][1]))
    return result


def chunk_markdown(text: str, *, content_type: str, max_chars: int = 800,
                   overlap_chars: int = 120) -> list[KnowledgeDraft]:
    """Split heading sections, paragraphs, callouts, tables, and intact code fences.

    Heading paths use `` / ``. Text before the first heading has empty heading
    fields. Only an exact ``> [!IMPORTANT]`` callout marks a key clause. Errors
    include one-based document line numbers; no sentence or row is truncated.
    """
    if max_chars <= 0 or overlap_chars < 0:
        raise ValueError('max_chars must be positive and overlap_chars nonnegative')
    lines = text.splitlines()
    headings = []
    result = []

    def emit(answer: str, source_line: int, key: bool = False) -> None:
        if not answer.strip():
            return
        titles = [title for _, title in headings]
        result.append(KnowledgeDraft(
            category=' / '.join(titles[:-1]), questions=titles[-1] if titles else '',
            answer=answer, section_path=' / '.join(titles), content_type=content_type,
            is_key_clause=key, source_order=len(result), source_line=source_line,
        ))

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _fence_open(line)
        if fence:
            start = index
            index += 1
            while index < len(lines):
                closing = _fence_close(lines[index], fence)
                index += 1
                if closing:
                    break
            answer = '\n'.join(lines[start:index])
            if len(answer) > max_chars:
                raise ValueError(f'line {start + 1}: oversized indivisible fenced code '
                                 f'(limit {max_chars})')
            emit(answer, start + 1)
            continue
        heading = _heading(line)
        if heading:
            title = re.sub(r'(?:^|\s+)#+\s*$', '', heading[2] or '').strip()
            if not title:
                raise ValueError(f'line {index + 1}: empty heading')
            level = len(heading[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        # Tables precede prose detection: punctuation inside rows never splits.
        if '|' in line and index + 1 < len(lines) and _separator(lines[index + 1]):
            header = line.strip() + '\n' + lines[index + 1].strip()
            width = len(_cells(line))
            if len(_cells(lines[index + 1])) != width:
                raise ValueError(f'line {index + 2}: malformed table row separator')
            if len(header) > max_chars:
                raise ValueError(f'line {index + 1}: oversized table header row (limit {max_chars})')
            index += 2
            group = []
            first = index + 1
            while index < len(lines) and '|' in lines[index]:
                row = lines[index].strip()
                if len(_cells(row)) != width:
                    raise ValueError(f'line {index + 1}: malformed table row (expected {width} columns)')
                if len(header + '\n' + row) > max_chars:
                    raise ValueError(f'line {index + 1}: oversized indivisible table row (limit {max_chars})')
                if group and len(header + '\n' + '\n'.join(group + [row])) > max_chars:
                    emit(header + '\n' + '\n'.join(group), first)
                    group = []
                if not group:
                    first = index + 1
                group.append(row)
                index += 1
            if group:
                emit(header + '\n' + '\n'.join(group), first)
            continue
        quoted = line.lstrip().startswith('>')
        key = line.strip() == '> [!IMPORTANT]'
        start = index
        paragraph = []
        if key:
            index += 1
            start = index
        while index < len(lines) and lines[index].strip():
            candidate = lines[index]
            if _heading(candidate) or _fence_open(candidate):
                break
            if candidate.lstrip().startswith('>') != quoted:
                break
            if '|' in candidate and index + 1 < len(lines) and _separator(lines[index + 1]):
                break
            paragraph.append(re.sub(r'^\s*> ?', '', candidate) if quoted else candidate)
            index += 1
        for answer, source_line in _prose('\n'.join(paragraph), start + 1, max_chars, overlap_chars):
            emit(answer, source_line, key)
    return result
