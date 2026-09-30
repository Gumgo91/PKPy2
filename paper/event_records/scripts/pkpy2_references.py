"""Reference numbering in order of first citation, shared by the manuscript and supplement builds.

The revised manuscript keeps the original (PeerJ) reference numbers; Springer journals number
references in order of first citation. citation_order() reads the revised manuscript and returns
{old number: new number}; renumber() rewrites bracketed citations such as [3-6,15] in any text.
"""
import re

CITATION = re.compile(r'\[(\d+(?:[-–]\d+)?(?:,\s?\d+(?:[-–]\d+)?)*)\]')


def expand(group):
    out = []
    for part in group.replace(' ', '').split(','):
        if '-' in part or '–' in part:
            a, b = re.split('[-–]', part)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def compress(numbers):
    parts, start, prev = [], None, None
    for n in list(numbers) + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            parts.append(f'{start}' if start == prev else f'{start},{prev}' if prev == start + 1 else f'{start}–{prev}')
            start = prev = n
    return ','.join(parts)


def citation_order(paragraph_texts, n_references):
    """{old: new} from the order of first citation in the given body texts (reference list excluded)."""
    order = []
    for text in paragraph_texts:
        for m in CITATION.finditer(text):
            for k in expand(m.group(1)):
                if k not in order:
                    order.append(k)
    missing = [k for k in range(1, n_references + 1) if k not in order]
    if missing:
        raise ValueError(f'references never cited: {missing}')
    return {old: new for new, old in enumerate(order, start=1)}


def renumber(text, mapping):
    return CITATION.sub(lambda m: '[' + compress(sorted(mapping[k] for k in expand(m.group(1)))) + ']', text)


def manuscript_mapping(docx_path):
    """Mapping for the revised manuscript: body paragraphs before the reference list."""
    import docx
    d = docx.Document(docx_path)
    refs = [p for p in d.paragraphs if re.match(r'\[\d+\] ', p.text)]
    body = [p.text for p in d.paragraphs if not re.match(r'\[\d+\] ', p.text)]
    return citation_order(body, len(refs))
