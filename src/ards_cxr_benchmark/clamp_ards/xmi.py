from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .tokenization import Utf16OffsetMap
from .types import EntitySpan, SentenceSpan, TokenSpan


@dataclass(frozen=True)
class ClampXmiDocument:
    text: str
    sentences: tuple[SentenceSpan, ...]
    tokens: tuple[TokenSpan, ...]
    entities: tuple[EntitySpan, ...]


def parse_clamp_xmi(payload: bytes) -> ClampXmiDocument:
    """Read the CLAMP annotations needed for restricted parity characterization."""

    root = ET.fromstring(payload)
    sofa = next((element for element in root if element.tag.endswith("Sofa")), None)
    if sofa is None or "sofaString" not in sofa.attrib:
        raise ValueError("CLAMP XMI does not contain a Sofa string")
    text = sofa.attrib["sofaString"]
    offsets = Utf16OffsetMap.from_text(text)
    sentences: list[SentenceSpan] = []
    tokens: list[TokenSpan] = []
    entities: list[EntitySpan] = []
    for element in root:
        if element.tag.endswith("Sentence"):
            start = int(element.attrib["begin"])
            end = int(element.attrib["end"])
            source_start, source_end = offsets.python_span(start, end)
            sentences.append(
                SentenceSpan(
                    start=start,
                    end=end,
                    sentence_number=int(element.attrib["sentenceNumber"]),
                    source_start=source_start,
                    source_end=source_end,
                )
            )
        elif element.tag.endswith("BaseToken"):
            start = int(element.attrib["begin"])
            end = int(element.attrib["end"])
            source_start, source_end = offsets.python_span(start, end)
            tokens.append(
                TokenSpan(
                    start=start,
                    end=end,
                    token_number=int(element.attrib["tokenNumber"]),
                    source_start=source_start,
                    source_end=source_end,
                )
            )
        elif element.tag.endswith("ClampNameEntityUIMA"):
            start = int(element.attrib["begin"])
            end = int(element.attrib["end"])
            source_start, source_end = offsets.python_span(start, end)
            entities.append(
                EntitySpan(
                    start=start,
                    end=end,
                    semantic_tag=element.attrib.get("semanticTag", ""),
                    assertion=element.attrib.get("assertion", "present"),
                    cui=element.attrib.get("cui"),
                    attribute=element.attrib.get("attribute"),
                    source_start=source_start,
                    source_end=source_end,
                )
            )
    return ClampXmiDocument(
        text=text,
        sentences=tuple(sorted(sentences, key=lambda item: item.sentence_number)),
        tokens=tuple(sorted(tokens, key=lambda item: item.token_number)),
        entities=tuple(entities),
    )
