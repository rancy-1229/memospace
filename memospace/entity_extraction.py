"""
Entity extraction from text using spaCy NLP.

Extracts various types of entities from text:
- **Named entities**: Persons, organizations, locations, products, etc.
- **Proper nouns**: Capitalized multi-word sequences
- **Key topics**: Important noun phrases
- **Quoted text**: Text in single or double quotes
"""

import re
from typing import List, Optional
from dataclasses import dataclass

# Try importing spaCy, mark as optional
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


@dataclass(frozen=True)
class Entity:
    """Represents an extracted entity"""
    entity_type: str
    text: str
    confidence: float = 1.0
    ner_label: Optional[str] = None


# Entity type constants
ENTITY_TYPE_NAMED = "named"
ENTITY_TYPE_PROPER = "proper"
ENTITY_TYPE_TOPIC = "topic"
ENTITY_TYPE_QUOTED = "quoted"

# Words that are too generic to be useful
GENERIC_TERMS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique", "method",
    "approach", "process", "step", "tool", "result", "outcome", "goal",
    "task", "item", "topic", "scale", "size", "level", "degree", "amount",
    "number", "style", "look", "color", "colour", "shape", "form", "piece",
    "section", "side", "end", "edge", "surface", "point", "user", "assistant",
    "agent", "customer", "client", "person", "people", "human", "memory",
    "message", "conversation", "chat", "session", "system", "top", "works",
    "items", "things", "stuff", "resources", "options", "tips", "ideas",
    "steps", "ways", "methods", "tools", "features", "benefits", "examples",
    "details", "notes", "instructions", "guidelines", "recommendations",
    "suggestions", "overview", "summary", "conclusion", "introduction",
    "pros", "cons", "advantages", "disadvantages"
}

# Accepted NER labels (excluding numeric/temporal)
ACCEPTED_NER_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART",
    "EVENT", "NORP", "LAW", "LANGUAGE"
}

# Rejected NER labels (numeric/temporal)
REJECTED_NER_LABELS = {
    "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"
}

# Lazy-loaded spaCy model
_NLP = None


def _load_spacy_model():
    """Lazy load spaCy model if available"""
    global _NLP
    if _NLP is None and SPACY_AVAILABLE:
        try:
            # Try to load en_core_web_sm, if not available try en_core_web_md
            model_names = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
            for model_name in model_names:
                try:
                    _NLP = spacy.load(model_name)
                    break
                except OSError:
                    continue
            if _NLP is None:
                raise ImportError("No spaCy model available")
        except Exception as e:
            print(f"Warning: Failed to load spaCy model: {e}")


def extract_entities(text: str) -> List[Entity]:
    """
    Extract entities from text using spaCy.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of extracted Entity objects
    """
    entities: List[Entity] = []
    seen = set()
    
    # 1. Try spaCy NER first if available
    if SPACY_AVAILABLE:
        _load_spacy_model()
        if _NLP is not None:
            doc = _NLP(text)
            for ent in doc.ents:
                if ent.label_ in ACCEPTED_NER_LABELS:
                    key = (ent.text.lower(), ENTITY_TYPE_NAMED)
                    if key not in seen:
                        seen.add(key)
                        entities.append(
                            Entity(
                                entity_type=ENTITY_TYPE_NAMED,
                                text=ent.text,
                                confidence=1.0,
                                ner_label=ent.label_
                            )
                        )
    
    # 2. Extract quoted text
    quoted_pattern = r'"([^"]+)"|\'([^\']+)\''
    for match in re.finditer(quoted_pattern, text):
        quoted_text = match.group(1) or match.group(2)
        if quoted_text and len(quoted_text.strip()) > 1:
            key = (quoted_text.lower(), ENTITY_TYPE_QUOTED)
            if key not in seen:
                seen.add(key)
                entities.append(
                    Entity(
                        entity_type=ENTITY_TYPE_QUOTED,
                        text=quoted_text,
                        confidence=0.9
                    )
                )
    
    # 3. Extract proper nouns (capitalized sequences) - fallback if spaCy not available
    proper_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    for match in re.finditer(proper_pattern, text):
        proper_text = match.group(1)
        # Skip single-letter words and too generic terms
        if (len(proper_text) > 1 
            and proper_text.lower() not in GENERIC_TERMS
            and not proper_text.isupper()):  # Skip ALL CAPS (likely acronyms already handled)
            key = (proper_text.lower(), ENTITY_TYPE_PROPER)
            if key not in seen:
                seen.add(key)
                entities.append(
                    Entity(
                        entity_type=ENTITY_TYPE_PROPER,
                        text=proper_text,
                        confidence=0.7
                    )
                )
    
    # 4. Simple heuristic for topics - clean noun phrases (simplified)
    if SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # Skip too short or too generic chunks
            if (len(chunk_text) > 2 
                and chunk_text.lower() not in GENERIC_TERMS
                and len(chunk_text.split()) <= 4):  # Limit to 4 words
                key = (chunk_text.lower(), ENTITY_TYPE_TOPIC)
                if key not in seen:
                    seen.add(key)
                    entities.append(
                        Entity(
                            entity_type=ENTITY_TYPE_TOPIC,
                            text=chunk_text,
                            confidence=0.6
                        )
                    )
    
    return entities


def extract_entities_simple(text: str) -> List[Entity]:
    """
    Simple fallback entity extraction without spaCy.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of extracted Entity objects
    """
    entities: List[Entity] = []
    seen = set()
    
    # Extract quoted text
    quoted_pattern = r'"([^"]+)"|\'([^\']+)\''
    for match in re.finditer(quoted_pattern, text):
        quoted_text = match.group(1) or match.group(2)
        if quoted_text and len(quoted_text.strip()) > 1:
            key = (quoted_text.lower(), ENTITY_TYPE_QUOTED)
            if key not in seen:
                seen.add(key)
                entities.append(
                    Entity(
                        entity_type=ENTITY_TYPE_QUOTED,
                        text=quoted_text,
                        confidence=0.9
                    )
                )
    
    # Extract proper nouns
    proper_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    for match in re.finditer(proper_pattern, text):
        proper_text = match.group(1)
        if (len(proper_text) > 1 
            and proper_text.lower() not in GENERIC_TERMS
            and not proper_text.isupper()):
            key = (proper_text.lower(), ENTITY_TYPE_PROPER)
            if key not in seen:
                seen.add(key)
                entities.append(
                    Entity(
                        entity_type=ENTITY_TYPE_PROPER,
                        text=proper_text,
                        confidence=0.7
                    )
                )
    
    return entities
