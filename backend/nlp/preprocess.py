import logging
from typing import List

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp: Language = self._load_model(model_name)

    def _load_model(self, model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Install it with: "
                "python -m spacy download %s",
                model_name,
                model_name,
            )
            raise

    def clean(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split())

    def segment_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        doc = self.nlp(self.clean(text))
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
