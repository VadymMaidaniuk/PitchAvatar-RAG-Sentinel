from __future__ import annotations

from dataclasses import dataclass

from pitchavatar_rag_sentinel.utils.naming import unique_document_id


@dataclass(frozen=True)
class SeedDocument:
    document_id: str
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class SearchCorpus:
    doc_a: str
    doc_b: str
    doc_c: str
    doc_d: str
    doc_e: str

    def core_ids(self) -> list[str]:
        return [self.doc_a, self.doc_b, self.doc_c]

    def quantum_range_ids(self) -> list[str]:
        return [self.doc_a, self.doc_c, self.doc_d]


def build_search_corpus(namespace: str) -> tuple[SearchCorpus, list[SeedDocument]]:
    suffix = unique_document_id(namespace, "search")
    corpus = SearchCorpus(
        doc_a=f"{suffix}-quantum-computing",
        doc_b=f"{suffix}-organic-gardening",
        doc_c=f"{suffix}-quantum-physics",
        doc_d=f"{suffix}-classical-physics",
        doc_e=f"{suffix}-machine-learning",
    )

    documents = [
        SeedDocument(
            document_id=corpus.doc_a,
            content=(
                "Quantum computing harnesses quantum mechanical phenomena like superposition "
                "and entanglement to process information in fundamentally new ways. Unlike "
                "classical computers that use binary bits representing either zero or one, "
                "quantum computers use qubits that can exist in multiple states simultaneously. "
                "This capability enables exponential speedup for certain computational problems."
            ),
            metadata={"user_id": "user-alpha", "type": "pdf"},
        ),
        SeedDocument(
            document_id=corpus.doc_b,
            content=(
                "Organic gardening relies on natural methods to grow healthy plants without "
                "synthetic fertilizers or chemical pesticides. Composting kitchen scraps and "
                "crop rotation prevent soil depletion and improve plant health."
            ),
            metadata={"user_id": "user-alpha", "type": "docx"},
        ),
        SeedDocument(
            document_id=corpus.doc_c,
            content=(
                "Quantum physics experiments have revealed the strange nature of subatomic "
                "reality. Bell test experiments have confirmed quantum entanglement, and the "
                "double-slit experiment demonstrates wave-particle duality."
            ),
            metadata={"user_id": "user-beta", "type": "pdf"},
        ),
        SeedDocument(
            document_id=corpus.doc_d,
            content=(
                "Classical mechanics describes the motion of macroscopic objects from "
                "projectiles to planetary orbits. Classical physics works well for large-scale "
                "phenomena, but quantum effects become significant at atomic scales."
            ),
            metadata={"user_id": "user-gamma", "type": "pdf"},
        ),
        SeedDocument(
            document_id=corpus.doc_e,
            content=(
                "Machine learning enables computers to learn patterns from datasets without "
                "being explicitly programmed for each task. Deep learning excels at processing "
                "text, images, and audio."
            ),
            metadata={"user_id": "user-gamma", "type": "txt"},
        ),
    ]
    return corpus, documents

