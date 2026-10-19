
# Technical Decisions

## 1. Speech-to-Text

**Choice:** Faster-Whisper with the tiny model on CPU.

**Reason:** A lightweight model is suitable for a laptop with
limited RAM and VRAM. CPU inference avoids requiring a dedicated
GPU setup.

**Trade-off:** A smaller model can be less accurate than larger
models, particularly for noisy speech or unfamiliar terminology.

## 2. FAQ Chunking

**Choice:** One FAQ question and its answer per chunk.

**Reason:** Each FAQ entry is a self-contained unit of information.
Keeping the question and answer together makes retrieval easier
to interpret and supports question-specific citations.

**Trade-off:** This approach works well for the current structured
FAQ but may need refinement for long documents or information
spanning multiple sections.

## 3. Embeddings and Retrieval

**Choice:** Sentence Transformers with all-MiniLM-L6-v2 and
ChromaDB using cosine distance.

**Reason:** This provides local semantic search without a paid
embedding API.

**Trade-off:** Retrieval quality depends on the embedding model,
chunking strategy, and similarity ranking. The current evaluation
set is small.

## 4. Local Language Model

**Choice:** Ollama with Llama 3.2 3B.

**Reason:** Local inference avoids per-request API charges and
keeps the core workflow independent of paid hosted model APIs.

**Trade-off:** A small model may omit instructions or generate
unsupported statements. Prompting alone is not a guarantee of
correctness.

## 5. Grounding and Citations

**Choice:** Retrieve FAQ evidence before generating an answer and
include FAQ IDs in the response.

**Reason:** This makes answers easier to trace to the source
document and discourages unsupported responses.

**Trade-off:** Citation presence does not guarantee that every
claim is supported by the cited source.

## 6. Evaluation

**Choice:** Separate retrieval evaluation from answer-quality
checks.

**Reason:** Retrieval and generation are different stages and
should be evaluated separately.

**Results:** The current retrieval test set achieved 10/10
Recall@3. The current answer-quality checks passed 7/7 for the
tested investor question.

**Limitation:** These results apply only to the current test cases.
They do not establish general accuracy or production readiness.

## 7. Production Readiness

Before production use, I would add broader test coverage, robust
handling of unsupported questions, structured citation validation,
logging, latency monitoring, privacy controls, and human review
for high-impact financial information.
