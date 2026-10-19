
# Voice-Powered Investor Support Assistant

A local, voice-enabled investor support assistant that transcribes
spoken questions, retrieves relevant information from a mutual fund
FAQ, and generates answers grounded in retrieved evidence.

## Features

- Speech-to-text transcription using Faster-Whisper
- Word-level timestamps and confidence estimates
- PDF FAQ ingestion and question-based chunking
- Semantic retrieval using Sentence Transformers and ChromaDB
- Local answer generation using Ollama and Llama 3.2
- FAQ source references in generated answers
- Retrieval and answer-quality evaluation scripts
- JSON outputs and end-to-end runtime measurement

## Tech Stack

- Python
- Faster-Whisper
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2
- PyPDF

## Prerequisites

- Python 3.11
- Git
- Ollama
- A compatible local Ollama model

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the Python dependencies.
4. Install Ollama and download the model.
5. Place the permitted input audio and FAQ PDF in `input/`.
6. Build the knowledge base.
7. Run the assistant.

### Windows PowerShell

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the local model:

```powershell
ollama pull llama3.2:3b
```

Place the input files at these paths:

- `input/investor_sample.mp3`
- `input/SunriseAMC_FAQ.pdf`

Build the knowledge base:

```powershell
python src/build_knowledge_base.py
```

Run the assistant:

```powershell
python src/main.py
```

To transcribe the audio again instead of reusing an existing
transcript:

```powershell
python src/main.py --force-transcribe
```

Evaluate FAQ retrieval:

```powershell
python src/evaluate.py
```

Evaluate the generated answer:

```powershell
python src/evaluate_answers.py
```

## Output

The pipeline writes its results to:

- `output/transcription.json`
- `output/final_answer.json`

The ChromaDB vector database is generated locally under `data/chroma_db/`.

## Evaluation

The project includes retrieval evaluation across ten FAQ queries
and rule-based answer-quality checks.

The answer-quality checks verify expected details and caveats;
they do not prove complete factual correctness or legal accuracy.

## Limitations

- Local inference performance depends on available hardware.
- Tax information is based on the supplied FAQ and must not be
  treated as independently verified current tax advice.
- Evaluation uses a small test set and requires broader testing
  before production use.
