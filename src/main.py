
import argparse
import json
import time
from pathlib import Path

from transcribe import transcribe_audio
from answer_question import answer_question


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = PROJECT_ROOT / "output" / "transcription.json"
FINAL_OUTPUT_PATH = PROJECT_ROOT / "output" / "final_answer.json"


def run_pipeline(force_transcribe=False):
    start_time = time.perf_counter()

    # Transcribe again if requested or if no saved transcript exists.
    if force_transcribe or not TRANSCRIPT_PATH.exists():
        print("Transcribing audio...")
        transcribe_audio()
    else:
        print("Loading existing transcription...")
        
    if not TRANSCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Transcription file was not created: {TRANSCRIPT_PATH}"
        )

    with TRANSCRIPT_PATH.open("r", encoding="utf-8") as file:
        transcription = json.load(file)

    question = transcription.get("transcript", "").strip()

    if not question:
        raise ValueError(
            "No speech was detected. Cannot answer an empty question."
        )

    print(f"\nInvestor question:\n{question}")
    print("\nGenerating answer using the FAQ...")

    result = answer_question(question)

    final_result = {
        "transcription": transcription,
        "answer": result["answer"],
        "sources": result["sources"],
        "metrics": {
            "end_to_end_seconds": round(
                time.perf_counter() - start_time, 3
            )
        },
    }

    FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with FINAL_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nGenerated answer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source["faq_id"])

    print(f"\nFinal output saved to: {FINAL_OUTPUT_PATH}")
    print(
        "End-to-end time:",
        final_result["metrics"]["end_to_end_seconds"],
        "seconds",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the voice-powered investor support assistant."
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Transcribe the audio again instead of reusing the saved transcript.",
    )
    args = parser.parse_args()

    try:
        run_pipeline(force_transcribe=args.force_transcribe)
    except Exception as error:
        print(f"Pipeline failed: {error}")
        raise
