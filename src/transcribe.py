
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel


# Find the project root, regardless of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_PATH = PROJECT_ROOT / "input" / "investor_sample.mp3"
OUTPUT_PATH = PROJECT_ROOT / "output" / "transcription.json"


def transcribe_audio():
    """Transcribe an audio file and save structured JSON output."""

    # Check that the audio file exists.
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    # Create the output folder if it doesn't exist.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading the Whisper tiny model...")

    # CPU + int8 is a memory-conscious starting configuration.
    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing audio...")
    start_time = time.perf_counter()

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        beam_size=3,
        word_timestamps=True,
        vad_filter=True,
    )

    transcript_parts = []
    word_results = []
    segment_results = []

    # Faster-Whisper generates segments as we iterate over them.
    for segment in segments:
        transcript_parts.append(segment.text.strip())

        segment_results.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        })

        # Word-level metadata is available because we enabled
        # word_timestamps=True above.
        for word in segment.words or []:
            word_results.append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "confidence": round(word.probability, 4),
            })

    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    transcript = " ".join(transcript_parts).strip()

    if not transcript:
        print("Warning: no speech was detected.")

    result = {
        "audio_file": AUDIO_PATH.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "transcript": transcript,
        "segments": segment_results,
        "words": word_results,
        "metrics": {
            "transcription_time_seconds": elapsed_seconds,
            "word_count": len(word_results),
        },
    }

    # Write readable JSON to disk.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nTranscription complete!")
    print(f"Transcript: {transcript}")
    print(f"Detected language: {info.language}")
    print(f"Words recorded: {len(word_results)}")
    print(f"Time taken: {elapsed_seconds} seconds")
    print(f"JSON saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        transcribe_audio()
    except FileNotFoundError as error:
        print(f"Input error: {error}")
    except Exception as error:
        print(f"Transcription failed: {error}")
        raise
