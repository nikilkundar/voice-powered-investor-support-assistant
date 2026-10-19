
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "output" / "final_answer.json"


def check_answer(answer_data):
    answer = answer_data.get("answer", "")
    answer_lower = answer.lower()
    normalized_answer = answer_lower.replace("–", "-").replace("—", "-")

    # The pipeline stores source IDs separately from the answer text.
    sources = {
        str(source.get("faq_id", "")).upper()
        for source in answer_data.get("sources", [])
    }

    checks = {
        "Cites FAQ Q9": (
            "Q9" in sources
            or bool(re.search(r"\[Q9\]", answer, re.IGNORECASE))
        ),
        "Cites FAQ Q10": (
            "Q10" in sources
            or bool(re.search(r"\[Q10\]", answer, re.IGNORECASE))
        ),
        "Includes LTCG rate of 12.5%": (
            "12.5%" in answer
        ),
        "Includes Rs. 1.25 lakh threshold": (
            "1.25 lakh" in answer_lower
        ),
        "Distinguishes resident and NRI TDS": (
            "resident" in answer_lower
            and "nri" in answer_lower
            and "tds" in answer_lower
        ),
        "Mentions FY 2025-26 limitation": (
            "2025-26" in normalized_answer
            and ("q9" in answer_lower or "faq" in answer_lower)
        ),
        "Recommends consulting a tax advisor": (
            "tax advisor" in answer_lower
            or "tax adviser" in answer_lower
        ),
    }

    print("\nANSWER QUALITY EVALUATION")
    print("-" * 45)

    passed = 0

    for description, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {description}")
        passed += result

    total = len(checks)
    print("-" * 45)
    print(f"Score: {passed}/{total} ({passed / total:.0%})")

    return passed == total


def main():
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Output file not found: {OUTPUT_PATH}\n"
            "Run python src/main.py first."
        )

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        answer_data = json.load(file)

    success = check_answer(answer_data)

    if not success:
        print(
            "\nSome checks failed. Review the answer-generation "
            "prompt and run the pipeline again."
        )


if __name__ == "__main__":
    main()
