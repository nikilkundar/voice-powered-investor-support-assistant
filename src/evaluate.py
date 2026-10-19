
from test_retrieval import retrieve_faqs


TEST_CASES = [
    ("What documents are required for KYC?", "Q1"),
    ("How long does KYC verification take?", "Q2"),
    ("Can a minor invest in mutual funds?", "Q3"),
    ("What is the minimum SIP amount?", "Q4"),
    ("How can I pause my SIP?", "Q5"),
    ("What happens if a SIP payment fails?", "Q6"),
    ("How long does redemption take for equity funds?", "Q7"),
    ("Can I redeem only part of my investment?", "Q8"),
    ("How are equity mutual fund gains taxed after 14 months?", "Q9"),
    ("Will TDS be deducted on mutual fund redemptions?", "Q10"),
]


def main():
    passed = 0

    print("Running FAQ retrieval evaluation...\n")

    for question, expected_faq in TEST_CASES:
        results = retrieve_faqs(question, n_results=3)
        retrieved_ids = [item["faq_id"] for item in results]

        success = expected_faq in retrieved_ids
        passed += int(success)

        status = "PASS" if success else "FAIL"
        print(f"[{status}] {question}")
        print(f"       Expected: {expected_faq}")
        print(f"       Retrieved: {retrieved_ids}\n")

    total = len(TEST_CASES)
    print(f"Recall@3: {passed}/{total} ({passed / total:.1%})")


if __name__ == "__main__":
    main()
