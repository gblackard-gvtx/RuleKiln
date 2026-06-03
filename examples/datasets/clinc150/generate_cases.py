from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import IO

# Hardcoded from clinc_oos "plus" config. Run --discover-labels to verify.
LABEL_NAMES = [
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "oos",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "top_up_savings",
    "hi",
    "whats_the_most_popular_song",
    "freeze_credit_card",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "recipe",
    "change_speed",
    "easy_enroll",
    "what_is_your_name",
    "next_song",
    "cancel_vacation",
    "car_payment",
    "yes",
    "calories",
    "insurance",
    "find_phone",
    "shopping_list_update",
    "plug_type",
    "travel_alert",
    "good_morning",
    "schedule_maintenance",
    "pto_request",
    "directions",
    "payday",
    "flight_status",
    "spending_history",
    "international_fees",
    "who_made_you",
    "pto_request_status",
    "how_old_are_you",
    "account_pin",
    "new_card",
    "rollover_401k",
    "pto_balance",
    "transfers",
    "gas",
    "fun_fact",
    "sync_device",
    "what_can_i_ask_you",
    "play_music",
    "update_playlist",
    "todo_list_update",
    "timezone",
    "cancel",
    "reminder",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "find_nearest_atm",
    "uber_to_airport",
    "alarm",
    "date",
    "spelling",
    "direct_deposit",
    "app_suggestion",
    "change_volume",
    "next_holiday",
    "mpg",
    "oil_change_when",
    "tire_pressure",
    "credit_score",
    "report_fraud",
    "apr",
    "transfer",
    "transactions",
    "credit_limit_change",
    "travel_notification",
    "calendar_update",
    "application_status",
    "minimum_payment",
    "order_checks",
    "are_you_a_bot",
    "taxes",
    "definition",
    "income",
    "vaccines",
    "irs",
    "calculator",
    "roll_dice",
    "goodbye",
    "greet",
    "thank_you",
    "when_is_your_birthday",
]

TRAIN_LIMIT = 500
VAL_LIMIT = 300


def _write_cases_to_file(
    rows: list[dict[str, object]],
    label_names: list[str],
    target_split: str,
    f: IO[str],
    limit: int,
) -> None:
    for i, row in enumerate(rows[:limit]):
        label_id = int(row["intent"])  # type: ignore[arg-type]
        label = label_names[label_id]
        case: dict[str, object] = {
            "schema_version": "rulekiln.case.v1",
            "id": f"clinc150_{target_split}_{i:06d}",
            "split": target_split,
            "task_mode": "classification",
            "input": {"utterance": str(row["text"])},
            "expected": {"label": label},
            "evaluation": {
                "assertions": [
                    {
                        "type": "must_equal",
                        "path": "$.label",
                        "value": label,
                        "weight": 1.0,
                    }
                ]
            },
            "metadata": {"source": "clinc150", "label_id": label_id},
            "weight": 1.0,
        }
        f.write(json.dumps(case) + "\n")


def write_rulekiln_cases(
    rows: list[dict[str, object]],
    label_names: list[str],
    *,
    target_split: str,
    out_path: Path,
    limit: int,
    mode: str = "w",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode) as f:
        _write_cases_to_file(rows, label_names, target_split, f, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RuleKiln cases for CLINC150")
    parser.add_argument(
        "--discover-labels",
        action="store_true",
        help="Print unique labels from first 100 rows and exit (no file writes)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print("datasets not installed; run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset("clinc_oos", "plus", trust_remote_code=False)

    label_names: list[str] = ds["train"].features["intent"].names  # type: ignore[index]

    if args.discover_labels:
        for name in label_names[:100]:
            print(name)
        return

    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [dict(r) for r in ds["train"]]  # type: ignore[index]
    val_rows = [dict(r) for r in ds["validation"]]  # type: ignore[index]

    write_rulekiln_cases(
        train_rows,
        label_names,
        target_split="train",
        out_path=generated_dir / "cases.train.jsonl",
        limit=TRAIN_LIMIT,
    )
    write_rulekiln_cases(
        val_rows,
        label_names,
        target_split="validation",
        out_path=generated_dir / "cases.validation.jsonl",
        limit=VAL_LIMIT,
    )

    combined_path = generated_dir / "cases.jsonl"
    write_rulekiln_cases(
        train_rows,
        label_names,
        target_split="train",
        out_path=combined_path,
        limit=TRAIN_LIMIT,
        mode="w",
    )
    write_rulekiln_cases(
        val_rows,
        label_names,
        target_split="validation",
        out_path=combined_path,
        limit=VAL_LIMIT,
        mode="a",
    )

    (generated_dir / "labels.json").write_text(
        json.dumps(label_names, indent=2) + "\n"
    )

    print(f"Train: {min(len(train_rows), TRAIN_LIMIT)} cases")
    print(f"Validation: {min(len(val_rows), VAL_LIMIT)} cases")
    print(f"Labels: {len(label_names)}")


if __name__ == "__main__":
    main()
