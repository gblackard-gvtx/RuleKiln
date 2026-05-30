from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

LABEL_NAMES = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def download_parquet(filename: str, out_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id="PolyAI/banking77",
            repo_type="dataset",
            filename=filename,
            revision="refs/pr/7",
            local_dir=out_dir,
        )
    )


def normalize_label(raw_label: object) -> str:
    if isinstance(raw_label, str):
        return raw_label

    label_id = int(raw_label)
    return LABEL_NAMES[label_id]


def write_rulekiln_cases(
    df: pd.DataFrame,
    *,
    target_split: str,
    out_path: Path,
    limit: int | None = None,
) -> None:
    if limit is not None:
        df = df.head(limit)

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            label = normalize_label(row["label"])

            case = {
                "schema_version": "rulekiln.case.v1",
                "id": f"banking77_{target_split}_{i:06d}",
                "split": target_split,
                "task_mode": "classification",
                "input": {
                    "utterance": row["text"],
                },
                "expected": {
                    "label": label,
                },
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
                "metadata": {
                    "source": "banking77",
                    "label_id": int(row["label"]) if not isinstance(row["label"], str) else None,
                },
                "weight": 1.0,
            }

            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "raw"
    generated_dir = base_dir / "generated"

    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_path = download_parquet("data/train-00000-of-00001.parquet", raw_dir)
    test_path = download_parquet("data/test-00000-of-00001.parquet", raw_dir)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Loaded train rows: {len(train_df)}")
    print(f"Loaded test rows: {len(test_df)}")
    print(train_df.head())

    # Optional separate files for debugging/reference.
    write_rulekiln_cases(
        train_df,
        target_split="train",
        out_path=generated_dir / "cases.train.jsonl",
        limit=500,
    )

    write_rulekiln_cases(
        test_df,
        target_split="validation",
        out_path=generated_dir / "cases.validation.jsonl",
        limit=300,
    )

    # Single upload file for the current RuleKiln UI.
    combined_path = generated_dir / "cases.jsonl"

    write_rulekiln_cases(
        train_df,
        target_split="train",
        out_path=combined_path,
        limit=500,
    )

    write_rulekiln_cases(
        test_df,
        target_split="validation",
        out_path=combined_path,
        limit=300,
    )

    (generated_dir / "labels.json").write_text(
        json.dumps(LABEL_NAMES, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote separate files to {generated_dir}")
    print(f"Wrote combined upload file to {combined_path}")


if __name__ == "__main__":
    main()
