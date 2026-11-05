"""Sanity checks on data/claims.json. No API calls."""
import json
from collections import Counter
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
CLAIMS = json.loads((DATA / "claims.json").read_text())
REQUIRED = {"id", "policyholder", "owner_user_id", "policy_doc", "type",
            "amount", "currency", "deductible", "status", "date_filed", "notes"}
STATUSES = {"submitted", "under_review", "approved", "denied", "paid"}


def test_is_list_of_claims():
    assert isinstance(CLAIMS, list) and len(CLAIMS) >= 15


def test_every_claim_has_required_fields():
    for c in CLAIMS:
        assert REQUIRED <= c.keys(), f"{c.get('id')} missing {REQUIRED - c.keys()}"


def test_ids_unique_and_c1042_present():
    ids = [c["id"] for c in CLAIMS]
    assert len(ids) == len(set(ids))
    assert "C-1042" in ids  # used by the Module 3 test prompt


def test_values_are_valid():
    for c in CLAIMS:
        assert c["status"] in STATUSES, c["id"]
        assert isinstance(c["amount"], (int, float)) and c["amount"] >= 0, c["id"]
        assert date.fromisoformat(c["date_filed"]) <= date.today(), c["id"]


def test_policy_doc_exists():
    for c in CLAIMS:
        assert (DATA / "policies" / c["policy_doc"]).exists(), f"{c['id']}: {c['policy_doc']} not found"


def test_owner_maps_to_one_policyholder():
    pairs = {(c["owner_user_id"], c["policyholder"]) for c in CLAIMS}
    owners = Counter(o for o, _ in pairs)
    assert all(n == 1 for n in owners.values()), "an owner_user_id has two names"


def test_enough_variety_for_access_control_tests():
    owners = Counter(c["owner_user_id"] for c in CLAIMS)
    assert len(owners) >= 3 and all(n >= 2 for n in owners.values())
