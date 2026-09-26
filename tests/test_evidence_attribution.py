"""Regression tests for fault-evidence attribution by effective sample pieces.

Evidence is attributed by the stretch each sample's *effective* held piece
(TTL-capped, maintenance-subtracted) actually covers inside the fault
interval — not by the sample's timestamp alone:

* a sample taken before the fault started whose value is still held when the
  fault opens (the AND intersection starts later, or the value keeps holding
  after a maintenance window ends) witnesses the fault and appears in its
  evidence events;
* every evidence ``held_intervals`` entry is clipped to the fault's
  half-open ``[start, end)`` and never crosses the fault boundary;
* samples whose piece expired (TTL gap) or was fully subtracted by
  maintenance cover nothing inside the fault and are not evidence.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import (
    Condition,
    Event,
    Interval,
    MaintenanceWindow,
    evaluate_rule,
    to_epoch,
)


def ev(eid, ts, value, device="d", metric="m"):
    return Event(id=eid, device=device, metric=metric, ts=ts, value=value)


def mw(wid, device, start, end, name="检修"):
    return MaintenanceWindow(id=wid, device=device, start=start, end=end, name=name)


def ivs(faults):
    return [(f.start, f.end) for f in faults]


# --------------------------------------------------------------------------- #
# AND intersection opening after a channel's sample
# --------------------------------------------------------------------------- #


def test_and_evidence_includes_pre_fault_sample_still_held():
    # a sampled once at t=0 (held [0,1000)); b opens the AND intersection at
    # t=300. The value in effect over the whole fault is a's t=0 sample.
    a = [ev(1, 0, 90, "d", "a")]
    b = [ev(2, 300, 5, "d", "b"), ev(3, 1000, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 50), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
    )
    assert ivs(result.faults) == [(300, 1000)]
    ev_a, ev_b = result.faults[0].evidence
    # the pre-fault sample is attributed by the stretch its value covers
    assert [w.ts for w in ev_a.events] == [0]
    assert ev_a.held_intervals == [Interval(300, 1000)]
    assert [w.ts for w in ev_b.events] == [300]
    assert ev_b.held_intervals == [Interval(300, 1000)]


def test_and_held_intervals_clipped_to_fault_bounds():
    # a's held interval [0,1000) is wider than the fault on both sides; the
    # evidence sub-intervals must be clipped to the fault's half-open bounds.
    a = [ev(1, 0, 90, "d", "a")]
    b = [ev(2, 300, 5, "d", "b"), ev(3, 600, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 50), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
    )
    assert ivs(result.faults) == [(300, 600)]
    ev_a = result.faults[0].evidence[0]
    assert ev_a.held_intervals == [Interval(300, 600)]
    assert [w.ts for w in ev_a.events] == [0]


def test_and_sample_ending_exactly_at_fault_start_is_not_evidence():
    # a's sample at t=0 is replaced at t=300 (half-open: its piece ends where
    # the fault opens), so only the t=300 sample witnesses the fault.
    a = [ev(1, 0, 90, "d", "a"), ev(2, 300, 90, "d", "a")]
    b = [ev(3, 300, 5, "d", "b"), ev(4, 1000, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 50), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
    )
    assert ivs(result.faults) == [(300, 1000)]
    ev_a = result.faults[0].evidence[0]
    assert [w.ts for w in ev_a.events] == [300]
    assert ev_a.held_intervals == [Interval(300, 1000)]


# --------------------------------------------------------------------------- #
# planned maintenance
# --------------------------------------------------------------------------- #


def test_maintenance_sample_held_past_window_end_witnesses_post_window_fault():
    # a's t=0 sample is held [0,1200); maintenance [300,600) cuts it. The
    # post-window fault [600,1200) is still witnessed by that same sample.
    events = [ev(1, 0, 90), ev(2, 1200, 10)]
    result = evaluate_rule(
        1, "AND", 0, [Condition("d", "m", ">", 50)], {("d", "m"): events},
        horizon=1200, maintenance=[mw(1, "d", 300, 600)],
    )
    assert ivs(result.faults) == [(0, 300), (600, 1200)]
    pre, post = (f.evidence[0] for f in result.faults)
    assert [w.ts for w in pre.events] == [0]
    assert pre.held_intervals == [Interval(0, 300)]
    assert [w.ts for w in post.events] == [0]
    assert post.held_intervals == [Interval(600, 1200)]


def test_maintenance_fully_subtracted_sample_is_not_evidence():
    # the t=0 sample's whole held span [0,600) is subtracted by maintenance;
    # the surviving fault is witnessed only by the t=600 sample.
    events = [ev(1, 0, 90), ev(2, 600, 90), ev(3, 1200, 10)]
    result = evaluate_rule(
        1, "AND", 0, [Condition("d", "m", ">", 50)], {("d", "m"): events},
        horizon=1200, maintenance=[mw(1, "d", 0, 600)],
    )
    assert ivs(result.faults) == [(600, 1200)]
    group = result.faults[0].evidence[0]
    assert [w.ts for w in group.events] == [600]
    assert group.held_intervals == [Interval(600, 1200)]


# --------------------------------------------------------------------------- #
# sample validity period (TTL)
# --------------------------------------------------------------------------- #


def test_ttl_expired_sample_is_not_evidence_for_later_fault():
    # a's t=0 sample expires at t=400 (TTL 400) — before the AND opens at
    # t=500 — so only the t=500 sample witnesses the fault.
    a = [ev(1, 0, 90, "d", "a"), ev(2, 500, 90, "d", "a")]
    b = [ev(3, 500, 5, "d", "b"), ev(4, 1000, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 50), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
        ttl_by_channel={("d", "a"): 400},
    )
    assert ivs(result.faults) == [(500, 900)]
    ev_a = result.faults[0].evidence[0]
    assert [w.ts for w in ev_a.events] == [500]
    assert ev_a.held_intervals == [Interval(500, 900)]


def test_ttl_still_valid_pre_fault_sample_is_evidence():
    # a sampled at t=400 with TTL 400 holds [400,800); the AND opens at
    # t=500 while that sample is still valid, so it witnesses the fault.
    a = [ev(1, 400, 90, "d", "a")]
    b = [ev(2, 500, 5, "d", "b"), ev(3, 700, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 50), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
        ttl_by_channel={("d", "a"): 400},
    )
    assert ivs(result.faults) == [(500, 700)]
    ev_a = result.faults[0].evidence[0]
    assert [w.ts for w in ev_a.events] == [400]
    assert ev_a.held_intervals == [Interval(500, 700)]


# --------------------------------------------------------------------------- #
# reset threshold (latching)
# --------------------------------------------------------------------------- #


def test_latched_pre_fault_sample_held_into_fault_is_evidence():
    # a latches at t=0 (>80, reset 75) and has no later sample, so its value
    # is held over the whole fault that b opens at t=300.
    a = [ev(1, 0, 90, "d", "a")]
    b = [ev(2, 300, 5, "d", "b"), ev(3, 1000, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 80, reset_threshold=75), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
    )
    assert ivs(result.faults) == [(300, 1000)]
    ev_a = result.faults[0].evidence[0]
    assert [w.ts for w in ev_a.events] == [0]
    assert ev_a.held_intervals == [Interval(300, 1000)]


def test_latched_band_sample_witnesses_only_its_own_piece():
    # a arms at t=0 (piece [0,100)), a band sample at t=100 keeps the latch
    # (piece [100,1000)); the fault opens at t=300, so only the band sample's
    # piece reaches it — the arming sample's piece ended at t=100.
    a = [ev(1, 0, 90, "d", "a"), ev(2, 100, 77, "d", "a")]
    b = [ev(3, 300, 5, "d", "b"), ev(4, 1000, 0, "d", "b")]
    result = evaluate_rule(
        1,
        "AND",
        0,
        [Condition("d", "a", ">", 80, reset_threshold=75), Condition("d", "b", ">", 3)],
        {("d", "a"): a, ("d", "b"): b},
        horizon=1000,
    )
    assert ivs(result.faults) == [(300, 1000)]
    ev_a = result.faults[0].evidence[0]
    assert [w.ts for w in ev_a.events] == [100]
    assert ev_a.held_intervals == [Interval(300, 1000)]


# --------------------------------------------------------------------------- #
# services chain: stored faults / page / API / export read the same evidence
# --------------------------------------------------------------------------- #


def at(minute: int) -> float:
    return to_epoch(f"2026-09-23T00:{minute:02d}:00Z")


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def test_stored_fault_evidence_attributes_pre_fault_sample(scratch_db):
    # Channel a has a single high sample at minute 0 held to the horizon;
    # channel b opens the AND at minute 10.  The stored fault's evidence —
    # read back verbatim by the page, /api/faults and the export — must
    # attribute a's pre-fault sample and clip its held interval.
    rows = [
        {"device": "d", "metric": "a", "ts": "2026-09-23T00:00:00Z", "value": 90},
        {"device": "d", "metric": "b", "ts": "2026-09-23T00:10:00Z", "value": 5},
        {"device": "d", "metric": "b", "ts": "2026-09-23T00:20:00Z", "value": 0},
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows))
    rule_id = db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("证据归属", "AND", 0, 0.0),
    ).lastrowid
    for ord_, (metric, threshold) in enumerate((("a", 50), ("b", 3))):
        db.execute(
            "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
            " VALUES (?,?,?,?,?,?)",
            (rule_id, "d", metric, ">", threshold, ord_),
        )
    services.evaluate_all_rules()

    faults = services.query_faults(None, None)
    assert [(f["start"], f["end"]) for f in faults] == [(at(10), at(20))]
    ev_a, ev_b = faults[0]["evidence"]
    assert ev_a["condition"]["metric"] == "a"
    assert [e["ts"] for e in ev_a["events"]] == [at(0)]
    assert ev_a["held_intervals"] == [[at(10), at(20)]]
    assert [e["ts"] for e in ev_b["events"]] == [at(10)]
    assert ev_b["held_intervals"] == [[at(10), at(20)]]
    # the fault-events view carries the same stored evidence verbatim
    events = services.query_fault_events(None, None)
    assert events[0]["members"][0]["evidence"] == faults[0]["evidence"]
