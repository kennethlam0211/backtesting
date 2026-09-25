"""Comprehensive state-tracking tests for TopStep ExamSimulator.
Tests every variable at every transition point across all modes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topstep import ExamSimulator

# All tracked variables
TRACKED = [
    'acc_pnl', 'daily_pnl', 'pass_exam', 'daily_locked', 'daily_target_hit',
    'daily_resets', 'exam_passed_today', 'mdd_limit', 'max_contracts',
    'subscription_days', 'exam_success', 'exam_counts', 'topstep_cost',
    'success_activation_days', 'back_to_fund_cnt', 'payout_cnt', 'activation_cnt',
    'exam_daily_target_hit_days',
]
LIST_TRACKED = [
    'daily_pnls', 'days_taken_to_pass_exams', 'final_pnl_list',
    'exam_daily_target_hit_day_list', 'exam_daily_resets_list', 'back_to_fund_list',
]


def snap(ts, label=""):
    """Snapshot all tracked state."""
    s = {k: getattr(ts, k) for k in TRACKED}
    s.update({k: list(getattr(ts, k)) for k in LIST_TRACKED})  # copy lists
    if label:
        print(f"  [{label}]")
        for k in TRACKED:
            print(f"    {k:30s} = {s[k]}")
        for k in LIST_TRACKED:
            print(f"    {k:30s} = {s[k]}")
    return s


def check(ts, label, **expected):
    """Assert specific fields match expected values."""
    for k, v in expected.items():
        actual = getattr(ts, k)
        if isinstance(v, list):
            actual = list(actual)
        assert actual == v, f"[{label}] {k}: expected {v}, got {actual}"


def sim_bar(ts, pnl, max_risk=0, day_end=False, traded=1):
    return ts.update(pnl, max_risk, day_end, trading_fees=0, traded_contracts=traded)


def sim_day(ts, pnls):
    results = []
    for i, pnl in enumerate(pnls):
        day_end = (i == len(pnls) - 1)
        r = sim_bar(ts, pnl, max_risk=-abs(pnl)*2, day_end=day_end)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════
# train_exam: Full state tracking
# ═══════════════════════════════════════════════════════════════════
def test_train_exam_full_state():
    print("\n" + "="*80)
    print("train_exam: FULL STATE TRACKING")
    print("="*80)

    # ── INIT ──
    ts = ExamSimulator(mode='train_exam')
    print("\n--- INIT ---")
    check(ts, "init",
          pass_exam=False, daily_locked=False, daily_target_hit=False,
          daily_resets=0, exam_passed_today=False, daily_pnl=0, acc_pnl=0,
          max_contracts=5, subscription_days=1, exam_success=0, exam_counts=1,
          mdd_limit=-2000, success_activation_days=0, payout_cnt=0,
          activation_cnt=0, exam_daily_target_hit_days=0, back_to_fund_cnt=0)
    snap(ts, "init")

    # ── Route 1: Normal trading, no target hit, day end ──
    print("\n--- Route 1: Normal day (no target, no MDD) ---")
    sim_bar(ts, pnl=500)
    check(ts, "mid-day +500",
          acc_pnl=500, daily_pnl=500, daily_locked=False, daily_target_hit=False)
    sim_bar(ts, pnl=200, day_end=True)
    check(ts, "day end +700",
          acc_pnl=700, daily_pnl=0, daily_locked=False,
          subscription_days=2)
    assert ts.daily_pnls == [700], f"daily_pnls: {ts.daily_pnls}"
    snap(ts, "after day 1")

    # ── Route 2: Hit daily target (1500), lock ──
    print("\n--- Route 2: Daily target hit -> lock ---")
    sim_bar(ts, pnl=800)
    check(ts, "mid-day +800", daily_locked=False, daily_target_hit=False)
    sim_bar(ts, pnl=800)  # daily_pnl = 1600 >= 1500
    check(ts, "daily target hit",
          daily_locked=True, daily_target_hit=True, daily_pnl=1600)
    # More PnL while locked — still locked
    sim_bar(ts, pnl=100)
    check(ts, "locked, extra trade", daily_locked=True, daily_target_hit=True)
    # Day end — unlock, record
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end unlock",
          daily_locked=False, daily_target_hit=False, daily_pnl=0,
          exam_daily_target_hit_days=1, subscription_days=3)
    assert ts.exam_daily_target_hit_day_list == [False, True], f"hit_list: {ts.exam_daily_target_hit_day_list}"
    assert ts.exam_daily_resets_list == [0, 0], f"resets_list: {ts.exam_daily_resets_list}"
    snap(ts, "after day 2 (target hit)")

    # ── Route 3: MDD blow mid-day ──
    print("\n--- Route 3: MDD blow mid-day ---")
    pre_daily_pnl = ts.daily_pnl
    sim_bar(ts, pnl=-2500, max_risk=-2500)  # MDD blow
    check(ts, "after MDD blow",
          acc_pnl=0, pass_exam=False,  # reset
          daily_resets=1)  # incremented
    # daily_pnl should persist (not reset by _reset_exam)
    assert ts.daily_pnl != 0, f"daily_pnl should persist: {ts.daily_pnl}"
    assert ts.daily_pnl == pre_daily_pnl + (-2500), f"daily_pnl: {ts.daily_pnl}"
    # Trade again after MDD reset (same day)
    sim_bar(ts, pnl=300)
    check(ts, "trade after reset", daily_resets=1)
    # Day end
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end after MDD",
          daily_pnl=0, daily_resets=0, daily_locked=False)
    assert ts.exam_daily_resets_list[-1] == 1, f"last resets: {ts.exam_daily_resets_list[-1]}"
    snap(ts, "after day 3 (MDD blow)")

    # ── Route 4: Multiple MDD blows same day ──
    print("\n--- Route 4: Multiple MDD blows same day ---")
    sim_bar(ts, pnl=-2500, max_risk=-2500)  # blow 1
    check(ts, "blow 1", daily_resets=1, daily_locked=False)
    sim_bar(ts, pnl=-2500, max_risk=-2500)  # blow 2
    check(ts, "blow 2", daily_resets=2, daily_locked=False)
    # Blow 3 — hits MAX_DAILY_RESETS (>= 2), locks
    sim_bar(ts, pnl=-2500, max_risk=-2500)
    check(ts, "blow 3 (at limit)", daily_resets=2, daily_locked=True)
    # Day end
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end after max resets",
          daily_pnl=0, daily_resets=0, daily_locked=False)
    snap(ts, "after day 4 (max resets)")

    # ── Route 5: Exam pass ──
    print("\n--- Route 5: Exam pass ---")
    ts2 = ExamSimulator(mode='train_exam')
    # Day 1: win 1600 -> daily target hit
    sim_day(ts2, [800, 800, 0])
    check(ts2, "day 1 end",
          acc_pnl=1600, daily_pnl=0, daily_locked=False,
          exam_daily_target_hit_days=1)
    # Day 2: win 1600 -> acc_pnl >= 3000 -> exam pass
    sim_bar(ts2, pnl=800)
    sim_bar(ts2, pnl=800)  # acc_pnl = 3200 >= 3000
    check(ts2, "exam passed mid-bar",
          pass_exam=True, exam_passed_today=True, daily_locked=True,
          acc_pnl=0,  # reset on pass
          exam_success=1, max_contracts=2)  # INIT_MAX_CONTRACTS
    # daily_pnl should persist
    assert ts2.daily_pnl == 1600, f"daily_pnl after pass: {ts2.daily_pnl}"
    assert len(ts2.days_taken_to_pass_exams) == 1
    # Day end — transition
    sim_bar(ts2, pnl=0, day_end=True)
    check(ts2, "day end after pass",
          daily_locked=False, exam_passed_today=False, daily_pnl=0,
          pass_exam=True)  # stays in activation
    snap(ts2, "after exam pass")

    # ── Route 6: Exam pass → activation → MDD → back to exam ──
    print("\n--- Route 6: Post-pass activation blow ---")
    # Next day: in activation, MDD blow
    sim_bar(ts2, pnl=-2500, max_risk=-2500)
    check(ts2, "activation MDD",
          pass_exam=False, acc_pnl=0)  # back to exam via _reset_exam
    # Day end
    sim_bar(ts2, pnl=0, day_end=True)
    check(ts2, "day end back in exam",
          pass_exam=False, daily_pnl=0)
    snap(ts2, "back in exam after activation blow")

    # ── Route 7: daily_pnl persists across MDD resets ──
    print("\n--- Route 7: daily_pnl persistence ---")
    ts3 = ExamSimulator(mode='train_exam')
    sim_bar(ts3, pnl=500)  # daily_pnl = 500
    sim_bar(ts3, pnl=-2500, max_risk=-2500)  # MDD blow, acc_pnl reset
    assert ts3.daily_pnl == -2000, f"daily_pnl after MDD: {ts3.daily_pnl}"
    assert ts3.acc_pnl == 0, f"acc_pnl should reset: {ts3.acc_pnl}"
    sim_bar(ts3, pnl=1000)  # trade again
    assert ts3.daily_pnl == -1000, f"daily_pnl continues: {ts3.daily_pnl}"
    sim_bar(ts3, pnl=0, day_end=True)
    assert ts3.daily_pnls[-1] == -1000, f"daily_pnls recorded: {ts3.daily_pnls[-1]}"
    assert ts3.daily_pnl == 0

    # ── Route 8: MDD blow then hit daily target same day ──
    print("\n--- Route 8: MDD blow + daily target same day ---")
    ts8 = ExamSimulator(mode='train_exam')
    # Blow MDD
    sim_bar(ts8, pnl=-2500, max_risk=-2500)
    check(ts8, "after MDD blow", acc_pnl=0, daily_resets=1, daily_locked=False, daily_target_hit=False)
    # Trade again — smaller wins to hit daily target without passing exam
    # daily_pnl = -2500, need daily_pnl >= 1500 → need +4000 more
    # but acc_pnl starts at 0, acc_pnl >= 3000 triggers exam pass first
    # So: win enough for daily target but NOT exam target
    # daily_pnl needs +4000 → acc_pnl = 4000 → passes exam
    # Can't hit daily target without passing exam after MDD blow from 0
    # Instead: start with some acc_pnl, blow, then recover to daily target only
    ts8b = ExamSimulator(mode='train_exam')
    sim_day(ts8b, [500, 0])  # day 1: acc=500
    # day 2: blow, then recover
    sim_bar(ts8b, pnl=-2600, max_risk=-2600)  # MDD blow, acc=0, daily_resets=1
    check(ts8b, "MDD blow", acc_pnl=0, daily_resets=1)
    # daily_pnl = -2600, need to reach 1500 → need +4100
    # acc_pnl would be 4100 → passes exam (>= 3000)
    # So daily target and exam pass happen together at acc=3000
    # Let's just test that after MDD + recovery, daily_resets is recorded
    sim_bar(ts8b, pnl=800)
    sim_bar(ts8b, pnl=800)  # acc=1600, daily_pnl=-1000
    check(ts8b, "recovering", daily_resets=1, daily_locked=False, daily_target_hit=False)
    sim_bar(ts8b, pnl=0, day_end=True)
    assert ts8b.exam_daily_resets_list[-1] == 1, f"should record 1 reset: {ts8b.exam_daily_resets_list}"
    assert ts8b.exam_daily_target_hit_day_list[-1] == False, f"no target hit: {ts8b.exam_daily_target_hit_day_list}"
    snap(ts8b, "MDD + partial recovery same day")

    # ── Route 9: MDD blow, hit daily target, then exam pass same day ──
    print("\n--- Route 9: MDD blow + target + exam pass same day ---")
    ts9 = ExamSimulator(mode='train_exam')
    # Day 1: big win
    sim_day(ts9, [1600, 0])  # acc=1600, daily target hit
    check(ts9, "day 1 end", exam_daily_target_hit_days=1)
    # Day 2: MDD blow, then recover to pass
    sim_bar(ts9, pnl=-2500, max_risk=-2500)  # MDD blow, acc=0
    check(ts9, "MDD blow", acc_pnl=0, daily_resets=1)
    # Trade big to pass: need acc >= 3000 from fresh start
    for _ in range(4):
        sim_bar(ts9, pnl=800)  # acc = 3200
    check(ts9, "exam passed after MDD",
          pass_exam=True, exam_passed_today=True, exam_success=1)
    # Day end
    sim_bar(ts9, pnl=0, day_end=True)
    check(ts9, "day end after pass",
          daily_locked=False, exam_passed_today=False, daily_pnl=0)
    snap(ts9, "MDD + pass same day")

    print("\n  [train_exam] ALL STATE TRACKING PASSED")


# ═══════════════════════════════════════════════════════════════════
# train_activation: Full state tracking
# ═══════════════════════════════════════════════════════════════════
def test_train_activation_full_state():
    print("\n" + "="*80)
    print("train_activation: FULL STATE TRACKING")
    print("="*80)

    # ── INIT ──
    ts = ExamSimulator(mode='train_activation', exam_success=5, exam_counts=10)
    print("\n--- INIT ---")
    check(ts, "init",
          pass_exam=True, max_contracts=2, activation_cnt=1,
          daily_locked=False, daily_pnl=0, acc_pnl=0,
          mdd_limit=-2000)
    snap(ts, "init")

    # ── Route 1: Normal trading days ──
    print("\n--- Route 1: Normal trading ---")
    sim_day(ts, [200, 200, 200])
    check(ts, "day 1 end",
          acc_pnl=600, daily_pnl=0, success_activation_days=1)
    assert ts.daily_pnls == [600]
    snap(ts, "after day 1")

    # ── Route 2: MDD blow → reset ──
    print("\n--- Route 2: MDD blow ---")
    pre_cost = ts.topstep_cost
    sim_bar(ts, pnl=-3000, max_risk=-3000)
    check(ts, "after MDD blow",
          acc_pnl=0, pass_exam=True,  # stays in activation
          activation_cnt=2, success_activation_days=0,
          mdd_limit=-2000)
    assert ts.topstep_cost < pre_cost, "cost should increase"
    # daily_pnl persists (day 1 ended, so starts fresh at -3000)
    assert ts.daily_pnl == -3000, f"daily_pnl: {ts.daily_pnl}"
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end after MDD", daily_pnl=0)
    snap(ts, "after MDD blow + day end")

    # ── Route 3: Build to payout ──
    print("\n--- Route 3: Payout ---")
    ts2 = ExamSimulator(mode='train_activation', exam_success=5, exam_counts=10)
    for day in range(8):
        sim_day(ts2, [300, 300, 300])
    check(ts2, "after 8 days", payout_cnt=1)
    assert ts2.payouts > 0, f"payouts: {ts2.payouts}"
    snap(ts2, "after payout")

    # ── Route 4: MDD relaxation ──
    print("\n--- Route 4: MDD relaxation ---")
    ts3 = ExamSimulator(mode='train_activation', exam_success=5, exam_counts=10)
    for day in range(4):
        sim_day(ts3, [300, 300, 0])
    assert ts3.mdd_limit > -2000, f"mdd should relax: {ts3.mdd_limit}"
    snap(ts3, "after MDD relaxation")

    # ── Route 5: Back-to-fund tracking ──
    print("\n--- Route 5: Back-to-fund ---")
    ts4 = ExamSimulator(mode='train_activation', exam_success=5, exam_counts=10)
    for i in range(5):
        sim_bar(ts4, pnl=-3000, max_risk=-3000)
    print(f"  back_to_fund_cnt: {ts4.back_to_fund_cnt}")
    print(f"  back_to_fund_list: {ts4.back_to_fund_list}")
    print(f"  activation_cnt: {ts4.activation_cnt}")
    assert ts4.activation_cnt == 6  # init(1) + 5 blows
    assert len(ts4.final_pnl_list) >= 6  # init cost + 5 reset costs
    snap(ts4, "after 5 blows")

    print("\n  [train_activation] ALL STATE TRACKING PASSED")


# ═══════════════════════════════════════════════════════════════════
# eval: Full lifecycle state tracking
# ═══════════════════════════════════════════════════════════════════
def test_eval_full_state():
    print("\n" + "="*80)
    print("eval: FULL LIFECYCLE STATE TRACKING")
    print("="*80)

    # ── INIT ──
    ts = ExamSimulator(mode='eval')
    print("\n--- INIT ---")
    check(ts, "init",
          pass_exam=False, max_contracts=5, daily_locked=False,
          exam_passed_today=False, acc_pnl=0)
    snap(ts, "init")

    # ── Route 1: Exam phase → daily target → day end ──
    print("\n--- Route 1: Exam daily target hit ---")
    sim_bar(ts, pnl=800)
    sim_bar(ts, pnl=800)  # daily_pnl = 1600 >= 1500
    check(ts, "daily target hit",
          daily_locked=True, daily_target_hit=True, pass_exam=False)
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end",
          daily_locked=False, daily_target_hit=False, daily_pnl=0,
          exam_daily_target_hit_days=1)
    snap(ts, "after exam day target hit")

    # ── Route 2: Pass exam → locked → day end → activation ──
    print("\n--- Route 2: Exam pass -> activation ---")
    sim_bar(ts, pnl=800)
    sim_bar(ts, pnl=800)  # acc_pnl >= 3000
    check(ts, "exam passed",
          pass_exam=True, exam_passed_today=True, daily_locked=True,
          acc_pnl=0, max_contracts=2, exam_success=1)
    # daily_pnl persists
    assert ts.daily_pnl == 1600, f"daily_pnl: {ts.daily_pnl}"
    # Day end — transition to activation
    sim_bar(ts, pnl=0, day_end=True)
    check(ts, "day end -> activation",
          daily_locked=False, exam_passed_today=False,
          pass_exam=True, daily_pnl=0)
    snap(ts, "in activation after pass")

    # ── Route 3: Activation trading → payout ──
    print("\n--- Route 3: Activation payout ---")
    for day in range(8):
        sim_day(ts, [300, 300, 300])
    print(f"  payout_cnt: {ts.payout_cnt}, payouts: {ts.payouts:.0f}")
    snap(ts, "after activation payouts")

    # ── Route 4: Activation MDD → reactivation (back-to-fund) ──
    print("\n--- Route 4: Activation MDD -> reactivation ---")
    ts2 = ExamSimulator(mode='eval', exam_success=1, exam_counts=50)  # low pass rate
    # Pass exam
    sim_day(ts2, [800, 800, 0])
    sim_day(ts2, [800, 800, 0])
    assert ts2.pass_exam == True
    snap(ts2, "exam passed (low rate)")
    # Blow 1 — check reactivation
    sim_bar(ts2, pnl=-3000, max_risk=-3000)
    check(ts2, "activation MDD #1", pass_exam=True)  # stays if reactivation
    print(f"  back_to_fund_cnt: {ts2.back_to_fund_cnt}")
    # Blow 2
    sim_bar(ts2, pnl=-3000, max_risk=-3000)
    print(f"  back_to_fund_cnt: {ts2.back_to_fund_cnt}, pass_exam: {ts2.pass_exam}")
    # Blow 3 — should go back to exam
    sim_bar(ts2, pnl=-3000, max_risk=-3000)
    print(f"  back_to_fund_cnt: {ts2.back_to_fund_cnt}, pass_exam: {ts2.pass_exam}")
    snap(ts2, "after 3 activation blows")

    # ── Route 5: Exam MDD blow in eval ──
    print("\n--- Route 5: Exam MDD in eval ---")
    ts3 = ExamSimulator(mode='eval')
    sim_bar(ts3, pnl=500)
    pre_exam_counts = ts3.exam_counts
    sim_bar(ts3, pnl=-3000, max_risk=-3000)
    check(ts3, "exam MDD",
          acc_pnl=0, pass_exam=False)
    assert ts3.exam_counts == pre_exam_counts + 1
    # daily_pnl persists
    assert ts3.daily_pnl == 500 + (-3000), f"daily_pnl: {ts3.daily_pnl}"
    sim_bar(ts3, pnl=0, day_end=True)
    check(ts3, "day end after exam MDD",
          daily_pnl=0, daily_resets=0)
    snap(ts3, "after exam MDD in eval")

    # ── Route 6: Exam pass day — daily_pnl recorded at day end ──
    print("\n--- Route 6: daily_pnl on exam pass day ---")
    ts4 = ExamSimulator(mode='eval')
    sim_day(ts4, [800, 800, 0])  # day 1: 1600
    sim_bar(ts4, pnl=800)
    sim_bar(ts4, pnl=800)  # acc = 3200, pass
    assert ts4.pass_exam == True
    assert ts4.daily_pnl == 1600  # persists
    sim_bar(ts4, pnl=0, day_end=True)
    # daily_pnls should have: [1600 (day1), 1600 (day2)]
    assert len(ts4.daily_pnls) == 2
    assert ts4.daily_pnls[1] == 1600, f"pass day pnl: {ts4.daily_pnls[1]}"
    snap(ts4, "daily_pnl on pass day")

    # ── Route 7: Subscription renewal ──
    print("\n--- Route 7: Subscription renewal ---")
    ts5 = ExamSimulator(mode='eval')
    pre_cost = ts5.topstep_cost
    for day in range(22):
        sim_day(ts5, [10, 10])  # small, won't pass
    assert ts5.subscription_days <= 20, f"sub_days: {ts5.subscription_days}"
    assert ts5.topstep_cost < pre_cost, "should have renewed (extra fee)"
    snap(ts5, "after subscription renewal")

    # ── Route 8: exam_passed_today skips activation phase ──
    print("\n--- Route 8: exam_passed_today skips activation ---")
    ts6 = ExamSimulator(mode='eval')
    sim_day(ts6, [800, 800, 0])
    sim_bar(ts6, pnl=800)
    sim_bar(ts6, pnl=800)  # pass
    assert ts6.exam_passed_today == True
    assert ts6.pass_exam == True
    # This bar should NOT enter _handle_activation_phase
    r = sim_bar(ts6, pnl=-5000, max_risk=-5000)  # big loss
    # Should NOT have reset (exam_passed_today blocks activation)
    assert r['reset_account'] == False, "should not reset during exam_passed_today"
    assert ts6.pass_exam == True, "should still be in activation"
    sim_bar(ts6, pnl=0, day_end=True)
    snap(ts6, "exam_passed_today protected")

    print("\n  [eval] ALL STATE TRACKING PASSED")


# ═══════════════════════════════════════════════════════════════════
# Cross-mode: daily_pnl list integrity
# ═══════════════════════════════════════════════════════════════════
def test_daily_pnl_integrity():
    print("\n" + "="*80)
    print("CROSS-MODE: daily_pnl list integrity")
    print("="*80)

    for mode in ['train_exam', 'train_activation', 'eval']:
        print(f"\n--- {mode} ---")
        kwargs = {'exam_success': 5, 'exam_counts': 10} if mode == 'train_activation' else {}
        ts = ExamSimulator(mode=mode, **kwargs)

        # Day 1: normal
        sim_day(ts, [100, 200, 300])
        assert ts.daily_pnls == [600], f"{mode} day1: {ts.daily_pnls}"

        # Day 2: MDD blow mid-day, then recover
        sim_bar(ts, pnl=500)
        sim_bar(ts, pnl=-3000, max_risk=-3000)  # MDD blow
        sim_bar(ts, pnl=200)
        sim_bar(ts, pnl=0, day_end=True)
        # daily_pnl = 500 + (-3000) + 200 = -2300
        assert ts.daily_pnls[-1] == -2300, f"{mode} day2: {ts.daily_pnls[-1]}"

        # Day 3: normal
        sim_day(ts, [400])
        assert ts.daily_pnls[-1] == 400, f"{mode} day3: {ts.daily_pnls[-1]}"

        # Verify total length
        assert len(ts.daily_pnls) == 3, f"{mode} total days: {len(ts.daily_pnls)}"
        print(f"  daily_pnls: {ts.daily_pnls}")

    print("\n  [daily_pnl integrity] ALL PASSED")


# ═══════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    test_train_exam_full_state()
    test_train_activation_full_state()
    test_eval_full_state()
    test_daily_pnl_integrity()
    print("\n" + "="*80)
    print("ALL COMPREHENSIVE STATE TRACKING TESTS PASSED")
    print("="*80)
