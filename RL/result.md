# Training Results

## Output File Index
| File | Run | Stop Loss | Re-entry | U/D Reset | Key Change |
|------|-----|-----------|----------|-----------|------------|
| output2.txt | Run 1 | K (active) | unlimited | every boundary | baseline |
| output3.txt | Run 2 | none (99999) | n/a | every boundary | no stops |
| output4.txt | Run 3 | K (active) | none | every boundary | no re-entry (partial, stopped early) |
| output5.txt | Run 4 | K (active) | once | every boundary | **BEST** |
| output6.txt | Run 5 | K (active) | once | carry on same dir | U/D carries over |

---

## Run 1 — output2.txt
**Stop loss**: K active, unlimited re-entry
**Result**: Val TH consistently negative. No payouts on val.
NOTE: used np.sum — divide by num_envs.

## Run 2 — output3.txt
**Stop loss**: Disabled (99999)
**Result**: Val TH positive multiple times (+1,615). Payouts on val (+2,274). Deeper LWM (-1,534).

## Run 3 — output4.txt (partial)
**Stop loss**: K active, no re-entry after stop
**Result**: Stopped early. Val TH -642 to -49. No payouts. More conservative LWM.

## Run 4 — output5.txt **BEST**
**Stop loss**: K active, one re-entry after stop, U/D resets every boundary
**Result**: Val TH best +4,077. Payouts +4,538. LWM -1,004.

## Run 5 — output6.txt
**Stop loss**: K active, one re-entry, U/D carries over on same consensus direction
**Time**: stopped at 226/500 (cyc:6.0)

### Val (last 5 checks)
| TakeHome | Payout | PassRate | LWM | VLoss best |
|----------|--------|----------|-----|------------|
| -855 to -98 | 0 | 0.00-0.39 | -1,252 | 102.01 |

**Result**: Val TH dropped — no positive TH in late val checks. No payouts in last 5 vals. U/D carry-over hurts — stale watermarks cause delayed entries. Reset every boundary (Run 4) is better.

---

## Comparison
| Run | Stop Loss | Re-entry | U/D | Val TH best | Val Payout best | Val LWM |
|-----|-----------|----------|-----|-------------|-----------------|---------|
| 1 | K | unlimited | reset | -164 | 0 | -436 |
| 2 | none | n/a | reset | +1,615 | +2,274 | -1,534 |
| 3 | K | none | reset | -49 | 0 | -1,006 |
| **4** | **K** | **once** | **reset** | **+4,077** | **+4,538** | **-1,004** |
| 5 | K | once | carry | +1,501 | +2,092 | -1,252 |

---

## Run 9 — output10.txt
**Changes from Run 4**:
- MAGNITUDE_FACTOR: 100→50 (tighter stops, K range halved)
- W_EXAM=3: separate exam count penalty (exam_counts/15 * 3)
- k_pts min 10 (floor in _step_1min)
- cost_ratio: no exam_count cap, just cost/payout
- _get_final_size: uses actual max_contracts, no hardcoded max_micro=5
**Status**: Running...
