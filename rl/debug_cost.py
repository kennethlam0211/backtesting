"""Debug: trace cost accumulation in train_exam mode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topstep import ExamSimulator

ts = ExamSimulator(mode='train_exam')
print(f'INIT: cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print(f'exam_fees={ts.exam_fees}, activation_fees={ts.activation_fees}')
print()

# Day 1: win enough to pass exam
for i in range(5):
    ts.update(700, -1400, False, 0, 1)
print(f'After 5 wins: acc={ts.acc_pnl}, daily={ts.daily_pnl}, pass={ts.pass_exam}')
print(f'cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print()

# Day end
ts.update(0, 0, True, 0, 0)
print(f'Day 1 end: acc={ts.acc_pnl}, pass={ts.pass_exam}, exam_passed={ts.exam_passed_today}')
print(f'cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print()

# Day 2: in activation, no position, just day end
ts.update(0, 0, True, 0, 0)
print(f'Day 2 end (activation idle): acc={ts.acc_pnl}, pass={ts.pass_exam}')
print(f'cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print()

# Day 3: blow activation
ts.update(-3000, -3000, False, 0, 1)
print(f'Activation blow: acc={ts.acc_pnl}, pass={ts.pass_exam}')
print(f'cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print()

# Day end
ts.update(0, 0, True, 0, 0)
print(f'Day 3 end: acc={ts.acc_pnl}, pass={ts.pass_exam}')
print(f'cost={ts.topstep_cost}, pnl_list={ts.final_pnl_list}')
print(f'exam_counts={ts.exam_counts}, exam_success={ts.exam_success}, total_resets={ts.total_resets}')
print()

# Do 5 more exam cycles: pass, idle, blow, repeat
for cycle in range(5):
    print(f'--- Cycle {cycle+2} ---')
    # Win to pass
    for i in range(5):
        ts.update(700, -1400, False, 0, 1)
    ts.update(0, 0, True, 0, 0)  # day end
    # Activation idle day
    ts.update(0, 0, True, 0, 0)
    # Blow
    ts.update(-3000, -3000, False, 0, 1)
    ts.update(0, 0, True, 0, 0)  # day end
    print(f'cost={ts.topstep_cost}, total_resets={ts.total_resets}, exam_s={ts.exam_success}/{ts.exam_counts}')
    print(f'pnl_list={ts.final_pnl_list}')
    print()

print(f'FINAL: cost={ts.topstep_cost}')
print(f'total items in pnl_list: {len(ts.final_pnl_list)}')
print(f'sum of pnl_list: {sum(ts.final_pnl_list)}')
