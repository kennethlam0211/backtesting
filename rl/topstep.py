import math
import params
from utils.app_logger import get_logger
logger = get_logger('topstep', level=params.LOG_LEVEL)

#FIXME after payout, mddlimit
#FIXME tweak the reward fucntion when acc_pnl is way pass the target and success day lag behind


class ExamSimulator:
    ACCOUNTS = {
        '50k': {
            'EXAM_FEES': 49,
            'NO_ACTIVATION_FEES': 109,
            'REACTIVATION_FEES': 599,
            'EXAM_TARGET': 3000,
            'MDD_THRESHOLD': -2000,
            'INIT_MAX_CONTRACTS': 2,
            'MAX_CONTRACTS': 5,
            'CONTRACT_SCALING': [(1500, 2), (2000, 3), (float('inf'), 5)],
         },
        # '100k': {
        #     'EXAM_FEES': 99,
        #     'NO_ACTIVATION_FEES': 159,
        #     'REACTIVATION_FEES': 699,
        #     'EXAM_TARGET': 6000,
        #     'MDD_THRESHOLD': -3000,
        #     'INIT_MAX_CONTRACTS': 3,
        #     'MAX_CONTRACTS': 10,
        #     'CONTRACT_SCALING': [(1500, 3), (2000, 4), (3000, 5), (float('inf'), 10)],
        # },
        # '150k': {
        #     'EXAM_FEES': 149,
        #     'NO_ACTIVATION_FEES': 209,
        #     'REACTIVATION_FEES': 829,
        #     'EXAM_TARGET': 9000,
        #     'MDD_THRESHOLD': -4500,
        #     'INIT_MAX_CONTRACTS': 3,
        #     'MAX_CONTRACTS': 15,
        #     'CONTRACT_SCALING': [(1500, 3), (2000, 4), (3000, 5), (4000, 10), (float('inf'), 15)],
        # },
    }

    ACTIVATION_FEES = 149
    SUBSCRIPTION_RENEWAL_DAYS = 20
    ACTIVATION_SUCCESS_DAYS = 5
    DAILY_ACTIVATION_TARGET = 150
    CONSISTENCY_THRESHOLD = 0.5
    DISCOUNT_FACTOR = 0.9
    SCALING_MULTIPLER = 5
    #EXAM_SUCCESS_DAYS = 2
    MAX_DAILY_RESETS = 1   # first MDD blow locks the day
    BACKTOFUND_LIMIT = 0
    # Callback hook — PM registers a sim-state reset function here
    _on_reset = None

    def get_account_size(self, pnl_per_contract, std_per_contract):
        best, _ = self.recommend_account(pnl_per_contract, std_per_contract)
        self.set_account_attr(best)
        return best

    def set_account_attr(self, account_size):
        acct = self.ACCOUNTS[account_size]
        self.EXAM_FEES = acct['EXAM_FEES']
        self.NO_ACTIVATION_FEES = acct['NO_ACTIVATION_FEES']
        self.EXAM_TARGET = acct['EXAM_TARGET']
        self.MDD_THRESHOLD = acct['MDD_THRESHOLD']
        self.INIT_MAX_CONTRACTS = acct['INIT_MAX_CONTRACTS'] #for activation only
        self.MAX_CONTRACTS = acct['MAX_CONTRACTS']
        self._CONTRACT_SCALING = acct['CONTRACT_SCALING']
        self.REACTIVATION_FEES = acct['REACTIVATION_FEES']
        self.ACTIVATION_PNL_TARGET = 2 * abs(self.MDD_THRESHOLD)
        self.EXAM_PASS_RATE_THRESHOLD = (self.NO_ACTIVATION_FEES - self.EXAM_FEES) / self.ACTIVATION_FEES

    def __init__(self, account_size='50k', mode='eval',exam_success=0,exam_counts=0):
        self.account_size = account_size
        self.mode = mode  # 'train_exam' | 'train_activation' | 'eval'
        self.set_account_attr(account_size)
        self.exam_success = exam_success
        self.exam_counts = exam_counts
        self.daily_resets = 0
        self.max_contracts = self.MAX_CONTRACTS if mode != 'train_activation' else self.INIT_MAX_CONTRACTS
        self.subscription_days = 0
        self.exam_fees = 0
        self.activation_fees = 0
        self.reactivation_fees = 0
        self.back_to_fund_cnt = 0
        self.activation_cnt = 1 if mode == 'train_activation' else 0
        self.get_plan()
        self.activation_reset_cost = self.get_activation_reset_cost(self.mode)
        self.topstep_cost = -self.exam_fees if mode != 'train_activation' else - self.activation_reset_cost
        self.final_pnl_list: list[float] = [-self.exam_fees] if mode != 'train_activation' else [-self.activation_reset_cost]
        if mode != 'train_activation':
            self.exam_counts += 1
        self.daily_pnl = 0
        self.acc_pnl = 0
        self.total_commission = 0.0
        self.payouts = 0
        self.mdd_limit = self.MDD_THRESHOLD
        self.pass_exam = (mode == 'train_activation')
        self.success_activation_days = 0
        self.days_taken_to_pass_exams = []
        self.days_taken_to_fail_exams = []  # days from start of failed exam attempt to MDD reset
        self.days_taken_to_payouts = []  # days from activation start to payout
        self.days_taken_to_fail_activation = []  # days from activation start to a failure event
        self.activation_day_counter = 0  # running day counter since activation started
        self.daily_locked = False
        self.daily_target_hit = False
        self.exam_passed_today = False  # only for eval
        self.exam_daily_target_hit_days = 0
        self.exam_daily_target_hit_day_list = []  # per-day: True/False if daily target hit
        self.exam_daily_resets_list = []  # per-day: number of resets that day
        self.total_resets = 1  # lifetime MDD resets (never resets except env.reset)
        self.back_to_fund_list = []  # per-activation: back_to_fund_cnt when cycle ends
        self.payout_cnt = 0

        self._cached_sharpe = 0.0
        self._cached_sharpe_n = 0
        # PM reward potential-based tracking (resets with exam)


    @staticmethod
    def _ewm_mu_sigma(values):
        """Exponentially weighted mean and std from a list of values."""
        span = max(10, len(values) // 3)
        alpha = 2 / (span + 1)
        w = [(1 - alpha) ** (len(values) - 1 - i) for i in range(len(values))]
        w_sum = sum(w)
        mu = sum(wi * xi for wi, xi in zip(w, values)) / w_sum
        var = sum(wi * (xi - mu) ** 2 for wi, xi in zip(w, values)) / w_sum
        return mu, var ** 0.5

    @staticmethod
    def _gambler_ruin(mu, sigma, target, mdd_limit, contracts):
        """P(hit target before -mdd_limit) for a random walk with drift mu*C and vol sigma*C."""
        if mu <= 0:
            return 0.0
        if sigma == 0:
            return 1.0
        # exponent: 2 * mu * x / (sigma^2 * C)
        exp_b = 2 * mu * mdd_limit / (sigma ** 2 * contracts)
        exp_ab = 2 * mu * (target + mdd_limit) / (sigma ** 2 * contracts)
        # clamp to avoid overflow
        exp_b = min(exp_b, 500)
        exp_ab = min(exp_ab, 500)
        return (1 - math.exp(-exp_b)) / (1 - math.exp(-exp_ab))

    @staticmethod
    def recommend_account(pnl_per_contract, std_per_contract):
        """Given per-contract stats, return best account_size and breakdown.

        Args:
            pnl_per_contract: mean daily PnL per contract (mu)
            std_per_contract: std dev of daily PnL per contract (sigma)

        Returns:
            (best_account_size, details_dict)
        """
        results = {}
        for size, acct in ExamSimulator.ACCOUNTS.items():
            exam_fee = acct['EXAM_FEES']
            target = acct['EXAM_TARGET']
            mdd_limit = abs(acct['MDD_THRESHOLD'])
            max_c = acct['MAX_CONTRACTS']
            init_c = acct['INIT_MAX_CONTRACTS']
            activation_pnl_target = 2 * mdd_limit
            activation_fees = ExamSimulator.ACTIVATION_FEES
            discount = ExamSimulator.DISCOUNT_FACTOR

            mu = pnl_per_contract
            sigma = std_per_contract

            p_exam = ExamSimulator._gambler_ruin(mu, sigma, target, mdd_limit, max_c)
            p_activation = ExamSimulator._gambler_ruin(mu, sigma, activation_pnl_target, mdd_limit, init_c)
            payout = activation_pnl_target / 2 * discount

            # expected cost per successful payout (amortized over failures)
            if p_exam > 0 and p_activation > 0:
                cost_per_success = exam_fee / (p_exam * p_activation) + activation_fees / p_activation
            else:
                cost_per_success = float('inf')

            ev_per_attempt = p_exam * p_activation * payout - p_exam * activation_fees - exam_fee

            results[size] = {
                'p_pass': round(p_exam, 4),
                'p_activation': round(p_activation, 4),
                'payout': round(payout, 2),
                'cost_per_success': round(cost_per_success, 2),
                'ev_per_attempt': round(ev_per_attempt, 2),
            }

        best = max(results, key=lambda k: results[k]['ev_per_attempt'])
        return best, results

    def get_plan(self):
        if self.exam_counts == 0:
            self.exam_fees , self.activation_fees, self.reactivation_fees =  self.EXAM_FEES, self.ACTIVATION_FEES , self.REACTIVATION_FEES
            return
        exam_pass_rate = self.exam_success / self.exam_counts
        if exam_pass_rate < self.EXAM_PASS_RATE_THRESHOLD or self.mode =='train_exam':
            self.exam_fees , self.activation_fees, self.reactivation_fees =  self.EXAM_FEES, self.ACTIVATION_FEES , self.REACTIVATION_FEES
        else:
            self.exam_fees , self.activation_fees, self.reactivation_fees = self.NO_ACTIVATION_FEES, 0 , self.REACTIVATION_FEES

    def get_max_contract(self, acc_pnl, max_contracts) -> int:
        for threshold, contracts in self._CONTRACT_SCALING:
            if acc_pnl <= threshold * self.SCALING_MULTIPLER:
                return max(contracts, max_contracts)
        return max_contracts

    def _handle_new_day(self):

        if self.mode == 'train_exam':
            self.exam_daily_target_hit_day_list.append(self.daily_target_hit)
            if len(self.exam_daily_target_hit_day_list) > 30:
                self.exam_daily_target_hit_day_list.pop(0)
            self.exam_daily_resets_list.append(self.daily_resets)
            if len(self.exam_daily_resets_list) > 30:
                self.exam_daily_resets_list.pop(0)

        if self.exam_passed_today:
            self.daily_locked = False
            self.exam_passed_today = False
            self.daily_resets = 0
            
        if not self.pass_exam:
            self.subscription_days += 1
            if self.subscription_days > self.SUBSCRIPTION_RENEWAL_DAYS:
                self.subscription_days = 0
                self.topstep_cost -= self.exam_fees
                self.final_pnl_list.append(-self.exam_fees)

            if self.daily_target_hit:
                self.exam_daily_target_hit_days += 1
            self.daily_locked = False
            self.daily_resets = 0
            self.daily_target_hit = False

        elif self.mode != 'train_exam' and self.pass_exam:

            self.activation_day_counter += 1

            if self.daily_pnl > self.DAILY_ACTIVATION_TARGET:
                self.success_activation_days += 1

            logger.debug(f'act_day {self.activation_day_counter}: daily_pnl={self.daily_pnl:+.0f} '
                         f'success_days={self.success_activation_days}/{self.ACTIVATION_SUCCESS_DAYS} '
                         f'acc={self.acc_pnl:+.0f} target={self.ACTIVATION_PNL_TARGET:.0f}')

            if self.acc_pnl > self.ACTIVATION_PNL_TARGET and self.success_activation_days >= self.ACTIVATION_SUCCESS_DAYS:
                payout =  self.acc_pnl/2 * self.DISCOUNT_FACTOR
                self.acc_pnl = round(self.acc_pnl / 2)
                self.payouts += payout
                self.final_pnl_list.append(payout)
                self.payout_cnt += 1
                self.success_activation_days = 0
                self.days_taken_to_payouts.append(self.activation_day_counter)
                if len(self.days_taken_to_payouts) > 30:
                    self.days_taken_to_payouts.pop(0)
                self.mdd_limit = 0
                self.activation_day_counter = 0

            if self.acc_pnl > 0 and self.mdd_limit < 0:
                self.mdd_limit = max(min(self.acc_pnl - abs(self.MDD_THRESHOLD), 0), self.mdd_limit)

            self.max_contracts = self.get_max_contract(self.acc_pnl, self.max_contracts)

            self.daily_locked = False
        
        self.daily_pnl = 0

        # train_exam: auto-reset back to exam after pass day ends
        if self.mode == 'train_exam' and self.pass_exam:
            self._reset_exam()

    def maybe_resize_account(self, pnl_list):
        """Resize account based on accumulated pnl stats."""
        if len(pnl_list) > 30:
            mu, sigma = self._ewm_mu_sigma(pnl_list)
            if sigma > 0:
                self.account_size = self.get_account_size(mu, sigma)
                self.mdd_limit = self.MDD_THRESHOLD

    def get_activation_survive_prob(self, pnl_list):
        """P(hit payout target before MDD) using current trade stats."""
        if len(pnl_list) < 10:
            return 0.1
        mu, sigma = self._ewm_mu_sigma(pnl_list)
        return self._gambler_ruin(mu, sigma, self.ACTIVATION_PNL_TARGET, abs(self.mdd_limit), self.INIT_MAX_CONTRACTS)

    @staticmethod
    def cal_opportunity_cost_for_taking_exam(pass_rate, exam_target, mdd_threshold, payout_rate):
        """Opportunity cost of being stuck in exam instead of activation.

        Simplified: net exam PnL from 1 pass + failed attempts, discounted by payout rate.
        Days cancel out — only net PnL matters.

        Given:
            p           = pass_rate             e.g. 0.40
            target      = EXAM_TARGET           e.g. $3000
            MDD         = MDD_THRESHOLD         e.g. $2000
            payout_rate = DISCOUNT_FACTOR       e.g. 0.45

        Step 1 - Geometric: 1 pass + (1/p - 1) failures
            failed_attempts = 1/p - 1               = 1/0.40 - 1 = 1.5

        Step 2 - Net exam PnL: pass earns target, failures lose MDD
            net_pnl = target - failed_attempts × MDD
                    = $3000 - 1.5 × $2000            = $0

        Step 3 - Opportunity = net payout you'd take home
            opportunity = max(net_pnl, 0) × payout_rate
                        = max($0, 0) × 0.45          = $0

        Example (p=0.59):
            failed = 0.69
            net_pnl = $3000 - 0.69 × $2000          = $1,610
            opportunity = $1,610 × 0.45              = $724
        """
        failed_attempts = 1 / max(pass_rate, 0.01) - 1
        net_pnl = exam_target - failed_attempts * abs(mdd_threshold)
        return max(net_pnl, 0) * payout_rate

    def get_activation_reset_cost(self, mode):
        """Total cost of restarting from exam.

        Given (50k, p=0.40):
            exam_cost   = exam_fee / p                  = $49 / 0.40 = $123
            act_fee     = ACTIVATION_FEES               = $149
            opportunity = cal_opportunity_cost(...)      = $0
            start_over  = $123 + $149 + $0              = $272
            reset_cost  = min(start_over, reactivation) = min($272, $599) = $272

        Given (50k, p=0.59):
            exam_cost   = $49 / 0.59                    = $83
            opportunity = $1,610 × 0.45                 = $724
            start_over  = $83 + $149 + $724             = $956
            reset_cost  = min($956, $599)               = $599
        """
        pass_rate = self.exam_success / max(self.exam_counts, 1)
        exam_cost = self.exam_fees / max(pass_rate, 0.01)

        opportunity_cost = self.cal_opportunity_cost_for_taking_exam(
            pass_rate, self.EXAM_TARGET, self.MDD_THRESHOLD, self.DISCOUNT_FACTOR)
        start_over_cost = exam_cost + self.activation_fees + opportunity_cost

        if self.back_to_fund_cnt < self.BACKTOFUND_LIMIT:
            return min(start_over_cost, self.reactivation_fees)

        return start_over_cost
    
    def _reset_exam(self):
        self.total_resets += 1
        # Track days-to-fail. `subscription_days`/`activation_day_counter` reflect the count
        # since the current attempt started; +1 for the day it failed.
        if not self.pass_exam:
            # Exam phase failure
            self.days_taken_to_fail_exams.append(self.subscription_days + 1)
            if len(self.days_taken_to_fail_exams) > 30:
                self.days_taken_to_fail_exams.pop(0)
        else:
            # Activation phase failure (was in funded/activation, blew up)
            self.days_taken_to_fail_activation.append(self.activation_day_counter + 1)
            if len(self.days_taken_to_fail_activation) > 30:
                self.days_taken_to_fail_activation.pop(0)
        self.mdd_limit = self.MDD_THRESHOLD
        # Fire callback: lets PM reset shadow sim state (sim_entry_px, sim_daily_pnl)
        if self._on_reset is not None:
            self._on_reset()

        if self.mode == 'train_activation':
            self.max_contracts = self.MAX_CONTRACTS
            reset_cost = self.get_activation_reset_cost(self.mode)
            if reset_cost == self.reactivation_fees:
                self.back_to_fund_cnt += 1
            else:
                self.back_to_fund_list.append(self.back_to_fund_cnt)
                self.back_to_fund_cnt = 0
                reset_cost = self.exam_fees + self.activation_fees
            self.topstep_cost -= reset_cost
            self.final_pnl_list.append(-reset_cost)
            self.activation_cnt += 1
            self.success_activation_days = 0
            self.activation_day_counter = 0
            
        elif self.pass_exam and self.mode != 'train_exam': # eval activation: 
            reset_cost = self.get_activation_reset_cost(self.mode)
            self.topstep_cost -= reset_cost
            self.final_pnl_list.append(-reset_cost)
            self.activation_cnt += 1
            self.success_activation_days = 0
            if reset_cost == self.reactivation_fees:
                # Reactivation: stay in activation
                self.max_contracts = self.INIT_MAX_CONTRACTS
                self.back_to_fund_cnt += 1
            else:
                # Exhausted reactivations: back to exam
                self.back_to_fund_list.append(self.back_to_fund_cnt)
                self.back_to_fund_cnt = 0
                self.pass_exam = False
                self.max_contracts = self.MAX_CONTRACTS
                self.subscription_days = 0
                self.exam_counts += 1
                self.daily_resets = 0
                self.activation_day_counter = 0

        else: # eval exam/train_exam:
            self.pass_exam = False
            self.max_contracts = self.MAX_CONTRACTS
            self.subscription_days = 0
            self.topstep_cost -= self.exam_fees
            self.final_pnl_list.append(-self.exam_fees)
            self.exam_counts += 1
            self.get_plan()
            self.daily_target_hit = False

        self.acc_pnl = 0
        self.daily_pnl = 0  # reset so lock check stays aligned with acc_pnl
        self.total_commission = 0.0

    def _handle_exam_phase(self, max_risk):
        if self.acc_pnl + max_risk < self.mdd_limit:
            self._reset_exam()
            if self.daily_resets >= self.MAX_DAILY_RESETS:
                self.daily_locked = True
            else:
                self.daily_resets += 1
            return True

        elif not self.daily_locked:
            if self.daily_pnl >= self.EXAM_TARGET * self.CONSISTENCY_THRESHOLD:
                if self.daily_pnl>1510:
                    print(f'[DAILY TARGET PASS] acc_pnl={self.acc_pnl} daily_pnl={self.daily_pnl:.2f}')
                self.daily_locked = True
                self.daily_target_hit = True

        if self.acc_pnl >= self.EXAM_TARGET and self.subscription_days >= 1:
            if self.acc_pnl>3010:
                print(f'[EXAM PASS]acc_pnl={self.acc_pnl} daily_pnl={self.daily_pnl:.2f}')
            self.days_taken_to_pass_exams.append(self.subscription_days + 1)
            if len(self.days_taken_to_pass_exams) > 30:
                self.days_taken_to_pass_exams.pop(0)
            self.exam_success += 1
            self.mdd_limit = self.MDD_THRESHOLD
            self.success_activation_days = 0
            self.subscription_days = 0
            self.acc_pnl = 0
            self.daily_pnl = 0
            self.total_commission = 0.0
            self.pass_exam = True
            self.exam_passed_today = True
            if self.mode == 'eval':
                self.max_contracts = self.INIT_MAX_CONTRACTS
                self.topstep_cost -= self.activation_fees
                self.final_pnl_list.append(-self.activation_fees)
            else:
                self.daily_locked = True
            #     self.daily_resets = 0
            #     self.daily_target_hit = False
                
        return False

    def _handle_activation_phase(self, max_risk):
        if self.acc_pnl + max_risk < self.mdd_limit:
            self._reset_exam()
            self.daily_locked = True
            return True
        # # When acc_pnl is already past target, lock the day as soon as daily hits $150 —
        # # bank the success day and stop risking it.
        # if self.acc_pnl > self.ACTIVATION_PNL_TARGET and self.daily_pnl >= self.DAILY_ACTIVATION_TARGET:
        #     self.daily_locked = True

        # Tiered daily-lock: harder threshold early, essentially unlocked late.
        # Tier 1: < 4 success days → require $1000 daily (push for big wins, preserve them).
        # Tier 2: >= 4 success days → 3 × target (rarely triggers — let model earn as much as possible).
        daily_lock_threshold = 1500 if self.success_activation_days < 4 else 3 * self.ACTIVATION_PNL_TARGET
        if self.daily_pnl >= daily_lock_threshold:
            self.daily_locked = True
            logger.debug(f'[DAILY_LOCK tier1/2] acc={self.acc_pnl:+.0f} daily={self.daily_pnl:+.0f} '
                         f'success_days={self.success_activation_days}/{self.ACTIVATION_SUCCESS_DAYS} '
                         f'threshold={daily_lock_threshold}')
        # Tier 3: once acc is past target AND still building success days (<4),
        # lock at 1.5 × daily target ($225) — protect the gain with any small profit.
        # Skipped when success_days >= 4 so Tier 2 takes over for the final-stretch grind.
        elif (self.success_activation_days < 4
              and self.acc_pnl >= self.ACTIVATION_PNL_TARGET
              and self.daily_pnl > 1.5 * self.DAILY_ACTIVATION_TARGET):
            self.daily_locked = True
            logger.debug(f'[DAILY_LOCK tier3] acc={self.acc_pnl:+.0f} daily={self.daily_pnl:+.0f} '
                         f'success_days={self.success_activation_days}/{self.ACTIVATION_SUCCESS_DAYS} '
                         f'threshold={1.5 * self.DAILY_ACTIVATION_TARGET}')

        return False

    def update(self, pnl, max_risk, day_end, trading_fees, rth=1):
        self.acc_pnl += ( pnl + trading_fees)
        self.daily_pnl += ( pnl + trading_fees)

        # RTH: unlock if locked from ETH MDD blow (not from daily target hit) #for train
        if rth and self.daily_locked and not self.daily_target_hit and not self.pass_exam and self.daily_resets < self.MAX_DAILY_RESETS:
            self.daily_locked = False

        if not self.pass_exam:
            reset = self._handle_exam_phase(max_risk)
        elif self.exam_passed_today:
            reset = False
        else:
            reset = self._handle_activation_phase(max_risk)

        if day_end:
            self._handle_new_day()

        return {
                'account_size': self.account_size,
                'max_contracts' : self.max_contracts,
                'acc_pnl': self.acc_pnl,
                'topstep_cost': self.topstep_cost,
                'payouts': self.payouts,
                'success_activation_days': self.success_activation_days/self.ACTIVATION_SUCCESS_DAYS,
                'pass_exam': self.pass_exam,
                'subscription_days': self.subscription_days / self.SUBSCRIPTION_RENEWAL_DAYS,
                'mdd_limit': self.mdd_limit,
                'reset_account': reset,
                'lock_trading': self.daily_locked,
            }




























