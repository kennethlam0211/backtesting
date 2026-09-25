import math

#FIXME reset only for during exam
#gonna train one for passing exam
#one for taking money
#FIXME Daily loss limit
#FIXME 打折 on weighting
#FIXME for val dont skip trades when its fullied 
#FIXME new_account is 10
#FIXME closing to the day end release more, for exam reset
#FIXME if it reaches the exam_target for any given day, +=1, if two days pass, denote the exam pass rate>>>
#FIXME on average how many days could pass the exam



class ExamSimulator:
    ACCOUNTS = {
        '50k': {
            'EXAM_FEES_': 49,
            'NO_ACTIVATION_FEES_': 109,
            'EXAM_TARGET': 3000,
            'MDD_THRESHOLD': -2000,
            'INIT_MAX_CONTRACTS': 2,
            'MAX_CONTRACTS': 5,
            'CONTRACT_SCALING': [(1500, 2), (2000, 3), (float('inf'), 5)],
        },
        # '100k': {
        #     'EXAM_FEES_': 99,
        #     'NO_ACTIVATION_FEES_': 159,
        #     'EXAM_TARGET': 6000,
        #     'MDD_THRESHOLD': -3000,
        #     'INIT_MAX_CONTRACTS': 3,
        #     'MAX_CONTRACTS': 10,
        #     'CONTRACT_SCALING': [(1500, 3), (2000, 4), (3000, 5), (float('inf'), 10)],
        # },
        # '150k': {
        #     'EXAM_FEES_': 149,
        #     'NO_ACTIVATION_FEES_': 209,
        #     'EXAM_TARGET': 9000,
        #     'MDD_THRESHOLD': -4500,
        #     'INIT_MAX_CONTRACTS': 3,
        #     'MAX_CONTRACTS': 15,
        #     'CONTRACT_SCALING': [(1500, 3), (2000, 4), (3000, 5), (4000, 10), (float('inf'), 15)],
        # },
    }

    ACTIVATION_FEES_ = 149
    SUBSCRIPTION_RENEWAL_DAYS = 20
    ACTIVATION_SUCCESS_DAYS = 5
    DAILY_ACTIVATION_TARGET = 150
    CONSISTENCY_THRESHOLD = 0.5
    DISCOUNT_FACTOR = 0.9
    SCALING_MULTIPLER = 5

    def get_account_size(self, pnl_per_contract, std_per_contract):
        best, _ = self.recommend_account(pnl_per_contract, std_per_contract)
        self.set_account_attr(best)
        return best

    def set_account_attr(self, account_size):
        acct = self.ACCOUNTS[account_size]
        self.EXAM_FEES_ = acct['EXAM_FEES_']
        self.NO_ACTIVATION_FEES_ = acct['NO_ACTIVATION_FEES_']
        self.EXAM_TARGET = acct['EXAM_TARGET']
        self.MDD_THRESHOLD = acct['MDD_THRESHOLD']
        self.INIT_MAX_CONTRACTS = acct['INIT_MAX_CONTRACTS']
        self.MAX_CONTRACTS = acct['MAX_CONTRACTS']
        self._contract_scaling = acct['CONTRACT_SCALING']
        self.ACTIVATION_PNL_TARGET = 2 * abs(self.MDD_THRESHOLD)
        self.EXAM_PASS_RATE_THRESHOLD = (self.NO_ACTIVATION_FEES_ - self.EXAM_FEES_) / self.ACTIVATION_FEES_

    MAX_EXAM_COUNTS = 30   # 10 accounts × ~3 attempts each
    MAX_DAILY_RESETS = 2   # TopStep limit: 2 resets per day

    def __init__(self, account_size='50k'):
        self.account_size = account_size
        self.set_account_attr(account_size)
        self.exam_success = 0
        self.exam_counts = 0
        self.daily_resets = 0
        self.max_contracts = self.MAX_CONTRACTS
        self.subscription_days = 1
        self.exam_fees, self.activation_fees = self.get_plan(self.exam_success, self.exam_counts)
        self.topstep_cost = -self.exam_fees
        self.final_pnl_list = [-self.exam_fees]
        self.exam_counts += 1
        self.daily_pnl = 0
        self.acc_pnl = 0
        self.total_commission = 0.0
        self.payouts = 0
        self.take_home_lwm = -self.exam_fees  # lowest water mark of take_home profit
        self.mdd_limit = self.MDD_THRESHOLD
        self.pass_exam = False
        self.success_activation_days = 0
        self.all_pnl_per_contract_list = []
        self._cached_sharpe = 0.0
        self._cached_sharpe_n = 0
        # PM reward potential-based tracking (resets with exam)
        self._prev_goal_progress = 0.0
        self._prev_mdd_buffer = 0.0

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
            exam_fee = acct['EXAM_FEES_']
            target = acct['EXAM_TARGET']
            mdd_limit = abs(acct['MDD_THRESHOLD'])
            max_c = acct['MAX_CONTRACTS']
            init_c = acct['INIT_MAX_CONTRACTS']
            activation_pnl_target = 2 * mdd_limit
            activation_fees = ExamSimulator.ACTIVATION_FEES_
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

    def get_plan(self, exam_success, exam_counts):
        if exam_counts == 0:
            return self.EXAM_FEES_, self.ACTIVATION_FEES_
        exam_pass_rate = exam_success / exam_counts
        if exam_pass_rate < self.EXAM_PASS_RATE_THRESHOLD:
            return self.EXAM_FEES_, self.ACTIVATION_FEES_
        else:
            return self.NO_ACTIVATION_FEES_, 0

    def get_max_contract(self, acc_pnl, max_contracts):
        for threshold, contracts in self._contract_scaling:
            if acc_pnl <= threshold * self.SCALING_MULTIPLER:
                return max(contracts, max_contracts)

    def _handle_new_day(self):
        if not self.pass_exam:
            self.subscription_days += 1
            if self.subscription_days > self.SUBSCRIPTION_RENEWAL_DAYS:
                self.subscription_days = 1
                self.topstep_cost -= self.exam_fees
                self.final_pnl_list.append(-self.exam_fees)
        else:
            if self.daily_pnl > self.DAILY_ACTIVATION_TARGET:
                self.success_activation_days += 1
            
            ACTIVATION_PNL_TARGET = self.ACTIVATION_PNL_TARGET
            if self.payouts-self.topstep_cost >= self.ACTIVATION_PNL_TARGET:
                ACTIVATION_PNL_TARGET *= 2
            
            if self.acc_pnl > self.ACTIVATION_PNL_TARGET and self.success_activation_days >= self.ACTIVATION_SUCCESS_DAYS:
                self.acc_pnl /= 2
                payout =  self.acc_pnl * self.DISCOUNT_FACTOR
                self.payouts += payout
                self.final_pnl_list.append(payout)
                self.success_activation_days = 0
            if self.acc_pnl > 0 and self.mdd_limit < 0:
                self.mdd_limit = max(min(self.acc_pnl - abs(self.MDD_THRESHOLD), 0), self.mdd_limit)

            self.max_contracts = self.get_max_contract(self.acc_pnl, self.max_contracts)

        self.daily_pnl = 0
        self.daily_resets = 0  # reset daily counter

    def _reset_exam(self):
        self.subscription_days = 1
        self.exam_fees, self.activation_fees = self.get_plan(self.exam_success, self.exam_counts)
        self.topstep_cost -= self.exam_fees
        self.final_pnl_list.append(-self.exam_fees)
        self.take_home_lwm = min(self.take_home_lwm, self.payouts + self.topstep_cost)
        self.exam_counts += 1
        self.acc_pnl = 0
        self.total_commission = 0.0
        self.daily_pnl = 0

        if len(self.all_pnl_per_contract_list) > 30:
            mu, sigma = self._ewm_mu_sigma(self.all_pnl_per_contract_list)
            if sigma > 0:
                self.account_size = self.get_account_size(mu, sigma)
        self.mdd_limit = self.MDD_THRESHOLD
        self.max_contracts = self.MAX_CONTRACTS
        # Reset reward tracking potentials
        self._prev_goal_progress = 0.0
        self._prev_mdd_buffer = 0.0

    def _handle_exam_phase(self, max_risk):
        if self.acc_pnl + max_risk < self.mdd_limit:
            self.daily_resets += 1
            self._reset_exam()
            return True

        elif self.acc_pnl > self.EXAM_TARGET:
            self.pass_exam = True
            self.topstep_cost -= self.activation_fees
            self.final_pnl_list.append(-self.activation_fees)
            self.take_home_lwm = min(self.take_home_lwm, self.payouts + self.topstep_cost)
            self.exam_success += 1
            self.mdd_limit = self.MDD_THRESHOLD
            self.success_activation_days = 0
            self.subscription_days = 0
            self.acc_pnl = 0
            self.total_commission = 0.0
            self.max_contracts = self.INIT_MAX_CONTRACTS
            # Reset reward potentials for activation phase
            self._prev_goal_progress = 0.0
            self._prev_mdd_buffer = 1.0
        return False

    def _handle_activation_phase(self, max_risk):
        if self.acc_pnl + max_risk < self.mdd_limit:
            #FIXME learn when to reset
            # if self.daily_resets >= self.MAX_DAILY_RESETS or self.exam_counts >= self.MAX_EXAM_COUNTS:
            #     return True  # can't reset — sit idle
            self.daily_resets += 1
            self.pass_exam = False
            self._reset_exam()
            return True
        return False

    def update(self, pnl, max_risk, day_end, trading_fees, traded_contracts):
        if traded_contracts > 0:
            per_contract_pnl = (pnl+trading_fees)/traded_contracts
            self.all_pnl_per_contract_list.append(per_contract_pnl)
        else:
            per_contract_pnl = 0.0

        self.acc_pnl += ( pnl + trading_fees)
        self.daily_pnl += ( pnl + trading_fees)

        if not self.pass_exam:
            reset = self._handle_exam_phase(max_risk)
        else:
            reset = self._handle_activation_phase(max_risk)
        
        if day_end:
            self._handle_new_day()

        return {
            'account_size': self.account_size,
            'max_contracts' : self.max_contracts,
            'per_contract_pnl': per_contract_pnl,
            'acc_pnl': self.acc_pnl,
            'topstep_cost': self.topstep_cost,
            'payouts': self.payouts,
            'success_activation_days': self.success_activation_days/self.ACTIVATION_SUCCESS_DAYS,
            'pass_exam': self.pass_exam,
            'subscription_days': self.subscription_days / 30,
            'mdd_limit': self.mdd_limit,
            'reset_account': reset,
        }
