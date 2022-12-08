import copy
import logging
import operator
import random

import numpy as np
import wandb
import torch

from fed_api.utils.utils_func import *
from fed_api.fed_api import FedAPI


class FedBFAPI(FedAPI):

    def _get_winners(self):
        '''
        :return: winners: List(int), payment: List(float)
        '''
        payment = []
        winners = []

        candidates = []
        for client in self.client_list:
            candidates.append(client.bid)

        cmp = operator.attrgetter('bidding_price')
        candidates.sort(key=cmp)
        sum_payment = 0

        t_max = 0
        for bid in candidates:
            if sum_payment + bid.get_bidding_price() > self.args.budget_per_round:
                break
            sum_payment += bid.get_bidding_price()
            winners.append(bid.client_idx)
            payment.append(bid.get_bidding_price())
            t_max = max(t_max, bid.get_time())
        return winners, payment