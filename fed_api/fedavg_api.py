import logging
import random

import numpy as np
import wandb
import torch

from fed_api.fed_api import FedAPI
from fed_api.utils.utils_func import *

class FedAvgAPI(FedAPI):

    def _get_winners(self):
        sum_p = 0
        client_indexes = []
        candidates = []
        for client in self.client_list:
            candidates.append(client.client_idx)

        random.shuffle(candidates)
        for client_i in candidates:
            client = self.client_list[client_i]
            if sum_p + client.get_bidding_price() <= self.args.budget_per_round:
                sum_p += client.get_bidding_price()
                client_indexes.append(client_i)

        logging.info("client_indexes = %s" % str(client_indexes))
        payment = [self.client_list[idx].get_bidding_price() for idx in client_indexes]
        return client_indexes, payment