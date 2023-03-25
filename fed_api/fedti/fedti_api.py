import logging
import random

import numpy as np
import torch
from operator import attrgetter

from fed_api.fed_api import FedAPI
from fed_api.utils.utils_func import *

class FedTiAPI(FedAPI):
    def _winner_determination(self, exc_client_index=None):
        '''
            exc_client: client index to exclude
        '''
        sum_ti = 0
        t_max = 0
        client_indexes = []
        candidates = []
        for client in self.client_list:
            if exc_client_index and client.client_idx == exc_client_index:
                continue
            candidates.append(client.bid)

        max_c = 0
        if exc_client_index:
            exc_client = self.client_list[exc_client_index]
        while sum_ti < self.args.training_intensity:
            for bid in candidates:
                ## TODO: update average cost
                bid.update_avg_cost(t_max)
            sorted(candidates, key=attrgetter('avg_cost'))
            winner_bid = candidates[0]
            candidates.pop(0)
            client_indexes.append(winner_bid.client_idx)
            if exc_client_index:
                temp_c = exc_client.get_training_intensity()*winner_bid.get_average_cost()-max(t_max, exc_client.get_time())
                max_c = max(max_c, temp_c)
            t_max = max(t_max, bid.get_time())

        return client_indexes, max_c

    

    def _get_winners(self):
        client_indexes, _ = self._winner_determination()
        payment = []
        for idx in client_indexes:
            _, payment_i = self._winner_determination(idx)
            payment.append(payment_i)

        logging.info("client_indexes = %s" % str(client_indexes))
        logging.info("client payment : %s" % str(payment))
        
        return client_indexes, payment