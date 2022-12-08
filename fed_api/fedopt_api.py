import copy
import logging
import random
import cvxpy as cp

import numpy as np
import torch
import wandb

from fed_api.fed_api import FedAPI
from fed_api.utils.utils_func import *


class FedOptAPI(FedAPI):

    def _get_winners(self):
        '''
        :return: winners: List(int), min_utility: int
        '''
        mx_utility = 0
        winners = []
        opt_time = 0
        for client in self.client_list:
            t_max = client.get_time()
            self.t_max = t_max
            # logging.info("------set t_max:{}---------".format(self.t_max))

            # DFS
            # temp_winners = self._winners_determination_dfs()
            # LP
            temp_winners = self._winners_determination()
            temp_winners_utility = self._get_utility(temp_winners)

            if temp_winners_utility > mx_utility:
                winners = copy.deepcopy(temp_winners)
                mx_utility = temp_winners_utility
                opt_time = self.t_max
        self.t_max = opt_time
        payment = [self.client_list[idx].get_bidding_price() for idx in winners]
        return winners, payment

    def _dfs(self, candidate_selected, bid_idx, sum_ti, sum_p):
        if bid_idx >= len(self.candidates):
            return
        bid = self.candidates[bid_idx]
        self._dfs(candidate_selected, bid_idx + 1, sum_ti, sum_p)
        if sum_p + bid.get_bidding_price() < self.args.budget_per_round:
            candidate_selected[bid_idx] = 1
            if sum_ti + bid.get_training_intensity() > self.mx_training_intensity:
                self.mx_training_intensity = sum_ti + bid.get_training_intensity()
                self.candidate_selected = copy.deepcopy(candidate_selected)
            self._dfs(candidate_selected, bid_idx + 1, sum_ti + bid.get_training_intensity(),
                      sum_p + bid.get_bidding_price())
            candidate_selected[bid_idx] = 0

    def _winners_determination_dfs(self, m_client_list=None):
        """
        :param T_max: int
        :param m_client_list: List(Client)
        :return winners_index: List(int)
        """
        if m_client_list is None:
            m_client_list = self.client_list

        # winners index is the index in the list of self.client_list
        winners_indexes = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)
        self.candidates = candidates

        # DFS
        candidates_selected = np.zeros(len(self.candidates))
        self.candidate_selected = np.zeros(len(self.candidates))
        self.mx_training_intensity = 0
        self.t_max = 0

        self._dfs(candidates_selected, 0, 0, 0)
        for bid_idx, bid_val in enumerate(self.candidate_selected):
            if bid_val == 1:
                winners_indexes.append(self.candidates[bid_idx].client_idx)
        logging.info("DFS: winners:{}".format(winners_indexes))

        return winners_indexes

    def _winners_determination(self, m_client_list=None):
        """
        :param T_max: int
        :param m_client_list: List(Client)
        :return winners_index: List(int)
        """
        if m_client_list is None:
            m_client_list = self.client_list

        # winners index is the index in the list of self.client_list
        winners_indexes = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)

        self.t_max = 0

        x = cp.Variable(len(candidates), boolean=True)
        # training intensity list
        ti_a = [bid.get_training_intensity() for bid in candidates]
        # payment list, bidding price list
        payment_a = [bid.get_bidding_price() for bid in candidates]
        prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(ti_a, x))),
                          [cp.sum(cp.multiply(x, payment_a)) <= self.args.budget_per_round])

        prob.solve(solver=cp.CPLEX)
        # print("model goal:{}".format(prob.value))
        # print(x.value)
        for idx, val in enumerate(x.value):
            if val == True:
                winners_indexes.append(candidates[idx].client_idx)

        # logging.info("lp selected winners:{}".format(winners_indexes))
        return winners_indexes