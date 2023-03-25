import logging
import operator

import numpy as np
import wandb
import torch

from fed_api.fed_api import FedAPI
from fed_api.utils.client import Client
from fed_api.utils.utils_func import *


class Fed3API(FedAPI):

    def _get_winners(self):
        '''
        :return: winners: List(int), min_utility: int
        '''
        payment = []
        mx_utility = 0
        winners = []
        opt_time = 0
        critical_client = Client()
        for client in self.client_list:
            t_max = client.get_time()
            self.t_max = t_max

            temp_winners_utility = 0
            prob = np.random.random()
            # logging.info("time:{}, prob:{}".format(self.t_max, prob))
            temp_winners_utility_1 = 0
            temp_winners_utility_2 = 0
            if prob <= 1.0 / 3.0:
                winner = 0
                mx_v = 0
                for client_i in self.client_list:
                    if client_i.get_time() > self.t_max:
                        continue
                    if client_i.get_training_intensity() > mx_v:
                        mx_v = client_i.get_training_intensity()
                        winner = client_i.client_idx
                        temp_winners_utility_1 = 1.0 * client_i.get_training_intensity()
                temp_winners = [winner]
                temp_payment = [self.args.budget_per_round]
            else:
                temp_winners, critical_client = self._winners_determination()
                temp_winners_list = get_client_list(temp_winners, self.client_list)
                temp_winners_utility_2 = get_total_training_intensity(temp_winners_list)
                temp_payment = self._get_payment(temp_winners, critical_client)

            # temp_winners_utility = (1.0/3.0 * temp_winners_utility_1 + 2.0/3.0*temp_winners_utility_2)/t_max
            temp_winners_utility = (1.0/3.0 * temp_winners_utility_1 + 2.0/3.0*temp_winners_utility_2)/(t_max**self.args.alpha)

            if temp_winners_utility > mx_utility:
                winners = temp_winners
                mx_utility = temp_winners_utility
                opt_time = self.t_max
                payment = temp_payment
        self.t_max = opt_time

        return winners, payment

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
        # winners list is the list of selected clients
        winners_list = []
        # candidates' bid
        candidates = []
        for client in m_client_list:
            if client.get_time() <= self.t_max:
                candidates.append(client.bid)

        t_max = 0

        for bid in candidates:
            bid.update_avg_cost()

        # sort candidates according to average cost
        # argmax \tau / c
        cmp = operator.attrgetter('avg_cost')
        candidates.sort(key=cmp)

        while len(candidates):
            winner_idx = candidates[-1].client_idx
            winner_client = self.client_list[winner_idx]
            candidates.pop()
            # f(W \cup s_i)
            winners_client_ti = get_total_training_intensity(winners_list + [winner_client])
            budget_limit = self.args.budget_per_round * winner_client.get_training_intensity() / winners_client_ti
            if winner_client.get_bidding_price() > budget_limit:
                break
            winners_indexes.append(winner_idx)
            winners_list.append(winner_client)
            t_max = max(t_max, winner_client.get_time())

        critical_client = None
        if len(candidates):
            critical_idx = candidates[-1].client_idx
            critical_client = self.client_list[critical_idx]
        return winners_indexes, critical_client

    def _get_payment(self, winners_index, critical_client = None):
        '''
        :param winners_index: List(int)
        :param critical_client: Client, k+1 th client.
        :return:
        '''
        payment = np.zeros(len(winners_index))
        tot_training_intensity = 0
        # compute total trianing intensity
        for client_index in winners_index:
            client = self.client_list[client_index]
            tot_training_intensity += client.get_training_intensity()

        for i, client_i_index in enumerate(winners_index):
            # logging.info("getting payment for {}".format(client_i_index))
            client_i = self.client_list[client_i_index]
            payment_1 = self.args.budget_per_round * client_i.get_training_intensity() / tot_training_intensity
            if critical_client is None:
                payment[i] = payment_1
            else:
                payment_2 = client_i.get_training_intensity() * critical_client.get_bidding_price() / critical_client.get_training_intensity()
                payment[i] = min(payment_1, payment_2)
        # logging.info("payment list" + str(payment))
        return payment
