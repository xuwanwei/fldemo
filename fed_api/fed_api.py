import copy
import logging

import numpy as np
import wandb
import torch

from fed_api.utils.client import Client
from fed_api.utils.testInfo import TestInfo
from fed_api.utils.utils_func import *
from utils.args_parser import parse_args
from utils.data_splitter import split_data
from utils.optimizer import Optimizer
from model.cnn import CNNCifar


class FedAPI(object):
    def __init__(self, device, args, global_net=None, train_data=None, test_data=None):
        self.device = device
        self.args = args

        self.client_list = []
        self.t_max = 0

        self.train_data = train_data
        self.test_data = test_data

        self.global_net = global_net
        # TODO: dataset args
        self.user_data, self.user_data_size = split_data(
            self.train_data, args.client_num_in_total, args.dataset, args.iid, args.uniform)
        logging.info("running:{}".format(self.args.fed_name))

        self._setup_clients()

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        logging.info("client_num_in_total:" +
                     str(self.args.client_num_in_total))
        for client_idx in range(self.args.client_num_in_total):
            optimizer = Optimizer(
                args=self.args, data=self.train_data, indexs=self.user_data[client_idx])
            c = Client(client_idx=client_idx, device=self.device, args=self.args, communication_time=0,
                       computation_coefficient=0, cost=1, training_intensity=0,
                       local_training_data=self.user_data[client_idx],
                       local_sample_number=self.user_data_size[client_idx], optimizer=optimizer)
            self.client_list.append(c)
        logging.info("number of clients in client_list:" +
                     str(len(self.client_list)))
        logging.info("############setup_clients (END)#############")

    def _get_winners(self):
        pass

    def test_properties(self):
        obj_list = []
        t_max_list = []
        ti_list = []

        for round_idx in range(self.args.comm_round):
            np.random.seed((self.args.seed*round_idx) % 10000000)

            # bids init
            for client in self.client_list:
                init_client_bid(client)

            client_indexes, payment = self._get_winners()
            # logging.info('winners:{}'.format(client_indexes))
            # logging.info('payment:{}'.format(payment))

            tot_training_intensity = 0
            tot_payment = 0
            t_max = 0

            for idx, client_idx in enumerate(client_indexes):
                client = self.client_list[int(client_idx)]
                # distribute payment
                client.receive_payment(payment[idx])
                tot_payment += payment[idx]
                t_max = max(t_max, client.get_time())
                tot_training_intensity += client.get_training_intensity()

            t_max_list.append(t_max)
            ti_list.append(tot_training_intensity)
            obj_list.append(tot_training_intensity/t_max)

            t_idx = np.random.randint(0, len(client_indexes))
            client = self.client_list[client_indexes[t_idx]]
            real_cost = client.get_cost()
            client_payment = client.get_payment()

        training_intensity_mean = np.mean(ti_list)
        t_mean = np.mean(t_max_list)
        obj_mean = np.mean(obj_list)

        return TestInfo(tot_payment=tot_payment, true_cost=real_cost,
                        payment=client_payment, time=t_mean,
                        training_intensity=training_intensity_mean,
                        objective=obj_mean)

    def train(self):
        train_loss = []
        test_acc = []
        test_loss = []
        time_list = []
        ti_sum_list = []
        round_list = []

        for round_idx in range(self.args.comm_round):
            np.random.seed((self.args.seed * (round_idx+1)) % 10000000)
            logging.info(
                "#############Communication round : {}".format(round_idx))
            t_max, ti_sum, loss_sum = 0, 0, 0
            w_locals = []

            # bids init
            for client in self.client_list:
                init_client_bid(client)

            client_indexes, payment = self._get_winners()

            logging.info("select {} clients".format(len(client_indexes)))

            # train on winners
            for idx, client_idx in enumerate(client_indexes):
                client = self.client_list[int(client_idx)]
                if client.get_training_intensity() <= 0:
                    continue

                # distribute payment
                client.receive_payment(payment[idx])
                t_max = max(t_max, client.get_time())
                ti_sum += client.get_training_intensity()

                # train on new dataset
                w, loss = client.train(self.global_net)
                w_locals.append(copy.deepcopy(w))
                loss_sum += loss

                # debug
                print("client: {}, tau: {}".format(
                    client_idx, client.get_training_intensity()))

            # debug
            # print("")

            # update global weights
            global_w = w_locals[0]
            for key in global_w.keys():
                for j in range(1, len(w_locals)):
                    global_w[key] += w_locals[j][key]
                global_w[key] = torch.div(global_w[key], len(w_locals))

            self.global_net.train()
            self.global_net.load_state_dict(global_w)

            # time, sum training intensity results
            train_loss_avg = loss_sum/len(client_indexes)
            time_list.append(t_max)
            ti_sum_list.append(ti_sum)
            train_loss.append(train_loss_avg)
            round_list.append(round_idx)

            # test
            tester = Optimizer(args=self.args, data=self.test_data,
                               indexs=np.arange(len(self.test_data)))
            acc, loss = tester.test(self.global_net.to(self.device))
            test_acc.append(acc)
            test_loss.append(loss)
            logging.info("train_loss:{:.6f}, time:{:.6f}, total_time:{:.6f} ti_sum_list:{:.6f} test_acc:{:.6f}, test_loss:{:.6f}".format(
                train_loss_avg, t_max, sum(time_list), ti_sum, acc, loss))

        return test_acc, test_loss, time_list, ti_sum_list, round_list, train_loss

    # used to test truthfulness
    def train_for_truthfulness(self, truth_ratio):
        np.random.seed(self.args.seed)

        # bids init
        for client in self.client_list:
            init_client_bid(client)

        # choose one bid in one particular round to test truthfulness
        truth_index = np.random.randint(0, len(self.client_list))
        self.client_list[truth_index].update_bidding_price_with_ratio(
            truth_ratio)
        logging.info(
            "truth_index" + str(truth_index) + ", true cost: " + str(
                self.client_list[truth_index].get_cost()) + ", bidding price: " + str(
                self.client_list[truth_index].get_bidding_price()) + ", time: " + str(
                self.client_list[truth_index].get_time()))

        client_indexes, payment = self._get_winners()
        logging.info('winners:{}'.format(client_indexes))
        logging.info('payment:{}'.format(payment))

        # train on winners
        for idx, client_idx in enumerate(client_indexes):
            if client_idx == truth_index:
                client = self.client_list[int(client_idx)]
                # distribute payment
                client.receive_payment(payment[idx])

        client_truth = self.client_list[truth_index]
        logging.info('id:{}, cost:{} bidding_price:{}, payment:{}, utility:{}'.format(truth_index,
                                                                                      client_truth.get_cost(),
                                                                                      client_truth.get_bidding_price(),
                                                                                      client_truth.get_payment(),
                                                                                      client_truth.get_utility()))
        # get utility for truthfulness test
        return self.client_list[truth_index].get_utility(), self.client_list[truth_index].get_bidding_price()

    def _winners_determination(self, m_client_list=None):
        """
        :param T_max: int
        :param m_client_list: List(Client)
        :return winners_index: List(int)
        """
        pass

    def _get_payment(self, winners_index, critical_client):
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
            payment_1 = self.args.budget_per_round * client_i.get_training_intensity() / \
                tot_training_intensity
            if critical_client is None:
                payment[i] = payment_1
            else:
                payment_2 = client_i.get_training_intensity() * critical_client.get_bidding_price() / \
                    critical_client.get_training_intensity()
                payment[i] = min(payment_1, payment_2)
        # logging.info("payment list" + str(payment))
        return payment

    def _get_utility(self, winners):
        '''
        :param winners: List(int)
        :return: int
        '''
        t_max = 0
        tot_training_intensity = 0
        for index in winners:
            client = self.client_list[index]
            t_max = max(t_max, client.get_time())
            tot_training_intensity += client.get_training_intensity()
        if len(winners) == 0:
            return 0
        return 1.0 * tot_training_intensity / t_max
