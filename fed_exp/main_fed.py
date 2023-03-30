import wandb
import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from utils.args_parser import add_args
from fed_api.utils.utils_func import get_objective
from fed_api.utils.draw import *
from fed_api.fedavg_api import FedAvgAPI
from fed_api.fedbf_api import FedBFAPI
from fed_api.fedopt_api import FedOptAPI
from fed_api.fed3_api import Fed3API
from model.cnn import CNN



def load_data(dataset_name):
    if dataset_name == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.1307], std=[0.3081])])
        train_data = datasets.FashionMNIST(root="./data/fmnist",
                                           transform=transform,
                                           train=True,
                                           download=True)
        test_data = datasets.FashionMNIST(root="./data/fmnist",
                                          transform=transform,
                                          train=False,
                                          download=True)

    elif dataset_name == 'cifar-10':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = datasets.CIFAR10(
            './../data/cifar10', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            './../data/cifar10', train=False, download=True, transform=transform)
    else:
        logging.info("OUT: dataname {}".format(dataset_name))
        sys.exit(0)

    dataset = [train_data, test_data]
    return dataset


def create_model(args):
    if args.model == 'cnn':
        global_net = CNN(args.dataset)
    else:
        global_net = CNN(args.dataset)

    return global_net


def get_API(args, dataset):
    global_net = create_model(args)
    if args.fed_name == "Fed3":
        return Fed3API(device=args.device, args=args, global_net=global_net, train_data=dataset[0], test_data=dataset[1])
    elif args.fed_name == "FedAvg":
        return FedAvgAPI(device=args.device, args=args, global_net=global_net, train_data=dataset[0], test_data=dataset[1])
    elif args.fed_name == "FedOpt":
        return FedOptAPI(device=args.device, args=args, global_net=global_net, train_data=dataset[0], test_data=dataset[1])
    elif args.fed_name == "FedBF":
        return FedBFAPI(device=args.device, args=args, global_net=global_net, train_data=dataset[0], test_data=dataset[1])
    else:
        sys.exit(0)


def get_file_name(args):
    file_name = '{}/{}-{}-{}-C{}-B{}-R{}-S{}-lr{}-al{}-mc{}'.format(args.fed_name, args.fed_name, args.model, args.dataset, args.client_num_in_total, args.budget_per_round, args.comm_round, args.seed, args.lr, args.alpha, args.max_client_num)
    return file_name


def test_truthfulness(args, dataset):
    truth_ratio_list = []
    utility_list = []
    logging.info("####################Truthfulness#####################")
    for truth_ratio in np.arange(0.2, 2, 0.2):
        fed_api = get_API(args=args, dataset=dataset)
        logging.info("Ratio:" + str(truth_ratio))
        client_utility, _ = fed_api.train_for_truthfulness(truth_ratio=truth_ratio)
        truth_ratio_list.append(truth_ratio)
        utility_list.append(client_utility)

    logging.info("####################End##############################")
    logging.info("utility list:" + str(utility_list))
    truth_data = [[round(x, 2), y]
                  for (x, y) in zip(truth_ratio_list, utility_list)]
    truth_table = wandb.Table(data=truth_data, columns=["The ratio of the submitted bid to the truthful cost",
                                                        "The utility of a single buyers"])
    wandb.log(
        {"Performance on truthfulness": wandb.plot.line(truth_table,
                                                        "The ratio of the submitted bid to the truthful cost",
                                                        "The utility of a single buyers",
                                                        title="Performance on truthfulness")})

    file_name = "{}-IC".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)
    draw_IC(file_name)


def test_budget_balance_with_client_num(args, dataset):
    tot_payment_list = []
    client_num_list = []
    budget_list = []
    logging.info("####################Budget Balance#####################")
    for client_num in np.arange(20, 220, 20):
        logging.info("client num:{}".format(client_num))
        args.client_num_in_total = client_num
        fed_api = get_API(args=args, dataset=dataset)
        res = fed_api.test_properties("BB")
        tot_payment_list.append(res.tot_payment)
        budget_list.append(args.budget_per_round)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(
        client_num_list, tot_payment_list, budget_list)]

    file_name = "{}-BB".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    draw_budget_balance(file_name)


def test_individual_rationality(args, dataset):
    logging.info("####################IR#####################")
    fed_api = get_API(args=args, dataset=dataset)
    res = fed_api.test_properties("IR")
    logging.info("####################End##############################")
    
    client_list = np.arange(1, len(res.true_cost)+1)
    truth_data = [[x, y, z] for (x, y, z) in zip(client_list, res.true_cost, res.payment)]

    file_name = "{}-IR".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    draw_individual_rationality(file_name)


def test_with_rounds(args, dataset):
    print("Run {} with {} rounds".format(args.fed_name, args.comm_round))
    fed_api = get_API(args=args, dataset=dataset)
    acc_list, loss_list, time_list, ti_sum_list, round_list, train_loss_list = fed_api.train()
    goal_list = []
    for idx, ti_val in enumerate(ti_sum_list):
        if ti_val == 0:
            goal_list.append(0)
        else:
            goal_list.append(get_objective(ti_val, time_list[idx], args.alpha))

    data_table = [[r, acc, loss, t, ti_sum, goal, train_loss] for (r, acc, loss, t, ti_sum, goal, train_loss) in
                  zip(round_list, acc_list, loss_list, time_list, ti_sum_list, goal_list, train_loss_list)]

    file_name = "{}-R".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)


'''value-budget'''
def test_with_budget(args, dataset):
    # acc_list = []
    # loss_list = []
    time_list = []
    ti_sum_list = []
    obj_list = []
    budget_list = []
    left = 10
    right = 100
    step = 10
    for budget in range(left, right, step):
        args.budget_per_round = budget
        logging.info("#########Budget:{}###########".format(budget))
        fedAPI = get_API(args, dataset)
        test_result = fedAPI.test_properties()
        # logging.info("t_ti_sum_list:{}".format(t_ti_sum_list))
        time_list.append(test_result.time)
        ti_sum_list.append(test_result.training_intensity)
        obj_list.append(test_result.objective)
        budget_list.append(budget)

    data_table = [[b, t, ti_sum, obj] for (b, t, ti_sum, obj) in
                  zip(budget_list, time_list, ti_sum_list, obj_list)]

    # writing data to file
    file_name = "{}/C{}-B-{}-l{}-r{}-s{}".format(args.fed_name, args.client_num_in_total, args.fed_name, left, right, step)
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)

'''value-client nums'''
def test_with_client_nums(args, dataset):
    time_list = []
    ti_sum_list = []
    obj_list = []
    client_nums_list = []
    for client_nums in range(50, 500, 10):
        args.client_num_in_total = client_nums
        logging.info("#########client   nums:{} ###########".format(client_nums))
        fedAPI = get_API(args, dataset)
        test_result = fedAPI.test_properties()
        # logging.info("t_ti_sum_list:{}".format(t_ti_sum_list))
        time_list.append(test_result.time)
        ti_sum_list.append(test_result.training_intensity)
        obj_list.append(test_result.objective)
        client_nums_list.append(client_nums)

    data_table = [[c, t, ti_sum, obj] for (c, t, ti_sum, obj) in
                  zip(client_nums_list, time_list, ti_sum_list, obj_list)]

    # writing data to file
    file_name = "{}-C".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = add_args(argparse.ArgumentParser(description='fed-standalone'))
    args = parser.parse_args()
    # args.file_name = args.dataset + "-" + args.model + "-C" + args.client_num_in_total +"-B" + args.budget_per_round + "-lr" +args.learning_rate + "-S" +args.seed+args.fed_name
    logger.info(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    logger.info(args.device)

    wandb.init(
        project="fedml",
        name="{}-b-{}-c-{}-cn-{}-s-{}".format(args.fed_name, args.budget_per_round,
                                              args.comm_round, args.client_num_in_total, args.seed),
        config=args
    )

    logging.info("before torch")
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)  # set the seed for numpy
    torch.manual_seed(args.seed)  # set the seed for generating random numbers
    # Set the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed(args.seed)
    # set the seed for generating random numbers on all GPUs.
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    dataset = load_data(args.dataset)

    # Test economic properties
    # test_truthfulness(args, dataset)
    # test_individual_rationality(args, dataset)
    # test_budget_balance_with_client_num(args, dataset)
    # test_with_budget(args)

    # Test Accuracy and Time
    test_with_rounds(args, dataset)

    # test_with_budget(args, dataset)
    # test_with_client_nums(args, dataset)
