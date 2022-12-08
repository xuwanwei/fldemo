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
    timestamp = time.time()
    datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
    if args.fed_name == "Fed3":
        file_name = 'Fed3/fed3-{}-{}'.format(args.seed, datatime)
    elif args.fed_name == "FedAvg":
        file_name = 'FedAvg/fedavg-{}-{}'.format(args.seed, datatime)
    elif args.fed_name == "FedOpt":
        file_name = 'FedOpt/fedopt-{}-{}'.format(args.seed, datatime)
    elif args.fed_name == "FedBF":
        file_name = 'FedBF/fedbf-{}-{}'.format(args.seed, datatime)
    else:
        sys.exit(0)
    return file_name


def test_truthfulness(device, args, dataset):
    truth_ratio_list = []
    utility_list = []
    logging.info("####################Truthfulness#####################")
    for truth_ratio in np.arange(0.2, 2, 0.2):
        fed_api = get_API(args=args, dataset=dataset)
        logging.info("Ratio:" + str(truth_ratio))
        client_utility, client_bidding_price = fed_api.train_for_truthfulness(
            truth_ratio=truth_ratio)
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
    if args.draw:
        draw_IC(file_name)


def test_budget_balance_with_client_num(device, args, dataset):
    tot_payment_list = []
    client_num_list = []
    budget_list = []
    logging.info("####################Budget Balance#####################")
    for client_num in np.arange(10, 100, 10):
        args.client_num_in_total = client_num
        fed_api = get_API(args=args, dataset=dataset)
        res = fed_api.test_properties()
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

    if args.draw:
        draw_budget_balance(file_name)


def test_individual_rationality(device, args, dataset, model_trainer):
    payment_list = []
    true_cost_list = []
    client_num_list = []
    logging.info("####################IR#####################")
    for client_num in np.arange(10, 100, 10):
        args.client_num_in_total = client_num
        fed_api = get_API(args=args, dataset=dataset)
        res = fed_api.test_properties()
        payment_list.append(res.payment)
        true_cost_list.append(res.true_cost)
        client_num_list.append(client_num)

    logging.info("####################End##############################")
    truth_data = [[x, y, z] for (x, y, z) in zip(
        client_num_list, true_cost_list, payment_list)]

    file_name = "{}-IC".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(truth_data)

    if args.draw:
        draw_individual_rationality(file_name)


def test_with_rounds(dataset, args):
    print("Run {} with {} rounds".format(args.fed_name, args.comm_round))
    fed_api = get_API(args=args, dataset=dataset)
    acc_list, loss_list, time_list, ti_sum_list, round_list, train_loss_list = fed_api.train()
    goal_list = []
    for idx, ti_val in enumerate(ti_sum_list):
        if ti_val == 0:
            goal_list.append(0)
        else:
            goal_list.append(float(ti_val) / float(time_list[idx]))

    data_table = [[r, acc, loss, t, ti_sum, goal, train_loss] for (r, acc, loss, t, ti_sum, goal, train_loss) in
                  zip(round_list, acc_list, loss_list, time_list, ti_sum_list, goal_list, train_loss_list)]

    file_name = "{}-R".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)

    if args.draw:
        draw_accuracy(file_name)
        draw_loss(file_name)
        draw_time(file_name)


def test_with_budget(dataset, device, args, model_trainer):
    acc_list = []
    loss_list = []
    time_list = []
    ti_sum_list = []
    goal_list = []
    budget_list = []
    for budget in range(4, 80, 4):
        args.budget_per_round = budget
        fed3API = Fed3API(device=device, args=args,
                          dataset=dataset, model_trainer=model_trainer)
        t_acc_list, t_loss_list, t_time_list, t_ti_sum_list, _, train_loss_list = fed3API.train()
        logging.info("client num in tot:{}".format(args.client_num_in_total))
        # logging.info("t_ti_sum_list:{}".format(t_ti_sum_list))
        t_goal_list = []
        for idx, ti_val in enumerate(t_ti_sum_list):
            if ti_val == 0:
                t_goal_list.append(0)
            else:
                t_goal_list.append(float(ti_val) / float(t_time_list[idx]))
        logging.info("budget:{}, time:{}".format(budget, t_time_list))
        if len(t_acc_list) == 0:
            acc_list.append(0)
            loss_list.append(0)
            time_list.append(0)
            ti_sum_list.append(0)
            goal_list.append(0)
        else:
            acc_list.append(np.mean(t_acc_list))
            loss_list.append(np.mean(t_loss_list))
            time_list.append(np.mean(t_time_list))
            ti_sum_list.append(np.mean(t_ti_sum_list))
            goal_list.append(np.mean(t_goal_list))
        budget_list.append(budget)

    data_table = [[b, acc, loss, t, ti_sum, goal, train_loss] for (b, acc, loss, t, ti_sum, goal, train_loss) in
                  zip(budget_list, acc_list, loss_list, time_list, ti_sum_list, goal_list, train_loss_list)]

    # writing data to file
    file_name = "{}-B".format(get_file_name(args))
    print("writing {}".format(file_name))
    with open('{}/{}.csv'.format(DATA_PATH_PRE, file_name), mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data_table)

    if args.draw:
        draw_accuracy_budget(file_name)
        draw_loss_budget(file_name)
        draw_time_budget(file_name)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = add_args(argparse.ArgumentParser(description='fed-standalone'))
    args = parser.parse_args()
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

    logging.info("after torch")
    # load data
    dataset = load_data(args.dataset)

    logging.info("after data")

    # Test economic properties
    # test_truthfulness(device, args, dataset, model_trainer)
    # test_budget_balance_with_client_num(device, args, dataset, model_trainer)
    # test_individual_rationality(device, args, dataset, model_trainer)
    # test_with_budget(device, args)

    logging.info("before rounds")
    # Test Accuracy and Time
    test_with_rounds(dataset, args)
    logging.info("done")

    # test_with_budget(dataset, device, args, model_trainer)
