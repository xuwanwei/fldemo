import csv
import logging
import time

from matplotlib import pyplot as plt

DATA_PATH_PRE = "../OutputData"
IMG_PATH_PRE = "../OutputImage"

in_filename = "fedopt-1104-INFO-2022-11-27-16-12"

filename_3 = "fed3-20221205-2022-12-08-12-23-R"
filename_avg = "fedavg-20221205-2022-12-08-12-20-R"
filename_bf = "fedbf-20221205-2022-12-08-12-27-R"
filename_opt = "fedopt-20221205-2022-12-08-12-29-R"

DATA_PATH_3 = "../../../OutputData/fed_3"
DATA_PATH_OPT = "../../../OutputData/fed_opt"
DATA_PATH_BF = "../../../OutputData/fed_bf"
DATA_PATH_AVG = "../../../OutputData/fed_avg"
timestamp = time.time()
datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
out_file_name = 'fed3-{}-{}'.format(1104, datatime)
OUTPUT_FILENAME = "fed3-opt-bf-{}".format(datatime)


def draw_IC(file_name=None):
    if file_name is None:
        file_name = in_filename
    ratio = []
    utility = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("ratio:{}, utility:{}".format(row[0], row[1]))
            ratio.append(float(row[0]))
            utility.append(float(row[1]))

    plt.plot(ratio, utility, marker='o')
    plt.title("Incentive compatibility")
    plt.ylabel("Utility of Clients")
    plt.xlabel("Ratio")
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name))
    plt.show()


def draw_budget_balance(file_name=None):
    if file_name is None:
        file_name = in_filename
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        budget_list = []
        client_num_list = []
        tot_payment_list = []
        for row in reader:
            print("number of clients:{}, tot_payment:{}, budget:{}".format(
                row[0], row[1], row[2]))
            budget_list.append(float(row[2]))
            client_num_list.append(int(row[0]))
            tot_payment_list.append(float(row[1]))
    ind = [i for i, _ in enumerate(client_num_list)]
    plt.bar(ind, budget_list, label='Budget')
    plt.bar(ind, tot_payment_list, label='Total Payment')

    plt.xticks(ind, client_num_list)
    plt.ylabel("The total payment and budget")
    plt.xlabel("the number of clients")
    plt.title("Budget Balance")
    plt.legend(loc="upper right")
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name))
    plt.show()


def draw_individual_rationality(file_name=None):
    if file_name is None:
        file_name = in_filename
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        client_num_list = []
        payment_list = []
        cost_list = []
        for row in reader:
            print("number of clients:{}, true_cost:{}, payment:{}".format(
                row[0], row[1], row[2]))
            client_num_list.append(int(row[0]))
            cost_list.append(float(row[1]))
            payment_list.append(float(row[2]))

    ind = [i for i, _ in enumerate(client_num_list)]
    plt.bar(ind, payment_list, label='payment')
    plt.bar(ind, cost_list, label='real cost')

    plt.xticks(ind, client_num_list)
    plt.ylabel("The real cost and payment")
    plt.xlabel("the number of clients")
    plt.title("Individual Rationality")
    plt.legend(loc="upper right")
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name))
    plt.show()


def draw_accuracy(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    acc_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, acc:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            acc_list.append(float(row[1]))

    plt.figure()
    plt.plot(round_list, acc_list)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-ACC.png".format(IMG_PATH_PRE, file_name))


def draw_loss(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, loss:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[2]))

    print("round_list:{}".format(round_list))
    plt.figure()
    plt.plot(round_list, loss_list)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-LOSS.png".format(IMG_PATH_PRE, file_name))


def draw_time(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("round:{}, time:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[3]))

    # print("round_list:{}".format(round_list))
    plt.figure()
    plt.plot(round_list, loss_list)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-TIME.png".format(IMG_PATH_PRE, file_name))


def _get_cmp_list(name="ACC"):
    x_list = []
    y_list_1 = []
    y_list_2 = []
    y_list_3 = []
    y_list_4 = []

    if name == "ACC":
        idx = 1
    elif name == "LOSS":
        idx = 2
    elif name == "T":
        idx = 3
    elif name == "TI":
        idx = 4
    elif name == "OBJ":
        idx = 5
    elif name == "TLOSS":
        idx = 6
    with open("{}/Fed3/{}.csv".format(DATA_PATH_PRE, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, {}:{}".format(row[0], name, row[idx]))
            x_list.append(int(row[0]))
            y_list_1.append(float(row[idx]))

    with open("{}/FedOpt/{}.csv".format(DATA_PATH_PRE, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, {}:{}".format(row[0], name, row[idx]))
            y_list_2.append(float(row[idx]))

    with open("{}/FedBF/{}.csv".format(DATA_PATH_PRE, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, {}:{}".format(row[0], name, row[idx]))
            y_list_3.append(float(row[idx]))

    with open("{}/FedAvg/{}.csv".format(DATA_PATH_PRE, filename_avg), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, {}:{}".format(row[0], name, row[idx]))
            y_list_4.append(float(row[idx]))

    dataset = [x_list, y_list_1, y_list_2, y_list_3, y_list_4]
    return dataset


def _draw_cmp(dataset, name="ACC", x_name="Rounds"):
    if name == "ACC":
        y_label = "Accuracy"
        title_name = "Tested Accuracy"
    elif name == "LOSS":
        y_label = "Loss"
        title_name = "Tested Loss"
    elif name == "T":
        y_label = "Time"
        title_name = "Time"
    elif name == "TI":
        y_label = "Training Intensity"
        title_name = "Training Intensity"
    elif name == "OBJ":
        y_label = "Objective"
        title_name = "Objective"
    elif name == "TLOSS":
        y_label = "Training Loss"
        title_name = "Training Loss"


    plt.figure()
    plt.plot(dataset[0], dataset[1])
    plt.plot(dataset[0], dataset[2])
    plt.plot(dataset[0], dataset[3])
    plt.plot(dataset[0], dataset[4])

    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel(x_name)
    plt.legend(['ours', 'optimal', 'bid price first', 'FedAvg'])
    output_filename = "{}-{}".format(OUTPUT_FILENAME, name)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    

def draw_accuracy_cmp():
    dataset = _get_cmp_list("ACC")
    _draw_cmp(dataset, "ACC", "Rounds")


def draw_loss_cmp():
    dataset = _get_cmp_list("LOSS")
    _draw_cmp(dataset, "LOSS", "Rounds")


def draw_time_cmp():
    dataset = _get_cmp_list("T")
    _draw_cmp(dataset, "T", "Rounds")


def draw_training_intensity_sum_cmp():
    dataset = _get_cmp_list("TI")
    _draw_cmp(dataset, "TI", "Rounds")

def draw_goal_cmp():
    dataset = _get_cmp_list("OBJ")
    _draw_cmp(dataset, "OBJ", "Rounds")


def draw_accuracy_budget(file_name=None):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    acc_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            acc_list.append(float(row[1]))

    plt.figure()
    plt.plot(budget_list, acc_list)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Budget")
    plt.savefig("{}/{}-ACC.png".format(IMG_PATH_PRE, file_name))
    # plt.show()


def draw_loss_budget(file_name=None):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, loss:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            loss_list.append(float(row[2]))

    plt.figure()
    plt.plot(budget_list, loss_list)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Budget")
    plt.savefig("{}/{}-LOSS.png".format(IMG_PATH_PRE, file_name))
    # plt.show()


def draw_time_budget(file_name=None):
    if file_name is None:
        file_name = in_filename
    budget_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[1]))
            budget_list.append(int(row[0]))
            loss_list.append(float(row[3]))

    print("budget_list:{}".format(budget_list))
    plt.figure()
    plt.plot(budget_list, loss_list)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Budget")
    plt.savefig("{}/{}-TIME.png".format(IMG_PATH_PRE, file_name))
    # plt.show()

def _get_time_list(name="ACC"):
    if name == "ACC":
        idx = 1
    elif name == "LOSS":
        idx = 2
    elif name == "TLOSS":
        idx = 6
    
    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    time_list_4 = []

    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []

    with open("{}/Fed3/{}.csv".format(DATA_PATH_PRE, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, {}:{}".format(row[3], name, row[idx]))
            if len(time_list_1) == 0:
                time_list_1.append(float(row[3]))
            else:
                time_list_1.append(time_list_1[-1] + float(row[3]))
            list_1.append(float(row[idx]))

    with open("{}/FedOpt/{}.csv".format(DATA_PATH_PRE, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, {}:{}".format(row[3], name, row[idx]))
            if len(time_list_2) == 0:
                time_list_2.append(float(row[3]))
            else:
                time_list_2.append(time_list_2[-1] + float(row[3]))
            list_2.append(float(row[idx]))

    with open("{}/FedBF/{}.csv".format(DATA_PATH_PRE, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, {}:{}".format(row[3], name, row[idx]))
            if len(time_list_3) == 0:
                time_list_3.append(float(row[3]))
            else:
                time_list_3.append(time_list_3[-1] + float(row[3]))
            list_3.append(float(row[idx]))

    with open("{}/FedAvg/{}.csv".format(DATA_PATH_PRE, filename_avg), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("time:{}, {}:{}".format(row[3], name, row[idx]))
            if len(time_list_4) == 0:
                time_list_4.append(float(row[3]))
            else:
                time_list_4.append(time_list_4[-1] + float(row[3]))
            list_4.append(float(row[idx]))

    time_dataset = [time_list_1, time_list_2, time_list_3, time_list_4]
    val_dataset = [list_1, list_2, list_3, list_4]
    return time_dataset, val_dataset

def _draw_time_list(name, time_data, val_data):
    plt.figure()
    plt.plot(time_data[0], val_data[0])
    plt.plot(time_data[1], val_data[1])
    plt.plot(time_data[2], val_data[2])
    plt.plot(time_data[3], val_data[3])

    if name == "ACC":
        title_name = "Tested Accuracy"
        y_label = "Accuracy"
    elif name == "LOSS":
        title_name = "Tested Loss"
        y_label = "Loss"
    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel("Time")
    plt.legend(['ours', 'optimal', 'bid price first', 'FedAvg'])
    output_filename = "{}-T-{}".format(OUTPUT_FILENAME, name)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))  

def draw_accuracy_cmp_with_time():
    time_dataset, val_dataset = _get_time_list("ACC")
    _draw_time_list("ACC", time_dataset, val_dataset)
    

def draw_loss_cmp_with_time():
    time_dataset, val_dataset = _get_time_list("LOSS")
    _draw_time_list("LOSS", time_dataset, val_dataset)


def draw_time_cmp_with_budget():
    budget_list = []

    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    time_list_4 = []

    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[3]))
            budget_list.append(float(row[0]))
            time_list_1.append(float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[3]))
            time_list_2.append(float(row[3]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[3]))
            time_list_3.append(float(row[3]))

    with open("{}/fed_avg/{}.csv".format(DATA_PATH_PRE, filename_avg), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[3]))
            time_list_4.append(float(row[3]))

    plt.figure()
    plt.plot(budget_list, time_list_1)
    plt.plot(budget_list, time_list_2)
    plt.plot(budget_list, time_list_3)
    plt.plot(budget_list, time_list_4)

    plt.title("Tested Time")
    plt.ylabel("Time")
    plt.xlabel("Budget")
    plt.legend(['ours', 'optimal', 'bid price first', 'FedAvg'])
    output_filename = "B-T-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    # plt.show()


def draw_accuracy_cmp_with_budget():
    budget_list = []

    acc_list_1 = []
    acc_list_2 = []
    acc_list_3 = []
    acc_list_4 = []

    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[1]))
            budget_list.append(float(row[0]))
            acc_list_1.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[1]))
            acc_list_2.append(float(row[1]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[1]))
            acc_list_3.append(float(row[1]))

    with open("{}/fed_avg/{}.csv".format(DATA_PATH_PRE, filename_avg), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[1]))
            acc_list_4.append(float(row[1]))

    plt.figure()

    plt.plot(budget_list, acc_list_1)
    plt.plot(budget_list, acc_list_2)
    plt.plot(budget_list, acc_list_3)
    plt.plot(budget_list, acc_list_4)

    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Budget")
    plt.legend(['ours', 'optimal', 'bid price first', 'FedAvg'])
    output_filename = "B-ACC-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    # plt.show()


def draw_goal_cmp_with_budget():
    budget_list = []

    time_list_1 = []
    time_list_2 = []
    time_list_3 = []
    time_list_4 = []

    with open("{}/{}.csv".format(DATA_PATH_3, filename_3), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, acc:{}".format(row[0], row[5]))
            budget_list.append(float(row[0]))
            time_list_1.append(float(row[5]))

    with open("{}/{}.csv".format(DATA_PATH_OPT, filename_opt), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[5]))
            time_list_2.append(float(row[5]))

    with open("{}/{}.csv".format(DATA_PATH_BF, filename_bf), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[5]))
            time_list_3.append(float(row[5]))

    with open("{}/fed_avg/{}.csv".format(DATA_PATH_PRE, filename_avg), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            print("budget:{}, time:{}".format(row[0], row[5]))
            time_list_4.append(float(row[5]))

    plt.figure()
    plt.plot(budget_list, time_list_1)
    plt.plot(budget_list, time_list_2)
    plt.plot(budget_list, time_list_3)
    plt.plot(budget_list, time_list_4)

    plt.title("Tested Objective")
    plt.ylabel("Objective")
    plt.xlabel("Budget")
    plt.legend(['ours', 'optimal', 'bid price first', 'FedAvg'])
    output_filename = "B-O-{}".format(OUTPUT_FILENAME)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename))
    # plt.show()
