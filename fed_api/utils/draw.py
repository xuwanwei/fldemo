import csv
import time

from matplotlib import pyplot as plt
import numpy as np

DATA_PATH_PRE = "../OutputData"
IMG_PATH_PRE = "../OutputImage"

in_filename = "fedopt-1104-INFO-2022-11-27-16-12"

filename_3 = "C200-Fed3-B"
filename_avg = "C200-FedAvg-B"
filename_bf = "C200-FedBF-B"
filename_opt = "C200-FedOpt-B"
filename_time = "fedtime-20220924-2022-12-15-23-24-C"

timestamp = time.time()
datatime = time.strftime("%Y-%m-%d-%H-%M", time.localtime(timestamp))
out_file_name = 'fed3-{}-{}'.format(1104, datatime)
OUTPUT_FILENAME = "fed3-avg-bf-{}".format(datatime)


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
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')
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

    # x = ['20', '40', '60', '80', '100', '120', '140', '160','180']
    # x_len = np.arange(len(x))
    # total_width = 2
    width = 0.6
    # xticks = x_len - (total_width - width) / 2
    xticks = np.arange(len(budget_list))

    plt.bar(xticks, budget_list, label='Budget', width=width)
    plt.bar(xticks, tot_payment_list, label='Total Payment', width=width)
    plt.ylim(0, 200)

    # plt.xticks(x_len, client_num_list)
    plt.xticks(xticks, client_num_list)
    plt.ylabel("The total payment and budget")
    plt.xlabel("the number of clients")
    plt.title("Budget Balance")
    plt.legend(loc="upper right")
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')
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
            print("client:{}, true_cost:{}, payment:{}".format(
                row[0], row[1], row[2]))
            client_num_list.append(int(row[0]))
            cost_list.append(float(row[1]))
            payment_list.append(float(row[2]))

    ind = [i for i, _ in enumerate(client_num_list)]
    plt.bar(ind, payment_list, label='payment')
    plt.bar(ind, cost_list, label='real cost')
    plt.ylim(0, 10)

    plt.xticks(ind, client_num_list)
    plt.ylabel("The real cost/payment")
    plt.xlabel("The selected winning bids")
    plt.title("Individual Rationality")
    plt.legend(loc="upper right")
    plt.savefig("{}/{}.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')
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
            # print("round:{}, acc:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            acc_list.append(float(row[1]))

    plt.figure()
    plt.plot(round_list, acc_list)
    plt.title("Tested Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-ACC.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')


def draw_loss(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, loss:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[2]))

    print("round_list:{}".format(round_list))
    plt.figure()
    plt.plot(round_list, loss_list)
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-LOSS.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')


def draw_time(file_name=None):
    if file_name is None:
        file_name = in_filename
    round_list = []
    loss_list = []
    print("drawing:{}/{}.csv".format(DATA_PATH_PRE, file_name))
    with open("{}/{}.csv".format(DATA_PATH_PRE, file_name), mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # print("round:{}, time:{}".format(row[0], row[1]))
            round_list.append(int(row[0]))
            loss_list.append(float(row[3]))

    # print("round_list:{}".format(round_list))
    plt.figure()
    plt.plot(round_list, loss_list)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Rounds")
    plt.savefig("{}/{}-TIME.png".format(IMG_PATH_PRE, file_name), bbox_inches = 'tight')


def _get_cmp_list(args, name="ACC", nums=None):
    x_list = []
    y_list = []

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

    for f_i, m_filepath in enumerate(args.filepath_cmp):
        temp_y_list = []
        with open("{}/{}.csv".format(DATA_PATH_PRE, m_filepath), mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i % args.frequency != 0:
                    continue
                if nums is not None and i >= nums:
                    break
                # print("round:{}, {}:{}".format(row[0], name, row[idx]))
                if f_i == 0:
                    x_list.append(int(row[0]))
                temp_y_list.append(float(row[idx]))
        y_list.append(temp_y_list)

    dataset = [x_list, y_list]
    return dataset


def _draw_cmp(dataset, name="ACC", x_name="Rounds", legend=None):
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
    y_data = dataset[1]
    for y_list in y_data:
        plt.plot(dataset[0], y_list)

    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel(x_name)
    if len(legend)==0:
        legend = ['ours', 'bid price first', 'FedAvg']
    plt.legend(legend)
    output_filename = "{}-{}".format(OUTPUT_FILENAME, name)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename), bbox_inches = 'tight')
    

# TODO: dataset changes
def _draw_cmp_bar(dataset, name="T", x_name="Rounds"):
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

    if x_name == "Total numbers of clients":
        bar_width = 25
    elif x_name == "Budget":
        bar_width = 5
    freq = 10
    plt.figure()
    x_1 = [dataset[0][i] for i in range(0, len(dataset[0]), freq)]
    x_2 = [i+bar_width for i in x_1]
    x_3 = [i+bar_width*2 for i in x_1]
    y_data = dataset[1]
    y_1 = [y_data[1][i] for i in range(0, len(dataset[0]), freq)]
    y_2 = [y_data[2][i] for i in range(0, len(dataset[0]), freq)]
    y_3 = [y_data[3][i] for i in range(0, len(dataset[0]), freq)]
    
    plt.bar(x_1, y_1, width=bar_width)
    plt.bar(x_2, y_2, width=bar_width)
    plt.bar(x_3, y_3, width=bar_width)
    
    x_tick = [i for i in x_2]
    plt.xticks(x_tick, x_1)
    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel(x_name)
    plt.legend(['ours', 'bid price first', 'FedAvg'])
    plt.ylim(0, 30)
    output_filename = "{}-{}".format(OUTPUT_FILENAME, name)
    # plt.savefig('./important/pics/FMNIST_TI_ACC.png', bbox_inches='tight')
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename), bbox_inches = 'tight')


def draw_accuracy_cmp(args, nums=None):
    dataset = _get_cmp_list(args, "ACC", nums)
    _draw_cmp(dataset, "ACC", "Rounds")


def draw_loss_cmp(args, nums=None):
    dataset = _get_cmp_list(args, "LOSS", nums)
    _draw_cmp(dataset, "LOSS", "Rounds")


def draw_train_loss_cmp(args, nums=None):
    dataset = _get_cmp_list(args, "TLOSS", nums)
    _draw_cmp(dataset, "TLOSS", "Rounds")


def draw_time_cmp(args):
    dataset = _get_cmp_list(args, name="T")
    _draw_cmp(dataset, "T", "Rounds")


def draw_training_intensity_sum_cmp(args):
    dataset = _get_cmp_list(args, "TI")
    _draw_cmp(dataset, "TI", "Rounds")


def draw_goal_cmp(args):
    dataset = _get_cmp_list(args,"OBJ")
    _draw_cmp(dataset, "OBJ", "Rounds")


def _get_time_list(args, name="ACC", nums=None):
    if name == "ACC":
        idx = 1
    elif name == "LOSS":
        idx = 2
    elif name == "TLOSS":
        idx = 6
    
    time_list = []
    y_list = []

    for m_filename in args.filepath_cmp:
        temp_time_list = []
        temp_y_list = []
        with open("{}/{}.csv".format(DATA_PATH_PRE, m_filename), mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i % args.frequency != 0:
                    continue
                if nums is not None and i >= nums:
                    break
                # print("time:{}, {}:{}".format(row[3], name, row[idx]))
                if len(temp_time_list) == 0:
                    temp_time_list.append(float(row[3]))
                else:
                    temp_time_list.append(temp_time_list[-1] + float(row[3]))
                temp_y_list.append(float(row[idx]))
        time_list.append(temp_time_list)
        y_list.append(temp_y_list)

    return time_list, y_list

def _draw_time_list(name, time_data, val_data, legend=None):
    plt.figure()
    for index, time_list in enumerate(time_data):
        plt.plot(time_list, val_data[index])

    if name == "ACC":
        title_name = "Tested Accuracy"
        y_label = "Accuracy"
    elif name == "LOSS":
        title_name = "Tested Loss"
        y_label = "Loss"
    elif name == "TLOSS":
        title_name = "Train Loss"
        y_label = "Loss"
    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel("Time")

    if legend == None:
        legend = ['ours', 'bid price first', 'FedAvg']
    print(legend)
    plt.legend(legend)
    output_filename = "{}-T-{}".format(OUTPUT_FILENAME, name)
    plt.savefig("{}/cmp/{}.png".format(IMG_PATH_PRE, output_filename), bbox_inches = 'tight')  
    print(out_file_name)


def draw_accuracy_cmp_with_time(args, rounds=None):
    time_dataset, val_dataset = _get_time_list(args, "ACC", rounds)
    _draw_time_list("ACC", time_dataset, val_dataset)
    

def draw_loss_cmp_with_time(args, rounds=None):
    time_dataset, val_dataset = _get_time_list(args, "LOSS", rounds)
    _draw_time_list("LOSS", time_dataset, val_dataset)


def draw_train_loss_cmp_with_time(args, rounds=None):
    time_dataset, val_dataset = _get_time_list(args, "TLOSS", rounds)
    _draw_time_list("TLOSS", time_dataset, val_dataset)


def _get_prop_list(args, metrics = "T"):
    if metrics == "T":
        idx = 1
    elif metrics == "TI":
        idx = 2
    elif metrics == "OBJ":
        idx = 3
    else:
        print("WRONG BUDGET METRICS")
        return 
    x_list = []
    y_list = []

    sample_freq = args.frequency
    for i, m_filename in enumerate(args.filepath_cmp):
        temp_y_list = []
        with open("{}/{}.csv".format(DATA_PATH_PRE, m_filename), mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for r_idx, row in enumerate(reader):
                if r_idx % sample_freq != 0:
                    continue
                if i == 0 :
                    x_list.append(int(row[0]))
                temp_y_list.append(float(row[idx]))

    dataset = [x_list, y_list]
    return dataset


def draw_goal_cmp_with_x(args):
    dataset = _get_prop_list(args, "OBJ")
    _draw_cmp(dataset=dataset, name="OBJ", x_name=args.xlabel)


def draw_time_cmp_with_x(args):
    dataset = _get_prop_list(args, "T")
    _draw_cmp_bar(dataset=dataset, name="T", x_name=args.xlabel)


def draw_training_intensity_cmp_with_x(args):
    dataset = _get_prop_list(args, "TI")
    _draw_cmp(dataset=dataset, name="TI", x_name=args.xlabel)


def draw_accuracy_cmp_para(args, nums=None):
    dataset = _get_cmp_list(args, "ACC", nums)
    _draw_cmp(dataset, "ACC", "Rounds", args.legend)

def draw_accuracy_cmp_with_time_para(args, rounds=None):
    time_dataset, val_dataset = _get_time_list(args, "ACC", rounds)
    _draw_time_list("ACC", time_dataset, val_dataset, args.legend)

def draw_train_loss_cmp_para(args, nums=None):
    dataset = _get_cmp_list(args, "TLOSS", nums)
    _draw_cmp(dataset, "TLOSS", "Rounds", args.legend)


def draw_train_loss_cmp_with_time_para(args, rounds=None):
    time_dataset, val_dataset = _get_time_list(args, "TLOSS", rounds)
    _draw_time_list("TLOSS", time_dataset, val_dataset, args.legend)