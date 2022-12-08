import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # federated learning arguments
    parser.add_argument('--rounds', type = int, default = 50, help = "number of communication rounds")
    parser.add_argument('--num_users', type = int, default = 100, help = "number of users: N")
    parser.add_argument('--frac', type = float, default = 0.1, help = "the fraction of clients for every epoch: frac")
    parser.add_argument('--local_ep', type = int, default = 5, help = "local training epoch: bs")
    parser.add_argument('--local_bs', type = int, default = 128, help = "local batch size: bs")
    parser.add_argument('--lr', type = float, default = 0.001, help = "learning rate")
    parser.add_argument('--momentum', type = float, default = 0.5, help = "SGD momentum (default: 0.5)")
    parser.add_argument('--modeltype', type = str, default = 'cnn', help = 'machine learning model')

    # dataset arguments
    parser.add_argument('--dataset', type = str, default = 'mnist', help = 'dataset name')
    parser.add_argument('--iid', action = 'store_true', help = 'whether i.i.d or not')
    parser.add_argument('--uniform', action = 'store_true', help = 'whether uniform or not')
    
    # running arguments
    parser.add_argument('--gpu', type = int, default = -1, help = "GPU ID, -1 for CPU")
    parser.add_argument('--seed', type = int, default = 10, help = 'random seed (default: 10)')

    args = parser.parse_args()
    return args

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='cnn', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar-10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--local_bs', type = int, default = 128, help = "local batch size: bs")

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    parser.add_argument('--budget_per_round', type=float,
                        default=100, help='the budget of the server in a round')

    parser.add_argument('--seed', type=int, default=1104,
                        help='numpy random seed')

    parser.add_argument('--draw', type=bool, default=False, help='draw pic')

    parser.add_argument('--fed_name', type=str, default="Fed3", help="the fed api chosen to run")

    parser.add_argument('--iid', action = 'store_true', help = 'whether i.i.d or not')

    parser.add_argument('--uniform', action = 'store_true', help = 'whether uniform or not')

    parser.add_argument('--momentum', type = float, default = 0.5, help = "SGD momentum (default: 0.5)")

    return parser