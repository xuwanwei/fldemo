import numpy as np

def init_client_bid(client):
    client.update_bid(training_intensity=np.random.randint(1, 10), cost=np.random.randint(2, 10),
                      truth_ratio=1, computation_coefficient=np.random.rand() * 1.5,
                      communication_time=np.random.randint(5, 10))

    # ppig thinks it is ok
    #client.update_bid(training_intensity=np.random.randint(1, 10), cost=np.random.randint(2, 10),
    #                  truth_ratio=1, computation_coefficient=np.random.rand() * 1.5,
    #                  communication_time=np.random.randint(5, 10))
    
    # client.update_bid(training_intensity=np.random.randint(5, 10), cost=np.random.randint(2, 10),
    #                  truth_ratio=1, computation_coefficient=np.random.rand() * 0.8,
    #                  communication_time=np.random.randint(1, 10))

    # client.update_bid(training_intensity=np.random.randint(50, 100), cost=np.random.random() * 3.0 + 2.0,
    #                           truth_ratio=1, computation_coefficient=np.random.rand() * 0.2,
    #                           communication_time=np.random.randint(10, 15))
                                                                                                                                                                                                                                                                                 
def get_client_list(client_index, client_list):
    ret_client_list = []
    for index in client_index:
        ret_client_list.append(client_list[index])
    return ret_client_list


def get_total_training_intensity(clients):
    tot_training_intensity = 0
    for client in clients:
        tot_training_intensity += client.get_training_intensity()
    return tot_training_intensity

def get_objective(ti_sum, t, alpha=1):
    return float(ti_sum) / float(t**alpha)
