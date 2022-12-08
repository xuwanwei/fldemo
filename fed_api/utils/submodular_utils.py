class SubmodularSet:
    def __init__(self, client_set, alpha=1):
        """
            client_set: List(Client)
            alpha: int
        """
        self.client_set = client_set
        self.alpha = alpha

    def update_client_to_set(self, client):
        self.client_set.append(client)

    def get_buyer_utility(self, m_set=None):
        t_max = 0
        tot_training_intensity = 0
        if m_set is None:
            m_set = self.client_set
        for client in m_set:
            t_max = max(client.get_time(), t_max)
            tot_training_intensity += client.get_training_intensity()
        return t_max - self.alpha * tot_training_intensity

    # TODO: positive or negative
    def get_submodular(self, m_set):
        """m_set: List(Client)"""
        # f(S) = U(S)-U(0)
        # f(s): smaller is better
        return self.get_buyer_utility(m_set)

    def get_dif_submodular_with_bid(self, m_time, m_training_intensity):
        """m_time, m_training_intensity are the info of the bid"""
        # delta f(s) = f(S) - f(S \cup {s})
        # delta f(s): bigger is better
        t_max_s = 0
        for client in self.client_set:
            t_max_s = max(client.get_time(), t_max_s)
        t_max_s_and_client = max(m_time, t_max_s)
        return t_max_s - t_max_s_and_client + self.alpha * m_training_intensity

    def get_dif_submodular_exclude_client(self, m_client_set, client_exclude_id, client_end_id):
        """m_client_set : List(Client)"""
        t_max_s = 0
        # index starts from 0
        if m_client_set is None:
            m_client_set = self.client_set
        for index, client in enumerate(m_client_set):
            if index == client_exclude_id:
                continue
            if index == client_end_id:
                break
            t_max_s = max(t_max_s, client.get_time())
        t_max_s_and_client = max(t_max_s, m_client_set[client_exclude_id])
        return t_max_s - t_max_s_and_client + self.alpha * m_client_set[client_exclude_id].get_training_intensity
