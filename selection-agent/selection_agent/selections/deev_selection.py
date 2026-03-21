from flwr.common import log
from logging import INFO
import numpy as np
from math import ceil

class DEEVS:
    name: str = 'poc'
    def __init__(self, decay: float):
        self.decay: float = decay

    def sample_clients(self, clients, sample_size, **kargs):
        log(INFO, "PoC Sample")
        cid_and_loss = [(client['cid'], client['train_loss']) for client in clients]
        clients_sorted_by_loss = sorted(cid_and_loss, key = lambda c: c[1], reverse = True) #ordena por loss decrescente

        n_clients = len(clients_sorted_by_loss) * (1-self.decay)**kargs['curr_rnd']

        clients_ids = [cid for cid, _ in clients_sorted_by_loss[:ceil(n_clients)]]

        selected_clients = [
            client
            for client in clients if client['cid'] in clients_ids
        ]
        return {
            'selected_clients': selected_clients,
            'selection_algorithm': self.name,
            'messages': [],
        }

if __name__ == "__main__":
    deev = DEEVS(0.1)
    print("TESTE PoC SELECTION")
    print(deev.sample_clients(clients = [
        {'cid': 0, 'train_loss': 0.1, 'train_dataset_size': 1233},
        {'cid': 1, 'train_loss': 0.38, 'train_dataset_size': 323},
        {'cid': 2, 'train_loss': 0.13, 'train_dataset_size': 1133},
        {'cid': 3, 'train_loss': 0.48, 'train_dataset_size': 6312},
        {'cid': 4, 'train_loss': 0.56, 'train_dataset_size': 3451},
        {'cid': 5, 'train_loss': 0.95, 'train_dataset_size': 1245},
    ], sample_size = 3, curr_rnd=20))
