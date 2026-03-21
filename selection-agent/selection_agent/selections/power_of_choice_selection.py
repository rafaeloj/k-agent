from flwr.common import log
from logging import INFO
import numpy as np
# from selection_agent.selections.selection import Selection

# class PowerOfChoice(Selection):
class PowerOfChoice:
    name: str = 'poc'
    def sample_clients(self, clients, sample_size, **kargs):
        log(INFO, "PoC Sample")
        dataset_sizes = [client['train_dataset_size'] for client in clients]
        # print(clients)
        total_size = sum(dataset_sizes)
        weights = [size / total_size for size in dataset_sizes]
        if len(clients) < sample_size:
            return clients
        random_selected_clients = np.random.choice(clients, p=weights, size=sample_size, replace=False).tolist()
        # Loss-based selection
        clients_sorted_by_loss = sorted(random_selected_clients, key = lambda c: c['train_loss'], reverse = True) # sort by descending loss
        return {
            'selected_clients': clients_sorted_by_loss[:sample_size],
            'selection_algorithm': self.name,
            'messages': [],
        }

if __name__ == "__main__":
    poc = PowerOfChoice()
    print("PoC SELECTION TEST")
    print(poc.sample_clients(clients = [
        {'cid': 0, 'train_loss': 0.1, 'train_dataset_size': 1233},
        {'cid': 1, 'train_loss': 0.38, 'train_dataset_size': 323},
        {'cid': 2, 'train_loss': 0.13, 'train_dataset_size': 1133},
        {'cid': 3, 'train_loss': 0.48, 'train_dataset_size': 6312},
        {'cid': 4, 'train_loss': 0.56, 'train_dataset_size': 3451},
        {'cid': 5, 'train_loss': 0.95, 'train_dataset_size': 1245},
    ], sample_size = 3))
