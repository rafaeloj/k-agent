from flwr.common import log
from logging import INFO
import numpy as np
# from selection_agent.selections.selection import Selection

# class Random(Selection):
class Random:
    name:str = "random"
    def sample_clients(self, clients, sample_size, **kargs):
        log(INFO, "Random Sample")
        if len(clients) < sample_size:
            return clients
        selected_clients = np.random.choice(clients, size=sample_size, replace=False).tolist()
        
        return {
            'selected_clients': selected_clients,
            'selection_algorithm': self.name,
            'messages': [],
        }

if __name__ == "__main__":
    poc = Random()
    print("TESTE PoC SELECTION")
    print(poc.sample_clients(clients = [
        {'cid': 0, 'train_loss': 0.1, 'train_dataset_size': 1233},
        {'cid': 1, 'train_loss': 0.38, 'train_dataset_size': 323},
        {'cid': 2, 'train_loss': 0.13, 'train_dataset_size': 1133},
        {'cid': 3, 'train_loss': 0.48, 'train_dataset_size': 6312},
        {'cid': 4, 'train_loss': 0.56, 'train_dataset_size': 3451},
        {'cid': 5, 'train_loss': 0.95, 'train_dataset_size': 1245},
    ], sample_size = 3))