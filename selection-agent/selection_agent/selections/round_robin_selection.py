from flwr.common import log
from logging import INFO
# from .selection import Selection

# class RoundRobin(Selection):
class RoundRobin:
    name: str = 'rrobin'
    def sample_clients(self, clients, sample_size, **kargs):
        log(INFO, "Round Robin Sample")
        if len(clients) < sample_size:
            return clients
        sorted_how_many_times_was_selected = sorted(clients, key = lambda c: c['how_many_times_was_selected'])
        selected_clients = sorted_how_many_times_was_selected[:sample_size]
        return {
            'selected_clients': selected_clients,
            'selection_algorithm': self.name,
            'messages': [],
        }
    
if __name__ == "__main__":
    poc = RoundRobin()
    print("TESTE RR SELECTION")
    print(poc.sample_clients(clients = [
        {'cid': 0, 'train_loss': 0.1, 'train_dataset_size': 1233, 'how_many_times_was_selected': 3},
        {'cid': 1, 'train_loss': 0.38, 'train_dataset_size': 323, 'how_many_times_was_selected': 5},
        {'cid': 2, 'train_loss': 0.13, 'train_dataset_size': 1133, 'how_many_times_was_selected': 1},
        {'cid': 3, 'train_loss': 0.48, 'train_dataset_size': 6312, 'how_many_times_was_selected': 0},
        {'cid': 4, 'train_loss': 0.56, 'train_dataset_size': 3451, 'how_many_times_was_selected': 7},
        {'cid': 5, 'train_loss': 0.95, 'train_dataset_size': 1245, 'how_many_times_was_selected': 1},
    ], sample_size = 3))