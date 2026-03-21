from flwr.common import (
    log,
)
import numpy as np
import math
from logging import INFO
import random
# from selection_agent.selections.selection import Selection

# class Oort(Selection):
class Oort:
    name: str = 'oort'
    def __init__(self,
            num_clients: int,
            exploration_factor: float,
            step_window: int,
            pacer_step: float,
            penalty: float,
            cut_off: float,
            blacklist_num: float,
            desired_duration: float,
        ):
        self.exploration_factor = exploration_factor
        self.step_window = step_window
        self.pacer_step = pacer_step
        self.penalty = penalty
        self.cut_off = cut_off
        self.blacklist_num = blacklist_num
        tmp_desired_duration = desired_duration
        self.desired_duration = np.inf if tmp_desired_duration > 0 else tmp_desired_duration
        self.blacklist = []
        self.explored_clients = []
        self.num_clients = num_clients
        self.unexplored_clients = None
        self.util_history = []
    
    def _create_clients(self, clients):
        return {
            str(client['cid']): client
            for client in clients
        }
    
    def sample_clients(self, clients, sample_size, **kargs):
        if not self.unexplored_clients:
            self.unexplored_clients = [client['cid'] for client in clients]
            
        available_cids = [
            (client['utility'], client['cid'])
            for client in clients
        ]
        client_dict = self._create_clients(clients = clients)
        selected_clients_ids = []

        # Exploitation
        exploited_clients_count = max(
            math.ceil((1.0 - self.exploration_factor) * sample_size),
            sample_size - len(self.unexplored_clients),
        )

        available_cids.sort(key=lambda x: x[0], reverse=True)

        sorted_by_utility = [cid for _, cid in available_cids]

        # Calculate cut-off utility
        cut_off_util = (
                client_dict[sorted_by_utility[exploited_clients_count - 1]]['utility'] * self.cut_off
        )

        # Include clients with utilities higher than the cut-off
        exploited_clients = []
        for client_id in sorted_by_utility:
            if (
                    client_dict[client_id]['utility'] > cut_off_util
                    and client_id not in self.blacklist
            ):
                exploited_clients.append(client_id)

        # Sample clients with their utilities
        total_utility = float(
            sum(client_dict[client_id]['utility'] for client_id in exploited_clients)
        )

        probabilities = [
            client_dict[client_id]['utility'] / total_utility
            for client_id in exploited_clients
        ]

        if len(probabilities) > 0 and exploited_clients_count > 0:
            selected_clients_ids = np.random.choice(
                exploited_clients,
                min(len(exploited_clients), exploited_clients_count),
                p=probabilities,
                replace=False,
            )
            selected_clients_ids = selected_clients_ids.tolist()

        last_index = (
            sorted_by_utility.index(exploited_clients[-1])
            if exploited_clients
            else 0
        )

        # If the result of exploitation wasn't enough to meet the required length
        if len(selected_clients_ids) < exploited_clients_count:
            for index in range(last_index + 1, len(sorted_by_utility)):
                if (
                        not sorted_by_utility[index] in self.blacklist
                        and len(selected_clients_ids) < exploited_clients_count
                ):
                    selected_clients_ids.append(sorted_by_utility[index])

        # Exploration
        # Select unexplored clients randomly
        unexplored_size = len(self.unexplored_clients)
        if unexplored_size > 0:
            selected_unexplore_clients = random.sample(
                self.unexplored_clients, min(unexplored_size, sample_size - len(selected_clients_ids))
            )
        else:
            selected_unexplore_clients = []

        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients_ids += selected_unexplore_clients

        # log(INFO, selected_clients_ids)
        selected_clients = []
        for client in selected_clients_ids:
            client_dict[client]['how_many_times_was_selected'] += 1
            selected_clients.append(client_dict[client])
        return {
            'selected_clients': selected_clients,
            'selection_algorithm': self.name,
            'messages': [],
        }

def calc_client_util(client, statistical_utility, server_round, desired_duration=100, penalty=2):
    """Calculate the client utility."""
    # Explored client
    if client['last_round'] == 0:
        client['last_round'] = server_round

    client_utility = statistical_utility + math.sqrt(
        0.1 * math.log(server_round) / client['last_round']
    )

    if desired_duration < client['training_time']:
        global_utility = (desired_duration / client['training_time']) ** penalty
        client_utility *= global_utility

    # Update exploited client
    client['last_round'] = server_round
    client['utility'] = client_utility
    return client


if __name__ == "__main__":
    oort = Oort(
        num_clients = 6,
        exploration_factor = 0.9,
        step_window = 5,
        pacer_step = 60,
        penalty = 2.0,
        cut_off = 0.95,
        blacklist_num = 1000,
        desired_duration = 10000,
    )
    print("TESTE OORT SELECTION")
    print(oort.sample_clients(clients = [
        {'cid': '0', 'utility': 0.1, 'times_selected': 0, 'how_many_times_was_selected': 0},
        {'cid': '1', 'utility': 0.38, 'times_selected': 0, 'how_many_times_was_selected': 0},
        {'cid': '2', 'utility': 0.13, 'times_selected': 0, 'how_many_times_was_selected': 0},
        {'cid': '3', 'utility': 0.48, 'times_selected': 0, 'how_many_times_was_selected': 0},
        {'cid': '4', 'utility': 0.56, 'times_selected': 0, 'how_many_times_was_selected': 0},
        {'cid': '5', 'utility': 0.95, 'times_selected': 0, 'how_many_times_was_selected': 0},
    ], sample_size = 3))
