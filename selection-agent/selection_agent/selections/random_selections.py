from random import choice
from flwr.common import log
from logging import INFO

class RandomSelections:
    name: str = 'random-selections'
    def __init__(self, **kargs):
        # Lazy import to avoid circular/dependency-order issues.
        from .oort_selection import Oort
        from .power_of_choice_selection import PowerOfChoice
        from .random_selection import Random
        from .round_robin_selection import RoundRobin
        self.selections = [
            Oort(
                num_clients=kargs['num_clients'],
                exploration_factor=0.3, # Fixed
                step_window = 5,
                pacer_step = 60,
                penalty = 2.0,
                cut_off = 0.95,
                blacklist_num = 1000,
                desired_duration = 1000000000,
            ),
            PowerOfChoice(),
            Random(),
            RoundRobin(),
        ]
    def sample_clients(self, clients, sample_clients: int, **kargs):
        sel = choice(self.selections)
        log(INFO, f"Random Selections: ({sel.name})")
        return sel.sample_clients(clients, sample_clients)