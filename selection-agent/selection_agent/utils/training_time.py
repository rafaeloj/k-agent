import typing as T
import random

def get_cids_to_change_time(cids: T.List[str], perc: float) -> T.List[str]:
    k = int(perc*len(cids))
    return random.sample(cids, k=k)

def delay_training_time(curr_training_time: float, perc: float) -> float:
    return curr_training_time + (curr_training_time * perc)

def accelerate_training_time(curr_training_time: float, perc: float) -> float:
    return curr_training_time - (curr_training_time * perc)