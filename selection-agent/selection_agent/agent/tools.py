import typing as T
from numbers import Number
from pydantic import Field
import numpy as np
from langchain.tools import tool, ToolRuntime
from .schemas import ClientSchema
from langgraph.types import Command
from langchain.messages import ToolMessage
from ..selections import Oort
import json


@tool
def get_clients_stats(runtime: ToolRuntime) -> T.Dict[str, float|str]:
    """
        A tool used to get statistics of clients.
        Returns the mean and standard deviation of each client attribute
    """
    clients = runtime.state['clients']
    client_keys = clients[0].keys()
    metrics = {
        key: []
        for key in client_keys
    }    
    for client in clients:
        for key in client_keys:
            metrics[key].append(client[key])
    def _to_scalar(value):
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.size == 1 and isinstance(value.item(), Number):
                return float(value.item())
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], Number):
                return float(value[0])
            return None
        return None

    result = {}
    for key in client_keys:
        numeric_values = [
            scalar for scalar in ( _to_scalar(value) for value in metrics[key])
            if scalar is not None
        ]
        if numeric_values:
            result[f"avg_{key}"] = round(float(np.mean(numeric_values)), 2)
            result[f"std_{key}"] = round(float(np.std(numeric_values)), 2)
        else:
            result[f"avg_{key}"] = 'n/a'
            result[f"std_{key}"] = 'n/a'
    return result

@tool
def get_cids(runtime: ToolRuntime):
    """
        If you want to know whats clients are available this function.
        Args: None
        return [0,1,2,...n]
    """
    clients = runtime.state['clients']
    return [
        client['cid'] for client in clients
    ]

@tool
def get_info_by_cid(
    runtime: ToolRuntime,
    cid: str|int = Field(description= "It's a client ID as type string"),
)-> ClientSchema|None:
    """
        Get all information about a client by Client ID.
        args: cid='0'
        
        return {attr1: value1, attr2: value2...}
    """
    clients = runtime.state['clients']
    for c in clients:
        if str(c['cid']) == cid:
            return c
    return None

@tool
def get_client_attributes(runtime: ToolRuntime):
    """
        A tool used to get all attribute of clients. That attribute is used to filter clients 
    """
    clients = runtime.state['clients']
    return clients[0].keys()

@tool
def random_selection(
    k: int,
    runtime: ToolRuntime
) -> T.List[ClientSchema]:
    """Select k client randomly
    When to use Random Selection: Use as a baseline or when unbiased convergence is the priority. It selects clients uniformly (or proportional to data size) to guarantee the model converges to the true global optimum without solution bias, but it is slower and vulnerable to stragglers (slow clients).
    """
    clients = runtime.state['clients']
    k = runtime.state['sample_size'] if runtime.state['sample_size'] > 0 else k
    if len(clients) < k:
        return clients
    selected_clients = np.random.choice(clients, size=k, replace=False).tolist()
    tool_message = ToolMessage(
        content = str([client['cid'] for client in selected_clients]),
        tool_call_id = runtime.tool_call_id
    )
    return Command(
        update = {
            'messages': [tool_message],
            'selected_clients': selected_clients,
            'selection_algorithm': 'random',
        }
    )

@tool
def oort_selection(
    k: int,
    runtime: ToolRuntime,
    exploration_factor: float = 0.2,
) -> T.List[ClientSchema]:
    """Select k client using Oort algorithm
    When to use Oort: Use when the primary goal is to improve time-to-accuracy performance by simultaneously balancing the statistical utility of data and the client's system speed, or when it is necessary to enforce specific data distribution criteria during testing.
    args:
        k: is the total number of clients to select.
        exploration_factor: Define how many of the K clients will be in the exploratory set. Values ​​between 0 and 1, for example: exploration_factor=0.9. Default is 0.2
    """
    clients = runtime.state['clients']
    num_clients = len(clients)
    k = runtime.state['sample_size'] if runtime.state['sample_size'] > 0 else k
    if len(clients) < k:
        return clients
    
    # exploration_factor = 0.9
    step_window = 5
    pacer_step = 60
    penalty = 2.0
    cut_off = 0.95
    blacklist_num = 1000
    desired_duration = 1000000000
    
    oort = Oort(
        num_clients = num_clients,
        exploration_factor = exploration_factor,
        step_window = step_window,
        pacer_step = pacer_step,
        penalty = penalty, 
        cut_off = cut_off,
        blacklist_num = blacklist_num,
        desired_duration = desired_duration,
    )

    selected_clients = oort.sample_clients(clients, sample_size = k,)['selected_clients']


    tool_message = ToolMessage(
        content = str([client['cid'] for client in selected_clients]),
        tool_call_id = runtime.tool_call_id
    )
    # Client utility must be precomputed and then consumed by the selection algorithm.
    return Command(
        update = {
            'messages': [tool_message],
            'selected_clients': selected_clients,
            'selection_algorithm': 'oort',
        }
    )
@tool
def poc_selection(
    k: int,
    runtime: ToolRuntime
) -> T.List[ClientSchema]:
    """Select k client  using PoC algorithm
    When to use PoC (Power-of-Choice): Use when the focus is on accelerating the convergence rate. The algorithm intentionally selects clients with higher local loss, trading off a small amount of solution bias to train the model significantly faster.
    """
    clients = runtime.state['clients']
    k = runtime.state['sample_size'] if runtime.state['sample_size'] > 0 else k
    dataset_sizes = [client['train_dataset_size'] for client in clients]
    total_size = sum(dataset_sizes)
    weights = [size / total_size for size in dataset_sizes]
    if len(clients) < k:
        return clients
    random_selected_clients = np.random.choice(clients, p=weights, size=k, replace=False).tolist()
    # Loss-based selection
    clients_sorted_by_loss = sorted(random_selected_clients, key = lambda c: c['train_loss'], reverse = True) # sort by descending loss
    selected_clients = clients_sorted_by_loss[:k]

    tool_message = ToolMessage(
        content = str([client['cid'] for client in selected_clients]),
        tool_call_id = runtime.tool_call_id
    )
    return Command(
        update = {
            'messages': [tool_message],
            'selected_clients': selected_clients,
            'selection_algorithm': 'poc',
        }
    )

@tool
def tifl_selection(
    k: int,
    runtime: ToolRuntime
) -> T.List[ClientSchema]:
    """Select k client using TiFL algorithm
    When to use TiFL: Use when the major bottleneck is resource heterogeneity causing stragglers (slow clients that delay the round). It groups clients into "tiers" based on similar response latencies to ensure all selected clients finish training at approximately the same time.
    """
    clients = runtime.state['clients']
    k = runtime.state['sample_size'] if runtime.state['sample_size'] > 0 else k
    if len(clients) < k:
        return clients
    selected_clients = np.random.choice(clients, size=k, replace=False).tolist()
    tool_message = ToolMessage(
        content = str([client['cid'] for client in selected_clients]),
        tool_call_id = runtime.tool_call_id
    )
    return Command(
        update = {
            'messages': [tool_message],
            'selected_clients': selected_clients,
            'selection_algorithm': 'tifl',
        }
    )

@tool
def round_robin_selection(
    k: int,
    runtime: ToolRuntime
) -> T.List[ClientSchema]:
    """Select k client using Round Robin algorithm
    When to use Round Robin: Use when fairness (equal participation or resource usage) is the absolute priority. It ensures every client participates equally over time, but it ignores statistical and system utility, which significantly degrades training efficiency and time-to-accuracy.
    """
    clients = runtime.state['clients']
    k = runtime.state['sample_size'] if runtime.state['sample_size'] > 0 else k
    if len(clients) < k:
        return clients
    sorted_how_many_times_was_selected = sorted(clients, key = lambda c: c['how_many_times_was_selected'])
    selected_clients = sorted_how_many_times_was_selected[:k]

    tool_message = ToolMessage(
        content = str([client['cid'] for client in selected_clients]),
        tool_call_id = runtime.tool_call_id
    )
    return Command(
        update = {
            'messages': [tool_message],
            'selected_clients': selected_clients,
            'selection_algorithm': 'round_robin',
        }
    )

@tool
def get_round_summary(
    round: int,
    runtime: ToolRuntime,
):
    """
        Retrieves the performance summary and metrics for a specific round.
        
        Use this tool to evaluate the results of a previous round and determine if the 
        strategy used was successful or if the performance metrics improved.
    
    :param round: The integer index of the round to retrieve (e.g., 1, 2, ...).
    """
    round_summaries = runtime.state['round_summaries']
    return round_summaries[int(round-1)]
    

