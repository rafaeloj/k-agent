from selection_agent.selections import Random, Oort, RoundRobin, PowerOfChoice, RandomSelections
from selection_agent.agent.k_agent import KAgent

def get_selection_method(method: str, **kargs):
    if method == 'random':
        return Random()
    if method == 'oort':
        return Oort(
            num_clients=kargs['num_clients'],
            exploration_factor=0.3, # Fixo
            step_window = 5,
            pacer_step = 60,
            penalty = 2.0,
            cut_off = 0.95,
            blacklist_num = 1000,
            desired_duration = 1000000000,
        )
    if method == 'rrobin':
        return RoundRobin()
    if method == 'poc':
        return PowerOfChoice()
    if method == 'random-selections':
        return RandomSelections(num_clients=kargs['num_clients'])
    if method == 'k-agent':
        return KAgent(
            llm_model_name=kargs['llm_model_name'],
            k_agent_selection_method=kargs['k_agent_selection_method'],
            prompt_type = kargs['prompt_type'],
        )
    raise ValueError(f"Selection method not implemented: {method}")