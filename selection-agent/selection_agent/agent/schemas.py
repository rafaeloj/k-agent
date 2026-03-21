from pydantic import BaseModel
from typing import List, Dict
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class ClientSchema(BaseModel):
    cid: str
    train_accuracy: float
    train_loss: float
    eval_accuracy: float
    eval_loss: float
    train_dataset_size: int
    eval_dataset_size: int
    how_many_times_was_selected: int
    utility: float


class RoundSummarySchema(BaseModel):
    selection_method: str
    selected_clients: List[ClientSchema]
    prev_global_loss: float
    prev_global_accuracy: float
    curr_global_loss: float
    curr_global_accuracy: float
    global_loss_difference: float
    global_accuracy_difference: float
    round_time: float

class AgentSchema(BaseModel):
    messages: Annotated[List[AnyMessage], operator.add]
    clients: List[ClientSchema]
    sample_size: int
    selection_algorithm: str
    selected_clients: List[ClientSchema]
    round_summaries: Dict[int, RoundSummarySchema]    
