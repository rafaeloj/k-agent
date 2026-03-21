"""selection-agent: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from logging import INFO
import time
from selection_agent.utils.getters import get_dataset, get_model
from flwr.common import (
    ArrayRecord,
    Message,
    MetricRecord,
    RecordDict,
)
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    partition_id: int = context.node_config["partition-id"]
    num_partitions: int = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    model_name: str = context.run_config['model-name']
    dataset_name: str = context.run_config['dataset-name']
    noniid: bool = context.run_config['noniid']
    # lstm
    vocab_size = context.run_config['vocab-size']
    embedding_dim = context.run_config['embedding-dim']
    hidden_dim = context.run_config['hidden-dim']
    num_layers = context.run_config['num-layers']
    seq_len: int = context.run_config['seq-len']
    batch_size: int = context.run_config['batch-size']

    # ---- CREATING MODEL ---- #
    # Load the model and initialize it with the received weights
    model, train_fn, _ = get_model(
        model_name = model_name,
        vocab_size=vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- LOADING DATASET ---- #
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = get_dataset(
        partition_id = partition_id,
        n_partitions = num_partitions,
        dataset_name = dataset_name,
        seq_len = seq_len,
        batch_size = batch_size,
        noniid = noniid
    )

    # ---- TRAINING MODEL ---- #
    start_time = time.process_time()
    train_loss, train_accuracy, stat_util = train_fn(
        model = model,
        trainloader = trainloader,
        epochs = context.run_config["local-epochs"],
        lr = msg.content["config"]["lr"],
        device = device,
    )
    end_time = time.process_time()

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "cid": msg.metadata.dst_node_id,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "num-examples": len(trainloader.dataset),
        "stat_util": stat_util,
        "training_time": end_time - start_time
        # "num-examples": len(llm_client.trainset.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model_name = context.run_config['model-name']
    dataset_name = context.run_config['dataset-name']
    batch_size = context.run_config['batch-size']
     # lstm
    seq_len = context.run_config['seq-len']
    vocab_size = context.run_config['vocab-size']
    embedding_dim = context.run_config['embedding-dim']
    hidden_dim = context.run_config['hidden-dim']
    num_layers = context.run_config['num-layers']
    noniid = context.run_config['noniid']

    # ---- CREATING MODEL ---- #
    # Load the model and initialize it with the received weights
    model, _, test_fn = get_model(
        model_name = model_name,
        vocab_size=vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- LOADING DATASET ---- #
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, evalloader = get_dataset(
        partition_id = partition_id,
        n_partitions = num_partitions,
        dataset_name = dataset_name,
        batch_size = batch_size,
        seq_len = seq_len,
        noniid = noniid,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model = model,
        testloader = evalloader,
        device = device,
    )
    
    metrics = {
        "cid": msg.metadata.dst_node_id,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "num-examples": len(evalloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
