import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from selection_agent.agent.selection_agent_strategy import FedAgent
from selection_agent.utils.getters import get_model
import typing as T
from selection_agent.selections import get_selection_method
import time

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    model_name: str = context.run_config['model-name']
    num_clients: int = context.run_config['num-supernodes']
    sample_size: int = int(context.run_config['sample-size'])
    selection_method: str = context.run_config['selection-method']
    # agent
    prompt_type: str = context.run_config['prompt-type']
    llm_model_name: str = context.run_config['llm-model-name']
    #k-agent
    k_agent_selection_method: str = context.run_config['k-agent-selection-method']
    # lstm
    vocab_size: int = context.run_config['vocab-size']
    embedding_dim: int = context.run_config['embedding-dim']
    hidden_dim: int = context.run_config['hidden-dim']
    num_layers: int = context.run_config['num-layers']

    # Load global model
    global_model, _, _ = get_model(
        model_name = model_name,
        vocab_size=vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        num_layers=num_layers,
    )
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAgent(
        fraction_train = fraction_train,
        selection_method_name = selection_method,
        selection_method = get_selection_method(
            method = selection_method,
            llm_model_name = llm_model_name,
            num_clients = num_clients,
            sample_size = sample_size,
            k_agent_selection_method = k_agent_selection_method,
            prompt_type = prompt_type,
        ),
        sample_size = sample_size,
        llm_model_name = llm_model_name,
        prompt_type = prompt_type,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    timestamp = time.strftime("%H_%M_%d_%m")

    torch.save(state_dict, f"models/{timestamp}_{selection_method}_{num_clients}_{sample_size}_final_model_.pt")
