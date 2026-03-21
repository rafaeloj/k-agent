import io
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
    logger,
)
from langchain_core.messages import messages_to_dict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info, sample_nodes
from typing import List
from .schemas import ClientSchema
from ..selections.oort_selection import calc_client_util
PROJECT_NAME = "FLOWER-advanced-pytorch"
import json
import typing as T
from selection_agent.selections.selection import Selection
import random

class FedAgent(FedAvg):
    def __init__(self, selection_method: Selection, selection_method_name: str, sample_size: int, llm_model_name: str, prompt_type: str, **kargs):
        super().__init__(**kargs)
        self.how_many_times_was_selected = {}
        self.federation_data = {}
        self.selection_method_name = selection_method_name
        self.selection_method = selection_method
        self.sample_size = sample_size
        self.llm_model_name = llm_model_name
        self.prompt_type = prompt_type

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
        clients: List[ClientSchema],
        total_rnd: int,
    ) -> Iterable[Message]:
        num_nodes = len(list(grid.get_node_ids()))
        s_time = time.process_time()
        node_ids, num_total = sample_nodes(grid, num_nodes, num_nodes) # All clients
        e_time = time.process_time()
        selection_algorithm = 'all-clients'
        messages = []
        if server_round > 1:
            log(INFO, "configure_train: Initialize selection with agent")
            s_time = time.process_time()
            sz = self.sample_size
            if self.sample_size == -2:
                sz = int(random.randint(2, len(clients)))
            log(INFO, f"K: {sz} SELECTING...")
            result = self.selection_method.sample_clients(clients, sz, curr_rnd=server_round, total_rnd=total_rnd)
            e_time = time.process_time()
            log(INFO, result)
            node_ids = [int(client['cid']) for client in result['selected_clients']]
            selection_algorithm = result['selection_algorithm']
            messages = messages_to_dict(result['messages'])
        self.federation_data[server_round]['sample_time'] = e_time - s_time
        self.federation_data[server_round]['selection_method'] = self.selection_method_name
        self.federation_data[server_round]['selection_algorithm'] = selection_algorithm
        self.federation_data[server_round]['prompt_type'] = self.prompt_type
        for cid in node_ids:
            if not cid in self.how_many_times_was_selected:
                self.how_many_times_was_selected[str(cid)] = 0
            self.how_many_times_was_selected[str(cid)] += 1

        self.federation_data[server_round]['messages'] = messages
        self.federation_data[server_round]['selected_clients'] = node_ids # Performance metrics are used to drive selection for the next round.
        # log(INFO, "configure_train: Sampled %s nodes (out of %s)", len(node_ids), len(num_total))
        config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        clients = {}
        self.federation_data = {}
        # TODO: run initial fit for all clients as round-0 registration.
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            self.federation_data[current_round] = {}
            self.federation_data[current_round]['llm_model_name'] = self.llm_model_name
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    server_round=current_round, # -1?
                    arrays=arrays,
                    config=train_config,
                    grid=grid,
                    clients=list(clients.values()),
                    total_rnd = num_rounds+1,
                ),
                timeout=timeout,
            )
            # Aggregate train
            # Update the client-utility mapping.
            # Comm_round_time is training time plus communication time.
            s_time = time.process_time()
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )                    
            e_time = time.process_time()
            training_time = e_time - s_time

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\tclients└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )
            clients = self.create_client_schema(
                clients = clients,
                train_replies = train_replies,
                evaluate_replies = evaluate_replies,
                server_round =  current_round
            )
            self.federation_data[current_round]['clients'] = list(clients.values())
            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            if 'agent' in self.selection_method_name:
                # log(INFO, "SUMMARI")
                # log(INFO, self.selection_method.round_summaries)
                prev_acc = self.selection_method.round_summaries[current_round-1]['prev_global_accuracy'] if current_round > 1 else 0
                prev_loss = self.selection_method.round_summaries[current_round-1]['prev_global_loss'] if current_round > 1 else 0
                diff_accuracy = prev_acc - agg_train_metrics['train_accuracy'] if 'train_accuracy' in agg_train_metrics else prev_acc
                diff_loss = prev_loss - agg_train_metrics['train_loss'] if 'train_loss' in agg_train_metrics else prev_loss
                self.selection_method.round_summaries[current_round] = {
                    "selection_method": self.federation_data[current_round]['selection_algorithm'],
                    "selected_clients": self.federation_data[current_round]['selected_clients'],
                    "prev_global_loss": prev_loss,
                    "prev_global_accuracy": prev_acc,
                    "curr_global_loss": agg_train_metrics['train_loss'],
                    "curr_global_accuracy": agg_train_metrics['train_accuracy'],
                    "global_loss_difference": diff_accuracy,
                    "global_accuracy_difference": diff_loss,
                    "round_time": training_time,
                }


            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Maybe save to disk if new best is found
                    self._update_best_acc(current_round, res["accuracy"], arrays)
        self.save_federation_data()
        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")
                
        return result

    def save_federation_data(self):
        random_id = random.randint(0,10)
        timestamp = time.strftime("%H_%M_%d_%m")
        with open(f"./outputs/timestamp_{timestamp}_method_{self.selection_method_name}_sample_size_{self.sample_size}_model_name_{self.llm_model_name}_federation_data_{random_id}.json", 'w') as f:
            json.dump(self.federation_data, f, indent=4)

    def create_client_schema(self, clients: List[ClientSchema], train_replies: List[Message], evaluate_replies: List[Message], server_round: int) -> List[ClientSchema]:
        # TODO: Optimize performance.
        c = {
            str(msg.content['metrics']['cid']): {
                "train_accuracy": round(msg.content['metrics']['train_accuracy'], 2),
                "train_loss": round(msg.content['metrics']['train_loss'], 2),
                "train_dataset_size": msg.content['metrics']['num-examples'],
                "stat_util": round(msg.content['metrics']['stat_util'], 3),
                "training_time": msg.content['metrics']['training_time'],
            } for msg in train_replies
        }
        for msg in evaluate_replies:
            client = msg.content['metrics']
            cid = str(client['cid'])
            if not cid in c:
                c[cid] = {
                    "train_accuracy": 0,
                    "train_loss": clients[cid]['train_loss'],
                    "train_dataset_size": client['num-examples'],
                    "stat_util": 0,
                    "training_time": clients[cid]['training_time'],
                }
            c[cid]["prev_train_accuracy"] = 0 if server_round == 1 else clients[cid]['train_accuracy']
            c[cid]["prev_train_loss"] = 0 if server_round == 1 else clients[cid]['train_loss']
            c[cid]["prev_eval_loss"] = 0 if server_round == 1 else clients[cid]['eval_loss']
            c[cid]["prev_eval_accuracy"] = 0 if server_round == 1 else clients[cid]['eval_accuracy']
            c[cid]['last_round'] = server_round
            c[cid]['eval_loss'] = round(client['eval_loss'], 2)
            c[cid]['eval_accuracy'] = round(client['eval_accuracy'], 2)
            c[cid]["eval_dataset_size"] = client['num-examples']
            c[cid]['cid'] = cid
            c[cid]['how_many_times_was_selected'] = self.how_many_times_was_selected[cid]
        
        for cid in c:            
            c[cid] = calc_client_util(
                c[cid],
                c[cid]['stat_util'],
                server_round,
                desired_duration=1000,
            )

        # log(INFO, c.keys())
        return c