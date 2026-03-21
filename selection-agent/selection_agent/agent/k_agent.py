from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from .prompts import K_SYSTEM_PROMPT, K_SYSTEM_PROMPT_COT, K_SYSTEM_PROMPT_FS
from .schemas import AgentSchema, ClientSchema
from typing import List
from flwr.common import log
from logging import INFO
import time

class KAgent:
    name:str = 'agent-selection'
    def __init__(self, llm_model_name: str, k_agent_selection_method: List[str], prompt_type:str):
        self.round_summaries = {}
        #lazy import porque ta foda...
        from .tools import (
            get_clients_stats,
            get_cids,
            get_info_by_cid,
            get_round_summary,
        )
        self.sel_name = k_agent_selection_method
        self.prompt_type = prompt_type

        llm = ChatOllama(
            model = llm_model_name,
            temperature=0,
            num_predict=1024,
            num_ctx=8192,
            keep_alive="5m",
        ) 
        self.agent = create_agent(
            model = llm,
            tools = [
                get_clients_stats,
                get_cids,
                get_info_by_cid,
                get_round_summary,
                self.get_selection_method(k_agent_selection_method)
            ],
            state_schema=AgentSchema,
            system_prompt=self.get_prompt_type(prompt_type),
        )
    def get_prompt_type(self, prompt_type: str):
        if prompt_type == 'chain-of-thought':
            return K_SYSTEM_PROMPT_COT
        if prompt_type == 'few-shot':
            return K_SYSTEM_PROMPT_FS
        if prompt_type == 'description-only':
            return K_SYSTEM_PROMPT

    def get_selection_method(self, selection: str):
        from .tools import (
            random_selection,
            oort_selection,
            poc_selection,
            round_robin_selection,
        )
        selection_dict = {
            "random_selection": random_selection,
            "oort_selection": oort_selection,
            "poc_selection": poc_selection,
            "round_robin_selection": round_robin_selection,
        }
        return selection_dict[selection]
            
    def sample_clients(self, clients: List[ClientSchema], sample_clients: int, **kargs):
        result = None
        fail = True
        tries = 0
        current_round = kargs['curr_rnd']
        total_rounds = kargs['total_rnd']

        instruction_text = (
            f"We are in round {current_round} of {total_rounds}. "
            f"N={len(clients)}."
            f"Define K and call the selection tool ({self.sel_name})."
        )

        message_history = [HumanMessage(content=instruction_text)]

        while fail:
            log(INFO, f"Selection tries (K-Agent): {tries + 1}")
            
            try:
                result = self.agent.invoke({
                    "messages": message_history,
                    "clients": clients,
                    "sample_size": sample_clients,
                    "round_summaries": self.round_summaries
                })

                if 'selected_clients' in result:
                    fail = False
                    log(INFO, "Agent selected clients successfully.")
                    return result                
                else:
                    if 'messages' in result:
                        message_history = result['messages']

                    log(INFO, "Valid response, but selection tool not called.")
                    log(INFO, result)

                    correction_msg = HumanMessage(
                        content="Your answer, but not call the selection tool."
                                "Use the knowledge to call the selection tool"
                    )
                    message_history.append(correction_msg)

            except Exception as e:
                log(INFO, f"Selection Agent Error Type: {type(e)}")
                log(INFO, e)
                log(INFO, "-"*40)
                error_feedback = (
                    f"System error: {str(e)}. \n"
                    "Your previosly answer cause error"
                    "Try again, check how you call the tool and fix it"
                )
                
                message_history.append(ToolMessage(content=error_feedback, status='error'))
            tries += 1
            
        return result