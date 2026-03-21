import json
import pandas as pd
import seaborn as sns
import glob

def json_to_df(json_data):
    data = []
    for i, (r, round) in enumerate(json_data.items()):
        r = int(r)
        clients = round['clients']
        selected_clients = round['selected_clients']
        selection_method = round['selection_method']
        selection_algorithm = round['selection_algorithm']
        delay = round['delay']
        for client in clients:
            client['round'] = r
            client['selection_method'] = selection_method
            client['selection_algorithm'] = selection_algorithm
            client['selected'] = True if int(client['cid']) in selected_clients else False
            client['delay'] = delay
            client['llm_model_name'] = round['llm_model_name'] if 'llm_model_name' in round else 'None'
            client['run_id'] = i
            client['sample_time'] = round['sample_time'] if 'sample_time' in round else None
            data.append(client)
    return pd.DataFrame(data)

def read_jsons(folder_path):
    pattners = '.json'
    files = glob.glob(f'{folder_path}/*{pattners}')
    jsons = []
    for file in files:
        with open(file) as f:
            js = json.load(f)
        jsons.append(js)
    return jsons

def inject_prompt_type(df:pd.DataFrame, prompt_type):
    if 'prompt_type' in df.columns:
        return df
    #'cot', 'description', 'fewshot
    if prompt_type == 'cot':
        df['prompt_type'] = 'chain-of-thought'
    elif prompt_type == 'description':
        df['prompt_type'] = 'description-only'
    elif prompt_type == 'fewshot':
        df['prompt_type'] = 'few-shot'
    return df