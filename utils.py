import os
import random
import numpy as np
import torch
import sys
from transformers import pipeline
import os
from torch.utils.data import Dataset
from torch_geometric.nn import GINEConv, GCNConv, SAGEConv, global_add_pool, global_max_pool, GATConv
from openai import OpenAI
from pydantic import BaseModel
import dotenv
import pickle
from transformers import BitsAndBytesConfig

dotenv.load_dotenv()

def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def agent(task, properties, num_concepts):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if task == 'dili':
        task = 'Predict the DILi (Drug-Induced Liver Injury) risk of a molecule.'
    elif task == 'bbbp':
        task = 'Predict the BBBP (Blood-Brain Barrier Penetration) risk of a molecule.'
    elif task == 'lipo':
        task = 'Predict the LIPO (Lipophilicity) of a molecule.'
    elif task == 'avail':
        task = 'Predict the bioavailability of a molecule.'
    elif task == 'solubility':
        task = 'Predict a molecules water solubility.'
    elif task == 'caco':
        task = 'Predict a molecules cell effective permeability.'
    elif task == 'hia_hou':
        task = 'Predict a molecules human intestinal absorption.'
    elif task == 'pgp':
        task = 'Predict a molecules (P-glycoprotein) Inhibition.'
    elif task == 'ppbr':
        task = 'Predict a molecules plasma protein binding rate.'
    elif task == 'vdss':
        task = 'Predict a molecules volume of distribution at steady state.'
    elif task == 'cyp2c9':
        task = 'Predict a molecules CYP2C9 inhibition.'
    elif task == 'cyp2d6':
        task = 'Predict a molecules CYP2D6 inhibition.'
    elif task == 'cyp3a4':
        task = 'Predict a molecules CYP3A4 inhibition.'
    elif task == 'cyp2c9_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP2C9).'
    elif task == 'cyp2d6_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP2D6).'
    elif task == 'cyp3a4_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP3A4).'
    elif task == 'half_life':
        task = 'Predict a molecules half-life.'
    elif task == 'ld50':
        task = 'Predict a molecules acute toxicity.'
    elif task == 'herg':
        task = 'Predict whether a molecule is a hERG blocker.'
    elif task == 'ames':
        task = 'Predict whether a molecule is an AMES mutagen.'

    response = client.responses.create(
        model = 'gpt-4o',
        instructions = 'You are a helpful chemistry assistant. You will be given a task and some properties to choose from. You will choose the properties most important for successfully completing/predicting the task.',
        input = f'Task: {task}\nProperties: {properties}. You should choose {num_concepts} properties and return them in a list formatted exactly as they are given to you. The list should be your ONLY output.',
    )
    return response.output[0].content[0].text

def llama_agent(task, properties, num_concepts):
    if task == 'dili':
        task = 'Predict the DILi (Drug-Induced Liver Injury) risk of a molecule.'
    elif task == 'bbbp':
        task = 'Predict the BBBP (Blood-Brain Barrier Penetration) risk of a molecule.'
    elif task == 'lipo':
        task = 'Predict the LIPO (Lipophilicity) of a molecule.'
    elif task == 'avail':
        task = 'Predict the bioavailability of a molecule.'
    elif task == 'solubility':
        task = 'Predict a molecules water solubility.'
    elif task == 'caco':
        task = 'Predict a molecules cell effective permeability.'
    elif task == 'hia_hou':
        task = 'Predict a molecules human intestinal absorption.'
    elif task == 'pgp':
        task = 'Predict a molecules (P-glycoprotein) Inhibition.'
    elif task == 'ppbr':
        task = 'Predict a molecules plasma protein binding rate.'
    elif task == 'vdss':
        task = 'Predict a molecules volume of distribution at steady state.'
    elif task == 'cyp2c9':
        task = 'Predict a molecules CYP2C9 inhibition.'
    elif task == 'cyp2d6':
        task = 'Predict a molecules CYP2D6 inhibition.'
    elif task == 'cyp3a4':
        task = 'Predict a molecules CYP3A4 inhibition.'
    elif task == 'cyp2c9_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP2C9).'
    elif task == 'cyp2d6_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP2D6).'
    elif task == 'cyp3a4_substrate':
        task = 'Predict if a molecule substrates to the enzyme (CYP3A4).'
    elif task == 'half_life':
        task = 'Predict a molecules half-life.'
    elif task == 'ld50':
        task = 'Predict a molecules acute toxicity.'
    elif task == 'herg':
        task = 'Predict whether a molecule is a hERG blocker.'
    elif task == 'ames':
        task = 'Predict whether a molecule is an AMES mutagen.'

    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B-Instruct", device='cuda', quantization_config=bnb_cfg)
    print('pipe loaded')
    messages = [
        {"role": "system", "content": "You are a helpful chemistry assistant. You will be given a task and some properties to choose from. You will choose the properties most important for successfully completing/predicting the task."},
        {"role": "user", "content": f'Task: {task}\nProperties: {properties}. You should choose {num_concepts} properties and return them in a list formatted exactly as they are given to you. The list should be your ONLY output.'},
    ]
    output = pipe(messages)

    print(output.text)

# create the dataset for the vanilla model
class MyDataset(Dataset):
    def __init__(self, split, features, means, stds, tokenizer, DATA):
        self.data = DATA[split]
        self.features = features
        self.means = means
        self.stds = stds
        self.stds[self.stds == 0] = 1
        self.tokenizer = tokenizer
        self.labels = self.data['Y']
        self.text = self.data["Drug"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        tokenized_text = self.tokenizer(
            self.text[index], 
            max_length=64,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
        )

        concept_labels = (self.data.iloc[index][self.features].values.astype(float) - self.means) / self.stds

        return_dict = {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'label': torch.tensor(self.labels[index], dtype=torch.float),
            'concept_labels': concept_labels,
            'features': self.features
        }

        return return_dict

# define the GNN embedding model
class MolNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MolNet, self).__init__()
                                
        # 3 is the number of edge features from the 'from_smiles' function
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=3)
        self.conv2 = GINEConv(nn2, edge_dim=3)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = self.conv2(x.to(torch.float32), edge_index, edge_attr.to(torch.float32)).relu()
        x = global_add_pool(x, batch)
        return x