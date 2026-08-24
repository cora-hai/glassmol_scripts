from torch_geometric.loader import DataLoader
import pandas as pd
import torch
from torch_geometric.utils.smiles import from_smiles
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
from utils import set_seed, agent
from backbone.backbone import ModelXtoCtoY_function
import ast
import pickle as pkl
from utils import MolNet
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('args.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

set_seed(config['seed'])
data_type = config['data_type']
num_epochs = config['num_epochs']
num_concepts = config['num_concepts']
loss_weight = config['loss_weight']

# molecules are PyG objects, so we need to attach the y and concepts to the object
def attach_y_and_concepts(row):
    row['Drug'].y = torch.tensor(row['Y'], dtype=torch.float32)
    row['Drug'].target_names = row['Drug_ID']
    try:
        row['Drug'].concepts = torch.tensor(np.asarray(row[features].values, dtype=np.float32))
    except:
        pass
    return row['Drug']

# load data
train_data = pd.read_csv(f"data/train_{data_type}.csv")
test_data = pd.read_csv(f"data/test_{data_type}.csv")
val_data = pd.read_csv(f"data/val_{data_type}.csv")

# choose num_concepts features with llm agent
features = agent(data_type, train_data.drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
features = ast.literal_eval(features)
print(f"selected concepts = {features}")

with open(f'model_output_dir/features_gnn_{data_type}.pkl', 'wb') as f:
    pkl.dump(features, f)

# turn the SMILES strings into PyG objects
train_data['Drug'] = train_data['Drug'].apply(from_smiles)
test_data['Drug'] = test_data['Drug'].apply(from_smiles)
val_data['Drug'] = val_data['Drug'].apply(from_smiles)  

# attach the y and concepts to the PyG objects
train_data['Drug'] = train_data.apply(attach_y_and_concepts, axis=1)
test_data['Drug'] = test_data.apply(attach_y_and_concepts, axis=1)
val_data['Drug'] = val_data.apply(attach_y_and_concepts, axis=1)

# create the data loaders
train_loader = DataLoader(train_data['Drug'], batch_size=32, shuffle=True)
val_loader = DataLoader(val_data['Drug'], batch_size=32, shuffle=False)
test_loader = DataLoader(test_data['Drug'], batch_size=32, shuffle=False)

# initialize the model, optimizer, and loss functions
ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0).to(device)
model = MolNet(in_channels=train_data['Drug'][1].x.shape[1], hidden_channels=768).to(device)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(ModelXtoCtoY_layer.parameters()), lr=2e-4)
loss_C = torch.nn.L1Loss().to(device)
loss_Y = torch.nn.BCEWithLogitsLoss().to(device)

best_acc_score = 0
for epoch in range(num_epochs):
    ######### train #########
    model.train()
    for data in train_loader:
        data = data.to(device)
        #print(f"data: {data.shape}")
        #print(f"concepts: {data.concepts.shape}")
        #print(f"squeezed concepts: {data.concepts.squeeze().shape}")
        optimizer.zero_grad()
        output = model(data)
        #print(f"output: {output.shape}")
        outputs = ModelXtoCtoY_layer(output)
        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]

        # XtoC_loss
        XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
        XtoC_loss = loss_C(torch.flatten(XtoC_output), data.concepts.squeeze())
        
        # XtoY_loss
        XtoY_loss = loss_Y(XtoY_output[0].squeeze(), data.y.squeeze())
        
        loss = XtoY_loss + XtoC_loss * loss_weight
        loss.backward()
        optimizer.step()

    ######### val #########
    model.eval()
    ModelXtoCtoY_layer.eval()

    val_accuracy = 0.
    predictions = np.array([])
    true_labels = np.array([])

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            outputs = ModelXtoCtoY_layer(output)
            XtoC_output = outputs[1:] 
            XtoY_output = outputs[0:1]

            true_labels = np.append(true_labels, batch.y.cpu().numpy())

            predictions = np.append(predictions, (XtoY_output[0].squeeze().cpu() > 0.5) == batch.y.squeeze().cpu())

    val_accuracy = predictions.sum() / len(predictions)
        
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, f'model_output_dir/model_gnn_{data_type}.pth')
        torch.save(ModelXtoCtoY_layer, f'model_output_dir/ModelXtoCtoY_layer_gnn_{data_type}.pth')


######### test #########
model = torch.load(f'model_output_dir/model_gnn_{data_type}.pth', weights_only=False)
ModelXtoCtoY_layer = torch.load(f'model_output_dir/ModelXtoCtoY_layer_gnn_{data_type}.pth', weights_only=False)
with torch.no_grad():
    model.eval()
    ModelXtoCtoY_layer.eval()
    predictions = np.array([])
    true_labels = np.array([])
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        outputs = ModelXtoCtoY_layer(output)
        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]

        predictions = np.append(predictions, XtoY_output[0].squeeze().cpu().numpy())
        true_labels = np.append(true_labels, data.y.squeeze().cpu().numpy())

print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')

with open(f'model_output_dir/test_loader_gnn_{data_type}.pkl', 'wb') as f:
    pkl.dump(test_loader, f)
