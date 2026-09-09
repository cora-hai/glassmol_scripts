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

import argparse

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# molecules are PyG objects, so we need to attach the y and concepts to the object
def attach_y_and_concepts(row, features):
    row['Drug'].y = torch.tensor(row['Y'], dtype=torch.float32)
    row['Drug'].target_names = row['Drug_ID']
    try:
        row['Drug'].concepts = torch.tensor(np.asarray(row[features].values, dtype=np.float32))
    except:
        pass
    return row['Drug']

def main(in_data_folder, model_folder, data_type, num_epochs, num_concepts, loss_weight, concept_selector):

    # load data
    DATA = {}
    DATA["train"] = pd.read_csv(f"{in_data_folder}/train_{data_type}.csv")
    DATA["val"] = pd.read_csv(f"{in_data_folder}/test_{data_type}.csv")
    DATA["test"] = pd.read_csv(f"{in_data_folder}/val_{data_type}.csv")

    # choose num_concepts features with llm agent
    if concept_selector == "llm":

        # introduce while loop to automatically skip invalid concept selections
        found_valid = False

        while found_valid == False:
            # original concept selector from GlassMol paper
            features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts).replace("```python", "").replace("```", "")
            features = ast.literal_eval(features)

            # check if all concepts selected by GPT are valid
            try: 
                valid_test = DATA['train'][features]

            except KeyError:
                continue

            # check if GPT returned the correct number of concepts
            if len(features) != 40:
                continue

            found_valid = True

        print(features)


    elif concept_selector == "l1":
        # according to https://scikit-learn.org/stable/modules/feature_selection.html#l1-based-feature-selection
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]

        # train linear support vector classifier with L1 penalty for "feature selection"
        lsvc = LinearSVC(C=0.01, penalty = "l1", dual = False).fit(X,y)     # C = regularisation parameter, strength inversely proportional to C
        selector = SelectFromModel(lsvc, prefit = True)
    
        # get selected features
        feature_mask = selector.get_support()
        features = X.columns[feature_mask].tolist()
        num_concepts = len(features)
    
        print(features)

    elif concept_selector == "tree":
        # according to https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        print(f"X = {X.shape}")

        # maybe max_features nutzen für feste Anzahl an concepts?
        clf = ExtraTreesClassifier(n_estimators = 30, random_state = 42).fit(X,y)  # n_estimators = number of trees in the forest
        selector = SelectFromModel(clf, prefit = True)
        feature_mask = selector.get_support()
        features = X.columns[feature_mask].tolist()
        print(features)
        num_concepts = len(features)


    elif concept_selector == "late-l1":
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        features = X.columns.to_list()
        num_concepts = len(features)

    elif concept_selector == "no":
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        features = X.columns.to_list()
        num_concepts = len(features)

    else:
        print("choose a valid concept selector method")


    with open(f'{model_folder}/features_gnn_{data_type}_{concept_selector}.pkl', 'wb') as f:
        pkl.dump(features, f)

    # turn the SMILES strings into PyG objects
    DATA["train"]['Drug'] = DATA["train"]['Drug'].apply(from_smiles)
    DATA["val"]['Drug'] = DATA["val"]['Drug'].apply(from_smiles)
    DATA["test"]['Drug'] = DATA["test"]['Drug'].apply(from_smiles)  

    # attach the y and concepts to the PyG objects
    DATA["train"]['Drug'] = DATA["train"].apply(attach_y_and_concepts, args = (features,), axis=1)
    DATA["val"]['Drug'] = DATA["val"].apply(attach_y_and_concepts, args = (features,), axis=1)
    DATA["test"]['Drug'] = DATA["test"].apply(attach_y_and_concepts, args = (features,), axis=1)

    # create the data loaders
    train_loader = DataLoader(DATA["train"]['Drug'], batch_size=32, shuffle=True)
    val_loader = DataLoader(DATA["val"]['Drug'], batch_size=32, shuffle=False)
    test_loader = DataLoader(DATA["test"]['Drug'], batch_size=32, shuffle=False)

    # initialize the model, optimizer, and loss functions
    ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0).to(device)
    model = MolNet(in_channels=DATA["train"]['Drug'][1].x.shape[1], hidden_channels=768).to(device)
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
            torch.save(model, f'{model_folder}/model_gnn_{data_type}_{concept_selector}.pth')
            torch.save(ModelXtoCtoY_layer, f'{model_folder}/ModelXtoCtoY_layer_gnn_{data_type}_{concept_selector}.pth')


    ######### test #########
    model = torch.load(f'{model_folder}/model_gnn_{data_type}_{concept_selector}.pth', weights_only=False)
    ModelXtoCtoY_layer = torch.load(f'{model_folder}/ModelXtoCtoY_layer_gnn_{data_type}_{concept_selector}.pth', weights_only=False)
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

    print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}', flush = True)

    with open(f'{model_folder}/test_loader_gnn_{data_type}_{concept_selector}.pkl', 'wb') as f:
        pkl.dump(test_loader, f)


if __name__ == "__main__":

    # make this scripts usable on cluster -> add arguments for folder locations
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type = str, help = "path to config yaml file")
    ap.add_argument("--data-dir", type = str, help = "path to input data directory")
    ap.add_argument("--output-dir", type = str, help = "path to directory where outputs and logs will be saved")
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(config['seed'])
    data_type = config['data_type']
    num_epochs = config['num_epochs']
    num_concepts = config['num_concepts']
    loss_weight = config['loss_weight']
    concept_selector = config["concept_selector"]

    main(args.data_dir, args.output_dir, data_type, num_epochs, num_concepts, loss_weight, concept_selector)
