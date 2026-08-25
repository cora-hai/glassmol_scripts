import sys
from apetokenizer.src.apetokenizer import ape_tokenizer
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from backbone.backbone import ModelXtoCtoY_function
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import set_seed, agent
import ast
import pickle as pkl
from utils import MyDataset
import yaml

import argparse

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(in_data_folder, model_folder, tok_folder, data_type, num_epochs, num_concepts, loss_weight, concept_selector):

    tokenizer = APETokenizer()
    tokenizer.load_vocabulary(f'{tok_folder}/tokenizer.json')
    model = AutoModelForSequenceClassification.from_pretrained('mikemayuare/SMILY-APE-BBBP').to(device)
    # print("parameter names:")
    # for name, _ in model.named_parameters():
    #     print(name)

    # load data
    DATA = {}
    DATA['train'] = pd.read_csv(f'{in_data_folder}/train_{data_type}.csv')
    DATA['val'] = pd.read_csv(f'{in_data_folder}/val_{data_type}.csv')
    DATA['test'] = pd.read_csv(f'{in_data_folder}/test_{data_type}.csv')

    # choose num_concepts features with llm agent
    if concept_selector == "llm":
        # original concept selector from GlassMol paper
        features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts).replace("```python", "").replace("```", "")
        features = ast.literal_eval(features)
        print(features)

    elif concept_selector == "l1":
        # according to https://scikit-learn.org/stable/modules/feature_selection.html#l1-based-feature-selection
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        print(f"shape before: {X.shape}")
        # train linear support vector classifier with L1 penalty for "feature selection"
        lsvc = LinearSVC(C=0.01, penalty = "l1", dual = False).fit(X,y)     # C = regularisation parameter, strength inversely proportional to C
        selector = SelectFromModel(lsvc, prefit = True)
        #X_new = selector.transform(X)
        # get selected features
        feature_mask = selector.get_support()
        features = X.columns[feature_mask].tolist()
        num_concepts = len(features)
        #print(f"shape after: {X_new.shape}")
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
        print(f"# of selected features: {num_concepts}")

    elif concept_selector == "late-l1":
        X, y = DATA["train"].drop(columns = ['Drug', 'Y', 'Drug_ID']), DATA["train"]["Y"]
        features = X.columns.to_list()
        num_concepts = len(features)

    else:
        print("choose a valid concept selector method")


    with open(f'{model_folder}/features_llm_{data_type}_{concept_selector}.pkl', 'wb') as f:
        pkl.dump(features, f)

    # means and std for standardization
    means = DATA['train'][features].mean().values
    stds = DATA['train'][features].std().values

    # create the dataloader
    train_loader = DataLoader(MyDataset('train', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=True)
    val_loader = DataLoader(MyDataset('val', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)
    test_loader = DataLoader(MyDataset('test', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)

    # num_concepts is the number of concepts, expand_dim is the dimension of the expanded layer (0 means no expansion)
    ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0, in_dims=768).to(device)

    loss_C = torch.nn.L1Loss()
    loss_Y = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(list(ModelXtoCtoY_layer.parameters()), lr=1e-5)

    best_acc_score = 0
    for epoch in range(num_epochs):
        ######### train #########
        ModelXtoCtoY_layer.train()
        model.eval()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            concept_labels = batch['concept_labels']

            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)

            XtoC_output = outputs[1:] 
            XtoY_output = outputs[0:1]

            # XtoC_loss
            XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
            XtoC_loss = loss_C(XtoC_output.to(device), concept_labels.squeeze().to(device))

            if concept_selector == "late-l1":
                # TODO: select only those parameters corresponding to concept MLP ???
                l1_norm = sum(param.abs().sum() for name, param in model.named_parameters() if name.startswith("classifier"))
                XtoC_loss += 0.01 * l1_norm

            
            # XtoY_loss
            XtoY_loss = loss_Y(XtoY_output[0].squeeze().to(device), label.squeeze().to(device))

            loss = XtoY_loss + XtoC_loss * loss_weight
            
            loss.backward()
            optimizer.step()

        ######### val #########
        ModelXtoCtoY_layer.eval()
        model.eval()
        val_accuracy = 0.
        concept_val_loss = 0.
        predict_labels = np.array([])

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)
                concept_labels = batch['concept_labels']

                outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
                pooled_output = outputs.hidden_states[-1][:,0]

                outputs = ModelXtoCtoY_layer(pooled_output)
                XtoY_output = outputs[0:1]

                predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().cpu() > 0.5) == label.bool().cpu())

            val_accuracy = predict_labels.sum() / len(predict_labels)
            
        if val_accuracy > best_acc_score:
            best_acc_score = val_accuracy
            torch.save(model, f'{model_folder}/model_llm_{data_type}_{concept_selector}.pth')
            torch.save(ModelXtoCtoY_layer, f'{model_folder}/ModelXtoCtoY_layer_llm_{data_type}_{concept_selector}.pth')

    ######### test #########
    model = torch.load(f'm{model_folder}/model_llm_{data_type}_{concept_selector}.pth', weights_only=False)
    ModelXtoCtoY_layer = torch.load(f'{model_folder}/ModelXtoCtoY_layer_llm_{data_type}_{concept_selector}.pth', weights_only=False) 
    model.eval()
    ModelXtoCtoY_layer.eval()

    predict_labels = np.array([])
    true_labels = np.array([])
    predictions = np.array([])
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)
            XtoY_output = outputs[0:1]
            predictions = np.append(predictions, XtoY_output[0].squeeze().to(torch.float32).cpu())
            predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().to(torch.float32).cpu() > 0.0) == label.bool().cpu())

            true_labels = np.append(true_labels, label.bool().cpu())

        test_accuracy = predict_labels.sum() / len(predict_labels)

        with open(f'{model_folder}/test_loader_llm_{data_type}_{concept_selector}.pkl', 'wb') as f:
            pkl.dump(test_loader, f)
        print(f'Test Acc = {test_accuracy*100}')
        print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')

if __name__ == "__main__":

    # make this scripts usable on cluster -> add arguments for folder locations
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type = str, help = "path to config yaml file")
    ap.add_argument("--tok", type = str, help = "path to ape_tokenizer folder")
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

    main(args.data_dir, args.output_dir, args.tok, data_type, num_epochs, num_concepts, loss_weight, concept_selector)