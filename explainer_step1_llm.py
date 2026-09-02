import torch
import sys
import pickle as pkl
import torch.nn as nn
import numpy as np

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(dataset, model_dir, concept_selector) -> None:

    # load the model
    model = torch.load(f'{model_dir}/model_llm_{dataset}_{concept_selector}.pth', weights_only=False)
    ModelXtoCtoY_layer = torch.load(f'{model_dir}/ModelXtoCtoY_layer_llm_{dataset}_{concept_selector}.pth', weights_only=False)
    ModelXtoCtoY_layer.eval()
    model.eval()

    # load the saved test loader
    with open(f'{model_dir}/test_loader_llm_{dataset}_{concept_selector}.pkl', 'rb') as f:
        test_loader = pkl.load(f)

    # load the features
    with open(f'{model_dir}/features_llm_{dataset}_{concept_selector}.pkl', 'rb') as f:
        features = pkl.load(f)

    # get the contributions for the entire test set
    all_contributions = []
    contributions_dict = {}
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        concept_labels = batch['concept_labels']
        features = batch['features']
                
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True)

        pooled_output = outputs.hidden_states[-1][:,0] 

        outputs = ModelXtoCtoY_layer(pooled_output)
        concepts = torch.stack(outputs[1:], dim=1)
            
        last_layer = None
        for name, m in ModelXtoCtoY_layer.named_modules():
            if name == 'sec_model':
                last_layer = m

        for name, m in last_layer.named_modules():
            if name == 'linear':
                last_layer = m

        W = last_layer.weight.squeeze()

        # contribution calculation as seen in the paper
        contributions = concepts.squeeze().detach().cpu().numpy()*W.detach().cpu().numpy()
        all_contributions.append(contributions.copy())

        for i, feature in enumerate(features):
            try:
                contributions_dict[feature[0]] = np.append(contributions_dict[feature[0]], ((contributions.T)[i]))
            except:
                contributions_dict[feature[0]] = ((contributions.T)[i])

    with open (f'{model_dir}/contributions_llm_{dataset}_{concept_selector}.pkl', 'wb') as f:
        pkl.dump(contributions_dict, f)


if __name__ == "__main__":

    # make this script usable on cluster
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type = str, help = "dataset that model was trained on")
    ap.add_argument("--model-dir", type = str, help = "path to load model from and to write contribution files to")
    ap.add_argument("--concept-selector", type = str, help = "concept selection method")
    args = ap.parse_args()

    main(args.dataset, args.model_dir, args.concept_selector)