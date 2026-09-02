import numpy as np
import pickle as pkl
import pandas as pd
import sys

import argparse


def main(model_dir, data_dir, model_type, dataset, analyze_mol):

    with open(f'{model_dir}/contributions_{model_type}_{dataset}.pkl', 'rb') as f:
        contributions = pkl.load(f)

    data = pd.read_csv(f'{data_dir}/test_{dataset}.csv').iloc[analyze_mol]

    # create a dictionary of the contributions for the molecule
    contributions_2 = {data['Drug_ID']: []}

    for k, v in contributions.items():
        for i in range(len(v)):
            if i == analyze_mol:
                contributions_2[data['Drug_ID']].append({'name': k, 'value': v[i]})

    # sort the contributions by absolute value
    for k, v in contributions_2.items():
        contributions_2[k] = sorted(v, key=lambda x: abs(x['value']), reverse=True)

    # retain the features that have non-zero values in the original data
    contributions_3 = {}
    for k, v in contributions_2.items():
        contributions_3[k] = [i for i in v if data[i['name']].item() != 0]

    print(f'Molecule name: {data["Drug_ID"]}\n' +'='*len(f'Molecule name: {data["Drug_ID"]}'))
    for idx, i in enumerate(contributions_3[data['Drug_ID']]):
        print(f'{i["name"]}: {i["value"]:.4f}')
        print('-'*(len(str(f'{i["name"]}: {i["value"]:.4f}'))-0))
        if idx > 1:
            print('...')
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type = str, help = "path to load model/ contribution files from")
    ap.add_argument("--data-dir", type = str, help = "path to input data directory")
    ap.add_argument("--model-type", type = str, help = "llm or gnn")
    ap.add_argument("--dataset", type = str, help = "dataset that the model was trained on")
    ap.add_argument("--analyze-mol", type = str, help = "ndex of the molecule in the test set to analyze")
    args = ap.parse_args()

    main(args.model_dir, args.data_dir, args.model_type, args.dataset, args.analyze_mol)