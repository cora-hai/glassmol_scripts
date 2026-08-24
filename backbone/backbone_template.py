import os
import torch
import numpy as np 
import torch.nn as nn

# fully connected layer architecture
class FC(torch.nn.Module):
    def __init__(self, input_dim, output_dim, expand_dim):
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = torch.nn.ReLU()
            self.fc_new = torch.nn.Linear(input_dim, expand_dim)
            self.fc = torch.nn.Linear(expand_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x

# multi-layer perceptron architecture
class MLP(torch.nn.Module):
    def __init__(self, input_dim, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = torch.nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(expand_dim, 1)
        else:
            self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x

# end-to-end model architecture: X -> C -> Y
class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, num_concepts):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.num_concepts = num_concepts

    def forward_stage2(self, stage1_out):
        attr_outputs = stage1_out

        stage2_inputs = attr_outputs

        XtoC_logits = torch.stack(attr_outputs, dim=0)
        XtoC_logits=torch.transpose(XtoC_logits, 0, 1)
        predictions_concept_labels = XtoC_logits.reshape(-1,self.num_concepts*1)

        stage2_inputs = predictions_concept_labels
        all_out = [self.sec_model(stage2_inputs[i]) for i in range(stage2_inputs.size(0))]
        all_out = [torch.stack(all_out, dim=0)]
        all_out.extend(stage1_out)
        return all_out

    def forward(self, x):
        outputs = self.first_model(x) 
        return self.forward_stage2(outputs)

# model architecture: X -> C
class ModelXtoC(torch.nn.Module):
    def __init__(self, num_concepts, expand_dim, in_dims=768):
        super(ModelXtoC, self).__init__()
        self.num_concepts = num_concepts
        #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        self.all_fc = torch.nn.ModuleList()
        dim = in_dims

        for i in range(num_concepts):
            self.all_fc.append(FC(dim, 1, expand_dim))

    def forward(self, x):
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        return out