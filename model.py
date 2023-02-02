from transformers import AutoModel, AutoConfig
import torch.nn as nn
from collections import OrderedDict

class CSModel(nn.Module):
    def __init__(self,pretrained_id, model_config = None) -> None:
        super(CSModel, self).__init__()
        if model_config == None:
            model_config = AutoConfig.from_pretrained(pretrained_id)
        self.hidden_size = model_config.hidden_size
        self.plm = AutoModel.from_pretrained(pretrained_id, add_pooling_layer = False)
        self.cs_model = nn.Sequential(OrderedDict({'Linear' : nn.Linear(self.hidden_size, self.hidden_size),
                                        'Active_fn_1' : nn.ReLU(),
                                        'Dropout' : nn.Dropout(p=0.1),
                                        'cls_layer' : nn.Linear(self.hidden_size, 1),
                                        'Active_fn_2' : nn.Tanh(),}))
    def forward(self,input):
        required_keys = ['input_ids', 'attention_mask', 'neg_input_ids', 'neg_attention_mask','labels']
        if not all(key in input for key in required_keys):
            raise KeyError("Input is missing required keys: {}".format(required_keys))

        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        neg_input_ids = input['neg_input_ids']
        neg_attention_mask = input['neg_attention_mask']
        pos_scores = self.plm(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).last_hidden_state
        neg_scores = self.plm(
            input_ids = neg_input_ids,
            attention_mask = neg_attention_mask
        ).last_hidden_state
        pos_scores = self.cs_model(pos_scores[:,0,:])
        neg_scores = self.cs_model(neg_scores[:,0,:])

        return {'output' : {'pos' : pos_scores, 'neg' : neg_scores}, 'labels' : input['labels']}
    def inference(self,input_ids, attention_mask):
        # input_ids = input['input_ids']
        # attention_mask = input['attention_mask']
        pos_scores = self.plm(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).last_hidden_state
        pos_scores = self.cs_model(pos_scores[:,0,:])
        return pos_scores