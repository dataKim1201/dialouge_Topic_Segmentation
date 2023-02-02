import torch
import torch.hub
from model import CSModel
model_name = "DTS_for_dialouge"
model_description = "dialouge Domain Topic Segmentation model"
model_author = "Hanseong Kim"
model_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07"

code_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07/blob/master/DTS/model.py"

def DTS_for_dialouge(pretrained_id = 'klue/roberta-large'):
    import torch
    import torch.hub
    from model import CSModel
    # model_name = "DTS_for_dialouge"
    # model_description = "dialouge Domain Topic Segmentation model"
    # model_author = "Hanseong Kim"
    # model_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07"

    # code_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07/blob/master/DTS/model.py"

    model = CSModel(pretrained_id=pretrained_id)
    model = model.load_state_dict(torch.load('/opt/ml/input/dialouge_Topic_Segmentation/Domain_Topic_segmentor.pt'))
    return model