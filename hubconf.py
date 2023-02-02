import torch
import torch.hub
from model import CSModel, model_d
model_name = "Domain_Topic_Segmentation"
model_description = "dialouge Domain Topic Segmentation model"
model_author = "Hanseong Kim"
model_url = "https://github.com/dataKim1201/dialouge_Topic_Segmentation"

code_url = "https://github.com/dataKim1201/dialouge_Topic_Segmentation/blob/main/model.py"
def Domain_Topic_Segmentation():
    import torch
    import torch.hub
    from model import CSModel,model_d
    # model_name = "DTS_for_dialouge"
    # model_description = "dialouge Domain Topic Segmentation model"
    # model_author = "Hanseong Kim"
    # model_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07"

    # code_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07/blob/master/DTS/model.py"

    model_return  = model_d
    return model_return

# python -m torch.hub publish-model Domain_Topic_Segmentation ./hubconf.py --trust_repo=True