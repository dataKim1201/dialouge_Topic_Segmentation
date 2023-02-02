
model_name = "DTS_for_dialouge"
model_description = "dialouge Domain Topic Segmentation model"
model_author = "Hanseong Kim"
model_url = "https://github.com/dataKim1201/dialouge_Topic_Segmentation"

code_url = "https://github.com/dataKim1201/dialouge_Topic_Segmentation/blob/main/model.py"

def Domain_Topic_Segmentation(pretrained_id = 'klue/roberta-large'):
    import torch
    import torch.hub
    from model import CSModel
    # model_name = "DTS_for_dialouge"
    # model_description = "dialouge Domain Topic Segmentation model"
    # model_author = "Hanseong Kim"
    # model_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07"

    # code_url = "https://github.com/boostcampaitech4lv23nlp1/final-project-level3-nlp-07/blob/master/DTS/model.py"

    model = CSModel(pretrained_id=pretrained_id)
    model.load_state_dict(torch.load('Domain_Topic_segmentor.pt'))
    return model

# python -m torch.hub publish-model Domain_Topic_Segmentation ./hubconf.py --trust_repo=True