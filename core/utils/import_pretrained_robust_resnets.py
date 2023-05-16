# %%
import torch
from robustness.datasets import ImageNet
from robustness import model_utils

from torchvision import models
from collections import OrderedDict
from os.path import join



# Dummy path to use imagenet dataset for loading
torchhome = r"M:\Data\torch_home"
ds = ImageNet(torchhome)

# model_name = 'resnet50_linf8'
robust_resnet_dict = {
    "resnet50_linf8": "resnet50_linf_eps8.0.ckpt",
    "resnet50_linf4": "resnet50_linf_eps4.0.ckpt",
    "resnet50_linf2": "resnet50_linf_eps2.0.ckpt",
    "resnet50_linf1": "resnet50_linf_eps1.0.ckpt",
    "resnet50_linf0.5": "resnet50_linf_eps0.5.ckpt"
}

robust_resnet_save_dict = {
    "resnet50_linf8": "resnet50_linf_eps8.0.pt",
    "resnet50_linf4": "resnet50_linf_eps4.0.pt",
    "resnet50_linf2": "resnet50_linf_eps2.0.pt",
    "resnet50_linf1": "resnet50_linf_eps1.0.pt",
    "resnet50_linf0.5": "resnet50_linf_eps0.5.pt"
}

for model_name in robust_resnet_dict.keys():
    torch_model = models.resnet50(pretrained=True)
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
    # the default models match in keys
    torch_model.state_dict().keys() == model.model.state_dict().keys()
    # load pretrained weights for robust versions https://github.com/microsoft/robust-models-transfer
    file_contents = torch.load(join(torchhome, robust_resnet_dict[model_name]))
    # the saved state_dict (torch.load(join(torchhome, robust_resnet_dict[model_name]))['model'] has keys named as:
    # odict_keys(['module.normalizer.new_mean', 'module.normalizer.new_std', 'module.model.conv1.weight', 'module.model.bn1.weight'
    # while the model.model from robustness and torchvision have keys named
    # odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var'
    # Remove the module. prefix to make them match.
    # https://discuss.pytorch.org/t/how-to-change-the-name-of-the-weights-to-a-new-name-when-saving-the-model/32166/4
    # https://discuss.pytorch.org/t/dataparallel-changes-parameter-names-issue-with-load-state-dict/60211/3
    modified_state_dict = OrderedDict([(key.split("module.")[-1], file_contents['model'][key])
                                       for key in file_contents['model']])
    # First load robust resnet to default resnet from robustness package.
    model.load_state_dict(modified_state_dict)
    # Then, load only the model part of the robust resnet into torchvision resnet50
    torch_model.load_state_dict(model.model.state_dict())
    # Now save the weights to avoid doing this every time we need to import robust resnets, check when we would need
    # full components of the robustness networks
    torch.save(torch_model.state_dict(), join(torchhome, robust_resnet_save_dict[model_name]))
    print('Saving robust weights for torchvision resnet < %s >' % model_name)
