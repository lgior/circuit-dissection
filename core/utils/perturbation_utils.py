import matplotlib.pyplot as plt
import numpy as np
import bisect
from collections import OrderedDict
import torch
from torch import nn
import re


def silence_weight_fraction_fully_connected_unit(weight_mat, target_unit_index, silencing_fraction, do_abs=False):
    # check how many neurons you need to kill to reduce total input by 10%
    # focus on excitatory inputs
    # TODO check sorting is correct
    if silencing_fraction < 0:
        print('Silencing negative weights')
        silencing_fraction *= -1
        unit_in_weights = -weight_mat[target_unit_index, :]
    elif do_abs:
        unit_in_weights = np.abs(weight_mat[target_unit_index, :])
    else:
        unit_in_weights = weight_mat[target_unit_index, :]
    sort_inds_ascending = np.argsort(unit_in_weights)
    pos_sorted_inds_ascending = sort_inds_ascending[unit_in_weights[sort_inds_ascending] >= 0]
    descending_exc_in_weights = unit_in_weights[pos_sorted_inds_ascending][::-1]
    # cumulative input
    cum_input = np.cumsum(descending_exc_in_weights)
    total_input_silenced = cum_input[-1] * silencing_fraction
    cum_input_fraction = cum_input / cum_input[-1]
    silencing_ind = bisect.bisect_left(cum_input_fraction, silencing_fraction)
    weights_to_silence = pos_sorted_inds_ascending[::-1][:silencing_ind + 1]
    weight_mat[target_unit_index, weights_to_silence] = 0
    return weight_mat, weights_to_silence


def silence_weight_topn_fully_connected_unit(weight_mat, target_unit_index, silencing_n):
    # focus on excitatory inputs
    # TODO check the case where inputs to silence are less than valid inputs, eg 3 requested for only 2 exc inputs
    if silencing_n < 0:
        print('Silencing negative weights')
        silencing_n *= -1
        unit_in_weights = -weight_mat[target_unit_index, :]
    else:
        unit_in_weights = weight_mat[target_unit_index, :]

    if silencing_n < 1 or silencing_n > unit_in_weights.shape[0]:
        raise ValueError(
            'Inputs to silence are 0 or larger than the number of inputs {}'.format(unit_in_weights.shape[0]))
    sort_inds_ascending = np.argsort(unit_in_weights)
    pos_sorted_inds_ascending = sort_inds_ascending[unit_in_weights[sort_inds_ascending] >= 0]
    if silencing_n > pos_sorted_inds_ascending.shape[0]:
        raise ValueError(
            'Inputs to silence are larger than the number of inputs {}'.format(pos_sorted_inds_ascending.shape[0]))
    weights_to_silence = pos_sorted_inds_ascending[::-1][:silencing_n]
    weight_mat[target_unit_index, weights_to_silence] = 0
    return weight_mat, weights_to_silence


# %%

def get_custom_modules_dict(model):
    # Testing how to map same recording hook layer to silencing
    module_list = []
    module_names = []

    def named_apply(module, name, prefix=None):
        # resemble the apply function but suits the functions here.
        cprefix = "" if prefix is None else prefix + "." + name
        for cname, child in module.named_children():
            named_apply(child, cname, cprefix)

        if not cprefix == "":
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            print(module.__class__)
            print(prefix)
            print(name)
            print(class_name)
            if (isinstance(module, nn.Sequential)
                    or isinstance(module, nn.ModuleList)
                    or isinstance(module, nn.Container)):
                print('special')
                module_name = prefix + "." + name
            else:
                module_name = prefix + "." + class_name + name
            print(module_name)
            module_names.append(module_name)
            module_list.append(module)

    named_apply(model, "", None)
    custom_module_dict = OrderedDict(zip(module_names, module_list))
    return custom_module_dict


# TODO remember to store original weights so you restore for looping across units
# %%
# GR
from core.utils.perturbation_utils import silence_weight_fraction_fully_connected_unit as silence_weights
from core.utils.perturbation_utils import get_custom_modules_dict


def get_target_module(model, layer: str):
    # List the modules in the same way we do to attach the hooks
    custom_module_dict = get_custom_modules_dict(model)
    if layer in custom_module_dict:
        target_module = custom_module_dict[layer]
    else:
        target_module = None
    return target_module


def apply_silence_weight_topn_fully_connected_unit(model, layer, target_unit_index, silencing_n):
    target_module = get_target_module(model, layer)
    with torch.no_grad():
        original_weights = target_module.weight.detach().clone()
    new_weight_mat, weights_to_silence = silence_weight_topn_fully_connected_unit(
        target_module.weight.detach().cpu().numpy(), target_unit_index, silencing_n)
    device = torch.device("cuda" if original_weights.is_cuda else "cpu")
    target_module.weight = nn.Parameter(torch.from_numpy(new_weight_mat).float().to(device))
    # https://discuss.pytorch.org/t/converting-numpy-array-to-tensor-on-gpu/19423
    with torch.no_grad():
        target_module.weight = nn.Parameter(torch.from_numpy(new_weight_mat).float().to(device))
        print('moved to cuda? %s' % target_module.weight.is_cuda)
    return target_module, original_weights, weights_to_silence


def apply_silence_weight_fraction_fully_connected_unit(model, layer, target_unit_index, silencing_fraction, do_abs=False):
    target_module = get_target_module(model, layer)
    with torch.no_grad():
        original_weights = target_module.weight.detach().clone()
    new_weight_mat, weights_to_silence = silence_weight_fraction_fully_connected_unit(
        target_module.weight.detach().cpu().numpy(), target_unit_index, silencing_fraction, do_abs)
    device = torch.device("cuda" if original_weights.is_cuda else "cpu")
    with torch.no_grad():
        target_module.weight = nn.Parameter(torch.from_numpy(new_weight_mat).float().to(device))
        print('moved to cuda? %s' % target_module.weight.is_cuda)
    return target_module, original_weights, weights_to_silence





# #%%
# import matplotlib as mpl
# plt.pcolor(target_module.weight.detach().cpu().squeeze().numpy()[(131-3):(131+4), :100], cmap='RdBu', norm=mpl.colors.CenteredNorm(0))
# plt.show()
#
# # %%
# target_module = get_target_module(scorer.model, args.layer)
#
# new_weight_mat = silence_weight_topn_fully_connected_unit(target_module.weight.detach().cpu().numpy(), 131, 0)
# with torch.no_grad():
#     target_module.weight = nn.Parameter(torch.from_numpy(new_weight_mat).float())
#
# #%%
# plt.pcolor(scorer.model.classifier[6].weight.detach().cpu().squeeze().numpy()[(131-3):(131+4), :400], cmap='RdBu', norm=mpl.colors.CenteredNorm(0))
# plt.show()

# %%
