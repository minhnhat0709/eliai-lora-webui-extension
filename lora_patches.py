import torch

import networks_eliai
from modules import patches


class LoraPatches:
    def __init__(self):
        self.Linear_forward = patches.patch(__name__, torch.nn.Linear, 'forward', networks_eliai.network_EliAI_Linear_forward)
        self.Linear_load_state_dict = patches.patch(__name__, torch.nn.Linear, '_load_from_state_dict', networks_eliai.network_EliAI_Linear_load_state_dict)
        self.Conv2d_forward = patches.patch(__name__, torch.nn.Conv2d, 'forward', networks_eliai.network_EliAI_Conv2d_forward)
        self.Conv2d_load_state_dict = patches.patch(__name__, torch.nn.Conv2d, '_load_from_state_dict', networks_eliai.network_EliAI_Conv2d_load_state_dict)
        self.GroupNorm_forward = patches.patch(__name__, torch.nn.GroupNorm, 'forward', networks_eliai.network_EliAI_GroupNorm_forward)
        self.GroupNorm_load_state_dict = patches.patch(__name__, torch.nn.GroupNorm, '_load_from_state_dict', networks_eliai.network_EliAI_GroupNorm_load_state_dict)
        self.LayerNorm_forward = patches.patch(__name__, torch.nn.LayerNorm, 'forward', networks_eliai.network_EliAI_LayerNorm_forward)
        self.LayerNorm_load_state_dict = patches.patch(__name__, torch.nn.LayerNorm, '_load_from_state_dict', networks_eliai.network_EliAI_LayerNorm_load_state_dict)
        self.MultiheadAttention_forward = patches.patch(__name__, torch.nn.MultiheadAttention, 'forward', networks_eliai.network_EliAI_MultiheadAttention_forward)
        self.MultiheadAttention_load_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict', networks_eliai.network_EliAI_MultiheadAttention_load_state_dict)

    def undo(self):
        self.Linear_forward = patches.undo(__name__, torch.nn.Linear, 'forward')
        self.Linear_load_state_dict = patches.undo(__name__, torch.nn.Linear, '_load_from_state_dict')
        self.Conv2d_forward = patches.undo(__name__, torch.nn.Conv2d, 'forward')
        self.Conv2d_load_state_dict = patches.undo(__name__, torch.nn.Conv2d, '_load_from_state_dict')
        self.GroupNorm_forward = patches.undo(__name__, torch.nn.GroupNorm, 'forward')
        self.GroupNorm_load_state_dict = patches.undo(__name__, torch.nn.GroupNorm, '_load_from_state_dict')
        self.LayerNorm_forward = patches.undo(__name__, torch.nn.LayerNorm, 'forward')
        self.LayerNorm_load_state_dict = patches.undo(__name__, torch.nn.LayerNorm, '_load_from_state_dict')
        self.MultiheadAttention_forward = patches.undo(__name__, torch.nn.MultiheadAttention, 'forward')
        self.MultiheadAttention_load_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict')

