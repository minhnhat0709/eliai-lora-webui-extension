import importlib
import re

import gradio as gr
from fastapi import FastAPI

import network_eliai
import networks_eliai
import lora_eliai  # noqa:F401
import extra_networks_eliai_lora
import ui_extra_networks_eliai_lora
import torch
import os
from modules import script_callbacks, ui_extra_networks, extra_networks, shared, launch_utils
 

git_tag = launch_utils.git_tag()
def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_eliai_network
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_eliai_network
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_eliai_network
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_eliai_network
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before__eliai_network
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_eliai_network
    if "1.5" not in git_tag:
        #Normalize
        torch.nn.GroupNorm.forward = torch.nn.GroupNorm_forward_before_eliai_network
        torch.nn.GroupNorm._load_from_state_dict = torch.nn.GroupNorm_load_state_dict_before_eliai_network
        torch.nn.LayerNorm.forward = torch.nn.LayerNorm_forward_before_eliai_network
        torch.nn.LayerNorm._load_from_state_dict = torch.nn.LayerNorm_load_state_dict_before_eliai_network

    #Reload Normal Lora
    # lora_networks = importlib.import_module("networks")
    # lora_patches = importlib.import_module("lora_patches")
    # lora_networks.originals = lora_patches.LoraPatches()

def before_ui():
    os.makedirs(shared.cmd_opts.eliai_lora_dir, exist_ok=True)
    if os.path.isfile(shared.cmd_opts.eliai_lora_dir + '\\token.txt') == False:
        with open(f'{shared.cmd_opts.eliai_lora_dir}\\token.txt', 'wt') as f:
            f.write('')

    #unload normal Lora
    # git_tag = launch_utils.git_tag

    if "1.5" in git_tag:
        lora_script = importlib.import_module("extensions-builtin.Lora.scripts.lora_script")
        lora_script.unload()
        # pass
    else:
        lora_networks = importlib.import_module("networks")
        lora_networks.originals.undo()

    ui_extra_networks.register_page(ui_extra_networks_eliai_lora.ExtraNetworksPageEliAILora())

    extra_network = extra_networks_eliai_lora.ExtraNetworkLoraEliAI()
    extra_networks.register_extra_network(extra_network)
    extra_networks.register_extra_network_alias(extra_network, "lyco")


if not hasattr(torch.nn, 'Linear_forward_before_eliai_network'):
    torch.nn.Linear_forward_before_eliai_network = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_eliai_network'):
    torch.nn.Linear_load_state_dict_before_eliai_network = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_eliai_network'):
    torch.nn.Conv2d_forward_before_eliai_network = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_eliai_network'):
    torch.nn.Conv2d_load_state_dict_before_eliai_network = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before__eliai_network'):
    torch.nn.MultiheadAttention_forward_before__eliai_network = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_eliai_network'):
    torch.nn.MultiheadAttention_load_state_dict_before_eliai_network = torch.nn.MultiheadAttention._load_from_state_dict


torch.nn.Linear.forward = networks_eliai.network_EliAI_Linear_forward
torch.nn.Linear._load_from_state_dict = networks_eliai.network_EliAI_Linear_load_state_dict
torch.nn.Conv2d.forward = networks_eliai.network_EliAI_Conv2d_forward
torch.nn.Conv2d._load_from_state_dict = networks_eliai.network_EliAI_Conv2d_load_state_dict
torch.nn.MultiheadAttention.forward = networks_eliai.network_EliAI_MultiheadAttention_forward
torch.nn.MultiheadAttention._load_from_state_dict = networks_eliai.network_EliAI_MultiheadAttention_load_state_dict

if "1.5" not in git_tag:
    #Normalize
    if not hasattr(torch.nn, 'GroupNorm_forward_before_eliai_network'):
        torch.nn.GroupNorm_forward_before_eliai_network = torch.nn.GroupNorm.forward
    if not hasattr(torch.nn, 'GroupNorm_load_state_dict_before_eliai_network'):
        torch.nn.GroupNorm_load_state_dict_before_eliai_network = torch.nn.GroupNorm._load_from_state_dict
    if not hasattr(torch.nn, 'LayerNorm_forward_before_eliai_network'):
        torch.nn.LayerNorm_forward_before_eliai_network = torch.nn.LayerNorm.forward
    if not hasattr(torch.nn, 'LayerNorm_load_state_dict_before_eliai_network'):
        torch.nn.LayerNorm_load_state_dict_before_eliai_network = torch.nn.LayerNorm._load_from_state_dict
    torch.nn.GroupNorm.forward = networks_eliai.network_EliAI_GroupNorm_forward
    torch.nn.GroupNorm._load_from_state_dict = networks_eliai.network_EliAI_GroupNorm_load_state_dict
    torch.nn.LayerNorm.forward = networks_eliai.network_EliAI_LayerNorm_forward
    torch.nn.LayerNorm._load_from_state_dict = networks_eliai.network_EliAI_LayerNorm_load_state_dict



script_callbacks.on_model_loaded(networks_eliai.assign_network_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks_eliai.infotext_pasted)


shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks_eliai.available_networks]}, refresh=networks_eliai.list_available_networks),
    "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
    "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
    "lora_show_all": shared.OptionInfo(False, "Always show all networks_eliai on the Lora page").info("otherwise, those detected as for incompatible version of Stable Diffusion will be hidden"),
    "lora_hide_unknown_for_versions": shared.OptionInfo([], "Hide networks_eliai of unknown versions for model versions", gr.CheckboxGroup, {"choices": ["SD1", "SD2", "SDXL"]}),
}))


shared.options_templates.update(shared.options_section(('compatibility', "Compatibility"), {
    "lora_functional": shared.OptionInfo(False, "Lora/Networks: use old method that takes longer when you have multiple Loras active and produces same results as kohya-ss/sd-webui-additional-networks_eliai extension"),
}))


# def create_lora_json(obj: network_eliai.NetworkOnDisk):
#     return {
#         "name": obj.name,
#         "alias": obj.alias,
#         "path": obj.filename,
#         "metadata": obj.metadata,
#     }


# def api_networks(_: gr.Blocks, app: FastAPI):
#     @app.get("/sdapi/v1/loras")
#     async def get_loras():
#         return [create_lora_json(obj) for obj in networks_eliai.available_networks.values()]

#     @app.post("/sdapi/v1/refresh-loras")
#     async def refresh_loras():
#         return networks_eliai.list_available_networks()


# script_callbacks.on_app_started(api_networks)

re_lora = re.compile("<lora:([^:]+):")


def infotext_pasted(infotext, d):
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    hashes = [x.strip().split(':', 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}

    def network_replacement(m):
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks_eliai.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f'<lora:{network_on_disk.get_alias()}:'

    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_infotext_pasted(infotext_pasted)