import os

import network_eliai
import networks_eliai

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from ui_edit_user_metadata import LoraUserMetadataEditor


class ExtraNetworksPageEliAILora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('EliAI Lora')

    def refresh(self):
        networks_eliai.list_available_networks()

    def create_item(self, name, index=None, enable_filter=True):
        lora_on_disk = networks_eliai.available_networks.get(name)

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(lora_on_disk.filename) + " " + (lora_on_disk.hash or ""),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
            "sd_version": lora_on_disk.sd_version.name,
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = quote_js(f"<eliai:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + quote_js(">")

        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)

        sd_version = item["user_metadata"].get("sd version")
        if sd_version in network_eliai.SdVersion.__members__:
            item["sd_version"] = sd_version
            sd_version = network_eliai.SdVersion[sd_version]
        else:
            sd_version = lora_on_disk.sd_version

        if shared.opts.lora_show_all or not enable_filter:
            pass
        elif sd_version == network_eliai.SdVersion.Unknown:
            model_version = network_eliai.SdVersion.SDXL if shared.sd_model.is_sdxl else network_eliai.SdVersion.SD2 if shared.sd_model.is_sd2 else network_eliai.SdVersion.SD1
            if model_version.name in shared.opts.lora_hide_unknown_for_versions:
                return None
        elif shared.sd_model.is_sdxl and sd_version != network_eliai.SdVersion.SDXL:
            return None
        elif shared.sd_model.is_sd2 and sd_version != network_eliai.SdVersion.SD2:
            return None
        elif shared.sd_model.is_sd1 and sd_version != network_eliai.SdVersion.SD1:
            return None

        return item

    def list_items(self):
        for index, name in enumerate(networks_eliai.available_networks):
            item = self.create_item(name, index)

            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.eliai_lora_dir, shared.cmd_opts.lyco_dir_backcompat]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)