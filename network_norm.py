import network_eliai


class ModuleTypeNorm(network_eliai.ModuleType):
    def create_module(self, net: network_eliai.Network, weights: network_eliai.NetworkWeights):
        if all(x in weights.w for x in ["w_norm", "b_norm"]):
            return NetworkModuleNorm(net, weights)

        return None


class NetworkModuleNorm(network_eliai.NetworkModule):
    def __init__(self,  net: network_eliai.Network, weights: network_eliai.NetworkWeights):
        super().__init__(net, weights)

        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    def calc_updown(self, orig_weight):
        output_shape = self.w_norm.shape
        updown = self.w_norm.to(orig_weight.device, dtype=orig_weight.dtype)

        if self.b_norm is not None:
            ex_bias = self.b_norm.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            ex_bias = None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)
