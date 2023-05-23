import copy

import torch

from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class OWMPlugin(StrategyPlugin):
    """
    A OWM plugin introduced in paper: https://arxiv.org/abs/1810.01256.
    """


    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha and param temperature are not used in this method
        They will be removed in future
        """

        super().__init__()

        dtype = torch.cuda.FloatTensor  # run on GPU
        with torch.no_grad():
            self.Pl = torch.autograd.Variable(torch.eye(2048).type(dtype))

    # @torch.no_grad()
    def before_update(self, strategy, **kwargs):
        # import pdb
        # pdb.set_trace()

        lamda = strategy.i_batch / len(strategy.dataloader) + 1
        alpha = 1.0 * 1 ** lamda
        # x_mean = torch.mean(strategy.mb_x, 0, True)
        if strategy.experience_id != 0:
            for n, w in strategy.model.named_parameters():
                if n == "module.weight":
                    # self.pro_weight(self.Pl, x_mean, w, alpha=alpha_array, stride=2, cnn=False)
                    r = torch.mean(strategy.mb_x, 0, True)
                    k = torch.mm(self.Pl, torch.t(r))
                    # self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))

                    pnorm2 = torch.norm(self.Pl.data,p='fro')
                    self.Pl.data = self.Pl.data / pnorm2
                    w.grad.data = torch.mm(w.grad.data, torch.t(self.Pl.data))

    def pro_weight(self, p,x, w, alpha=1.0, cnn="layer", stride=1):
        if cnn == "layer":
            _, _, L = x.shape
            F, _, LL = w.shape
            S = stride  # stride
            Lo = int(1 + (L - LL) / S)
            for i in range(Lo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                r = x[:, :, i * S: i * S + LL].contiguous().view(1, -1)
                # r = r[:, range(r.shape[1] - 1, -1, -1)]
                k = torch.mm(p, torch.t(r))
                p = torch.sub(p, (torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k))))                                    
            w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
        elif cnn=="pooling":
            r = x.squeeze()
            k = torch.mm(p, torch.t(r))
            p = torch.sub(p, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))
            w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
        else:
            r = x
            k = torch.mm(p, torch.t(r))
            p = torch.sub(p, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))
            w.grad.data = torch.mm(w.grad.data, torch.t(p.data))