import copy

import torch

from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class RAWMPlugin(StrategyPlugin):
    """
    A Regularized Adaptive Weight Modification plugin.
    """


    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha and param temperature is not used in this method
        They will be removed in future
        """

        super().__init__()

        dtype = torch.cuda.FloatTensor  # run on GPU
        with torch.no_grad():
            self.Pl = torch.autograd.Variable(torch.eye(2048).type(dtype))
            self.Ql = torch.autograd.Variable(torch.eye(2048).type(dtype))
        """ In Avalanche, targets of different experiences are not ordered. . 
        """

    # @torch.no_grad()
    def before_update(self, strategy, **kwargs):
        background_num, baseball_num, bus_num, camera_num, cosplay_num, dress_num, hockey_num, laptop_num, racing_num, soccer_num, sweater_num = torch.unique(strategy.mb_y, return_counts=True)[1]
        # The compuation of beta is illustrated in our paper
        # which should be modified according to the downstream task
        beta = (baseball_num + soccer_num + hockey_num + laptop_num + camera_num) / (background_num + cosplay_num + dress_num + racing_num + sweater_num + bus_num) 
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

                    mq = torch.mm(self.Pl, torch.linalg.solve(torch.mm(torch.t(self.Pl), self.Pl), torch.t(self.Pl)))
                    e = torch.autograd.Variable(torch.eye(len(mq)).type(torch.cuda.FloatTensor))
                    self.Ql.data = (e - mq).data                                    

                    qnorm2 = torch.norm(self.Ql.data, p='fro')
                    self.Ql.data = self.Ql.data / qnorm2
                    pwq = self.Pl + beta*self.Ql

                    w.grad.data = torch.mm(w.grad.data, torch.t(pwq.data))
