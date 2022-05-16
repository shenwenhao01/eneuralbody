import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
from lib.train import make_optimizer


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch, rand_bkgd=None):
        ret = self.renderer.render(batch, rand_bkgd=rand_bkgd)       # if_clight_renderer

        idx = torch.nonzero(batch['rgb'][:, :, 0] < 0)
        if rand_bkgd is not None:
            batch['rgb'][idx] = rand_bkgd
        else:
            batch['rgb'][idx] = 0.

        scalar_stats = {}
        loss = 0
        
        mask = batch['mask_at_box']
        try:
            img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        except:
            #mask = torch.full( (1, int(mask.sum()) ), True, dtype=bool)
            mask = torch.ones( (1, int(mask.sum()) )).astype(bool)
            img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
            #print(mask.shape, ret['rgb_map'].shape, batch['rgb'].shape)
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
