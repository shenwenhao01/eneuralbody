from matplotlib.pyplot import grid
import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from . import embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #self.c = nn.Embedding(6890, 16)
        #self.xyzc_net = SparseConvNet()

        self.latent = nn.Embedding(cfg.num_train_frame, 128)
        
        self.triplane_res = 128
        self.triplane_c = 32
        self.triplanes = nn.Parameter( torch.randn((self.triplane_c * 3, self.triplane_res, self.triplane_res)) )

        # Decoder
        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(self.triplane_c*3, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_grid_coords(self, pts, sp_input):
        '''
        Returns:
        
        grid_coords: (batch, num_pixel * num_sample, 3)   whd(xyz): 分布在[-1,1]之间'''
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def get_grid_coords_(self, pts, viewdir, sp_input):
        '''
        Returns:
        
        grid_coords: (batch, num_pixel * num_sample, 3)   whd(xyz): 分布在[-1,1]之间'''
        # convert xyz to the voxel coordinate dhw
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]] - 0.5
        max_dhw = sp_input['bounds'][:, 1, [2, 1, 0]] + 0.5
        #valid_msk = ((pts>min_dhw[:,]) & (pts<max_dhw[:,])).all(dim=-1)
        #pts = pts[valid_msk]
        #viewdir = viewdir[valid_msk]
        dhw = pts[..., [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / ( max_dhw - min_dhw)
        # convert the voxel coordinate to [0, 1]
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return (grid_coords+1.)/2., viewdir

    def bilinear_sample_triplanes(self, points, feat_xy, feat_yz, feat_xz):
        '''
        Inputs:
        
        points: (batch, num_pixel * num_sample, 3) bbox坐标系内的点, 大部分分布在[-1, 1]
        
        Returns:
        
        xyz_f: (batch, num_pixel * num_sample, channel)'''
        batch, _, _ = points.shape
        feat_xy = feat_xy.unsqueeze(0).repeat(batch, 1, 1, 1)
        feat_yz = feat_yz.unsqueeze(0).repeat(batch, 1, 1, 1)
        feat_xz = feat_xz.unsqueeze(0).repeat(batch, 1, 1, 1)
        batch, channel, H, W = feat_xy.shape
        points = points.reshape(batch, 1, -1, 3)
        #points = points/1.5
        x = points[ ... , 0:1]
        y = points[ ... , 1:2] 
        z = points[ ... , 2:3] 
        xy = torch.cat([x, y], dim=-1)
        xz = torch.cat([x, z], dim=-1)
        yz = torch.cat([y, z], dim=-1)

        xy_f = F.grid_sample(feat_xy, grid=xy, mode='bilinear', align_corners=True)
        xz_f = F.grid_sample(feat_xz, grid=xz, mode='bilinear', align_corners=True)
        yz_f = F.grid_sample(feat_yz, grid=yz, mode='bilinear', align_corners=True)

        xyz_f = (xy_f + xz_f + yz_f).reshape(batch, -1, channel)

        return xyz_f

    def nn_sample_triplanes(self, points, feat_xy, feat_yz, feat_xz):
        batch, _, _ = points.shape
        feat_xy = feat_xy.unsqueeze(0).repeat(batch, 1, 1, 1)
        feat_yz = feat_yz.unsqueeze(0).repeat(batch, 1, 1, 1)
        feat_xz = feat_xz.unsqueeze(0).repeat(batch, 1, 1, 1)
        batch, channel, H, W = feat_xy.shape
        points = points.reshape(batch, 1, -1, 3)
        #points = points/1.5
        x = points[ ... , 0:1]
        y = points[ ... , 1:2] 
        z = points[ ... , 2:3] 
        xy = torch.cat([x, y], dim=-1)
        xz = torch.cat([x, z], dim=-1)
        yz = torch.cat([y, z], dim=-1)

        xy_f = F.grid_sample(feat_xy, grid=xy, mode='nearest', align_corners=True)
        xz_f = F.grid_sample(feat_xz, grid=xz, mode='nearest', align_corners=True)
        yz_f = F.grid_sample(feat_yz, grid=yz, mode='nearest', align_corners=True)

        xyz_f = (xy_f + xz_f + yz_f).reshape(batch, -1, channel)

        return xyz_f

    def calculate_density(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)                      # smpl coord 采样点坐标
        grid_coords = self.get_grid_coords(ppts, sp_input)              # (batch, num_pixel * num_sample, 3) 大部分分布在[-1, 1]
        
        feat_xy, feat_yz, feat_xz = self.triplanes.chunk(3, dim=0)

        feats = self.bilinear_sample_triplanes(grid_coords, feat_xy, feat_yz, feat_xz)
        
        # decoder -> alpha & rgb
        net = self.actvn(self.fc_0(feats))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = net[..., 0:1].permute(0, 2, 1)          # (batch, 1, num_pixel * num_sample)
        alpha = alpha.transpose(1, 2)

        return alpha

    def calculate_density_color(self, wpts, viewdir, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)                      # smpl coord 采样点坐标

        plane_xy, plane_yz, plane_xz = self.triplanes.chunk(3, dim=0)   # (32,128, 128)

        # (batch, num_pixel * num_sample, 3) 分布在[0, 1]
        grid_coords, viewdir_inlier = self.get_grid_coords_(ppts, viewdir, sp_input)
        grid_coords = grid_coords * (self.triplane_res - 1)
        grid_coords = torch.round(grid_coords).clamp(min=0, max= self.triplane_res).to(torch.int64)          # Nearest neighbor
        x, y, z = grid_coords[..., 0], grid_coords[..., 1], grid_coords[...,2]

        feat_xy = plane_xy[:, x, y]
        feat_xz = plane_xz[:, x, z]
        feat_yz = plane_yz[:, y, z]

        feats = torch.cat((feat_xy, feat_yz, feat_xz), dim=0).permute(1, 0, 2)

        # feats = self.bilinear_sample_triplanes(grid_coords, feat_xy, feat_yz, feat_xz)
        # feats = self.nn_sample_triplanes(grid_coords, feat_xy, feat_yz, feat_xz)

        # decoder -> alpha & rgb
        # calculate density
        net = self.actvn(self.fc_0(feats))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        # calculate color
        features = self.feature_fc(net)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = embedder.view_embedder(viewdir_inlier)
        viewdir = viewdir.transpose(1, 2)
        light_pts = embedder.xyz_embedder(wpts)
        light_pts = light_pts.transpose(1, 2)

        features = torch.cat((features, viewdir, light_pts), dim=1)

        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw

    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        raise NotImplementedError
        feat_xy, feat_yz, feat_xz = self.triplanes.chunk(3, dim=0)

        feats = self.bilinear_sample_triplanes(grid_coords, feat_xy, feat_yz, feat_xz)
        
        # decoder -> alpha & rgb
        net = self.actvn(self.fc_0(feats))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = net[..., 0:1].permute(0, 2, 1)          # (batch, 1, num_pixel * num_sample)

        rgb = net[..., 1:4].permute(0, 2, 1)              # (batch, 3, num_pixel * num_sample)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)           # (batch, num_pixel * num_sample, 4)

        return raw


if __name__ == "__main__":
    net = Network()
    print(net)