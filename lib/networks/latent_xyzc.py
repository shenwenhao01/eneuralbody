from matplotlib.pyplot import grid
import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from .networks_stylegan2 import SynthesisNetwork, MappingNetwork
from . import embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #self.c = nn.Embedding(6890, 16)
        #self.xyzc_net = SparseConvNet()

        self.latent = nn.Embedding(cfg.num_train_frame, 128)
        
        self.triplane_res = 128
        self.triplane_c = 48
        self.multi_res = True
        if not self.multi_res:
            self.triplanes = nn.Parameter( torch.randn((self.triplane_c * 3, self.triplane_res, self.triplane_res)) )
        else:
            triplane_c = int(self.triplane_c / 4)
            self.res_list = [16, 64, 128, 512]
            self.triplane_0 = nn.Parameter( torch.randn((triplane_c * 3, self.res_list[0] ,self.res_list[0] )) )
            self.triplane_1 = nn.Parameter( torch.randn((triplane_c * 3, self.res_list[1] ,self.res_list[1] )) )
            self.triplane_2 = nn.Parameter( torch.randn((triplane_c * 3, self.res_list[2] ,self.res_list[2])) )
            self.triplane_3 = nn.Parameter( torch.randn((triplane_c * 3, self.res_list[3], self.res_list[3])) )
            self.triplanes_list = [self.triplane_0, self.triplane_1, self.triplane_2, self.triplane_3]

        self.use_bilinear = True
        # dynamic deformation
        self.use_dynamic = False
        self.use_timestep = False
        if self.use_dynamic:
            self.w_dim = 512
            self.z_dim = 0
            if self.use_timestep:
                self.c_dim = embedder.time_dim
            else:
                self.total_frames = 300
                self.c_dim = 128
                self.latent_embedding = nn.Embedding(self.total_frames, self.c_dim)
            mapping_kwargs = {}
            self.deform_synthesis = SynthesisNetwork(w_dim = self.w_dim, 
                                                    img_resolution = self.triplane_res, 
                                                    img_channels = self.triplane_c * 3,
                                                    num_fp16_res = 0,
                                                    use_noise = True)
            self.deform_mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, 
                                                num_ws=self.deform_synthesis.num_ws, **mapping_kwargs)        

        # Neuralbody Decoder
        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(self.triplane_c * 3, 256, 1)
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
        
        grid_coords: (batch, num_pixel * num_sample, 3)   whd(xyz): 分布在[0,1]之间'''
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
        return grid_coords, viewdir

    def bilinear_sample_triplanes(self, points, feat_xy, feat_yz, feat_xz):
        '''
        Inputs:
        
        points: (batch, num_pixel * num_sample, 3) bbox坐标系内的点, 大部分分布在[-1, 1]
        
        Returns:
        
        xyz_f: (batch, num_pixel * num_sample, 3 * channel)'''
        points = points * 2. - 1.
        bs, _, _ = points.shape
        feat_xy = feat_xy.unsqueeze(0).repeat(bs, 1, 1, 1)
        feat_yz = feat_yz.unsqueeze(0).repeat(bs, 1, 1, 1)
        feat_xz = feat_xz.unsqueeze(0).repeat(bs, 1, 1, 1)
        bs, channel, H, W = feat_xy.shape
        points = points.reshape(bs, 1, -1, 3)
        x = points[ ... , 0:1]
        y = points[ ... , 1:2]
        z = points[ ... , 2:3]
        xy = torch.cat([x, y], dim=-1)
        xz = torch.cat([x, z], dim=-1)
        yz = torch.cat([y, z], dim=-1)

        xy_f = F.grid_sample(feat_xy, grid=xy, mode='bilinear', align_corners=True)
        xz_f = F.grid_sample(feat_xz, grid=xz, mode='bilinear', align_corners=True)
        yz_f = F.grid_sample(feat_yz, grid=yz, mode='bilinear', align_corners=True)

        xyz_f = torch.cat((xy_f, yz_f, xz_f), dim=1).reshape(bs, channel * 3, -1)

        return xyz_f

    def my_2d_bilinear_sample(self, res, ix, iy, feat):
        '''
        Inputs:
        
        points: (batch, num_pixel * num_sample, 2) bbox坐标系内的点 [0, 1]
        feat_xy: (channel, H, W)
        
        Returns:
        
        xy_f: (batch, num_pixel * num_sample, channel)
        '''
        #feat = feat.unsqueeze(0).repeat(bs, 1, 1, 1)        # (channel, H, W)
        ix = ix * (res - 1)            # (batch, num_pixel * num_sample, 1)
        iy = iy * (res - 1)

        with torch.no_grad():
            ix_floor = torch.floor(ix).long()
            iy_floor = torch.floor(iy).long()
            ix_ceil = ix_floor + 1
            iy_ceil = iy_floor + 1
        
        w_00 = (ix_ceil - ix) * (iy_ceil - iy).unsqueeze(0)      # (1, bs, n, 1)
        w_11 = (ix - ix_floor) * (iy - iy_floor).unsqueeze(0)
        w_10 = (ix - ix_floor) * (iy_ceil - iy).unsqueeze(0)
        w_01 = (ix_ceil - ix) * (iy - iy_floor).unsqueeze(0)

        with torch.no_grad():
            torch.clamp(ix_floor, 0, res-1, out=ix_floor)
            torch.clamp(iy_floor, 0, res-1, out=iy_floor)
            torch.clamp(ix_ceil, 0, res-1, out=ix_ceil)
            torch.clamp(iy_ceil, 0, res-1, out=iy_ceil)
        
        v_00 = feat[:, ix_floor, iy_floor]          # (channel, bs, n, 1)
        v_11 = feat[:, ix_ceil, iy_ceil]
        v_01 = feat[:, ix_floor, iy_ceil]
        v_10 = feat[:, ix_ceil, iy_floor]

        ret = (v_00 * w_00 + v_01 * w_01 + v_10 * w_10 + v_11 * w_11 ).permute(1, 0, 2, 3)      # (self.triplane_c, bs, n, 1)
        return ret.squeeze(-1)         # (bs, self.triplane_c, n)

    def calculate_density(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)                      # smpl coord 采样点坐标

        triplanes = self.triplanes
        if self.use_dynamic:
            if self.use_timestep:
                latent_embedding = embedder.time_embedder( sp_input['time_step'] ).unsqueeze(0)
            else:
                latent_embedding = self.latent_embedding(sp_input['latent_index'])
            ws = self.deform_mapping(z=None, c = latent_embedding)
            deform_triplanes = self.deform_synthesis( ws[:, :self.deform_synthesis.num_ws] )[0]
            triplanes = triplanes + deform_triplanes
        plane_xy, plane_yz, plane_xz = triplanes.chunk(3, dim=0)   # (32,128, 128)

        viewdir = None
        grid_coords, viewdir_inlier = self.get_grid_coords_(ppts, viewdir, sp_input)
        if not self.use_bilinear:
            grid_coords = grid_coords * (self.triplane_res - 1)
            grid_coords = torch.round(grid_coords).clamp(min=0, max= self.triplane_res).to(torch.int64)
            x, y, z = grid_coords[..., 0], grid_coords[..., 1], grid_coords[...,2]
            feat_xy = plane_xy[:, x, y]
            feat_xz = plane_xz[:, x, z]
            feat_yz = plane_yz[:, y, z]
            feats = torch.cat((feat_xy, feat_yz, feat_xz), dim=0).permute(1, 0, 2)
        else:
            grid_coords = grid_coords * 2. - 1.
            feats = self.bilinear_sample_triplanes(grid_coords, plane_xy, plane_yz, plane_xz)

        net = self.actvn(self.fc_0(feats))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)
        alpha = alpha.transpose(1, 2)

        return alpha

    def calculate_density_color(self, wpts, viewdir, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)                                          # smpl coord 采样点坐标
        grid_coords, viewdir_inlier = self.get_grid_coords_(ppts, viewdir, sp_input)        # (batch, num_pixel * num_sample, 3) 分布在[0, 1]

        if not self.multi_res:
            triplanes = self.triplanes
            if self.use_dynamic:
                if self.use_timestep:
                    latent_embedding = embedder.time_embedder( sp_input['time_step'] ).unsqueeze(0)
                else:
                    latent_embedding = self.latent_embedding(sp_input['latent_index'])
                ws = self.deform_mapping(z=None, c = latent_embedding)
                deform_triplanes = self.deform_synthesis( ws[:, :self.deform_synthesis.num_ws] )[0]
                triplanes = triplanes + deform_triplanes
            plane_xy, plane_yz, plane_xz = triplanes.chunk(3, dim=0)   # (32,128, 128)
            
            if not self.use_bilinear:          # Nearest neighbor
                grid_coords = grid_coords * (self.triplane_res - 1)
                grid_coords = torch.round(grid_coords).clamp(min=0, max= self.triplane_res).to(torch.int64)
                x, y, z = grid_coords[..., 0], grid_coords[..., 1], grid_coords[...,2]
                feat_xy = plane_xy[:, x, y]
                feat_xz = plane_xz[:, x, z]
                feat_yz = plane_yz[:, y, z]
                feats = torch.cat((feat_xy, feat_yz, feat_xz), dim=0).permute(1, 0, 2)          # (bs, 3*self.triplane_c, n)
                #feats = (feat_xy + feat_yz + feat_xz).permute(1, 0, 2)
            else:
                #feats = self.bilinear_sample_triplanes(grid_coords, plane_xy, plane_yz, plane_xz)
                x, y, z = grid_coords[..., 0:1], grid_coords[..., 1:2], grid_coords[...,2:]
                feat_xy = self.my_2d_bilinear_sample(self.triplane_res, x, y, plane_xy)
                feat_yz = self.my_2d_bilinear_sample(self.triplane_res, y, z, plane_yz)
                feat_xz = self.my_2d_bilinear_sample(self.triplane_res, x, z, plane_xz)
                feats = torch.cat((feat_xy, feat_yz, feat_xz), dim=1)
        else:
            feat_list = []
            for triplane_ in self.triplanes_list:
                plane_xy, plane_yz, plane_xz = triplane_.chunk(3, dim=0)   # (32, res, res)
                x, y, z = grid_coords[..., 0:1], grid_coords[..., 1:2], grid_coords[...,2:]
                res = triplane_.shape[1]
                assert triplane_.shape[1] == triplane_.shape[2]
                feat_xy = self.my_2d_bilinear_sample(res, x, y, plane_xy)
                feat_yz = self.my_2d_bilinear_sample(res, y, z, plane_yz)
                feat_xz = self.my_2d_bilinear_sample(res, x, z, plane_xz)
                feats = torch.cat((feat_xy, feat_yz, feat_xz), dim=1)       # (bs, 3 * channels, n) TODO: xy/yz/xz三维的feat要不要靠在一起
                feat_list.append(feats)
            feats = torch.cat(feat_list, dim=1)

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