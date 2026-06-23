"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    """UNet-style upsampling block that fuses a low-resolution feature map with a
    higher-resolution skip connection via bilinear upsampling followed by two
    Conv-BN-ReLU layers."""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        """Build the upsample layer and the two-stage conv block.

        Args:
            in_channels: Total channels after concatenating the upsampled tensor
                with the skip connection (i.e. upsampled_C + skip_C).
            out_channels: Number of output channels produced by the conv block.
            scale_factor: Spatial magnification factor for bilinear upsampling.
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        """Upsample x1 to match x2's spatial size, concatenate along the channel
        dimension, then refine through the conv block.

        Args:
            x1: Lower-resolution tensor to upsample (B, C1, H, W).
            x2: Higher-resolution skip-connection tensor (B, C2, H*scale, W*scale).

        Returns:
            Fused feature map of shape (B, out_channels, H*scale, W*scale).
        """
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    """Camera feature encoder that lifts each 2-D image pixel into a D-bin depth
    distribution and a C-dimensional feature vector, producing a 3-D feature
    volume per camera via an outer product (the "Lift" step of LSS).

    Architecture: EfficientNet-B0 backbone → FPN-style Up block → 1×1 depthnet
    head that outputs D + C channels simultaneously.
    """

    def __init__(self, D, C, downsample):
        """Initialise the EfficientNet backbone, FPN Up block, and depthnet head.

        Args:
            D: Number of discrete depth bins in the frustum.
            C: Number of feature channels per depth bin.
            downsample: Spatial downsampling factor of the backbone output
                relative to the input image (used externally; stored here for
                reference).
        """
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        """Convert raw depth logits to a probability distribution over D bins
        using a softmax along the depth dimension (dim=1).

        Args:
            x: Raw depth logits of shape (B, D, H, W).

        Returns:
            Depth probabilities of shape (B, D, H, W) summing to 1 along dim=1.
        """
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        """Run the full encode pipeline: backbone → depthnet → outer product.

        The depthnet output is split into depth logits (first D channels) and
        context features (next C channels). The depth distribution and context
        are combined via an outer product so that each spatial location produces
        a D×C feature volume — every depth bin gets a scaled copy of the context
        feature weighted by its depth probability.

        Args:
            x: Batch of images, shape (B, 3, H, W).

        Returns:
            depth: Depth probability map, shape (B, D, fH, fW).
            new_x: Lifted feature volume, shape (B, C, D, fH, fW).
        """
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        """Extract multi-scale features from EfficientNet-B0 and fuse them with
        the Up block to produce a single high-resolution feature map.

        Runs the EfficientNet stem and all MBConv blocks, recording the feature
        map just before each spatial downsampling step as a 'reduction' endpoint.
        The two deepest reduction maps (reduction_4 at 112 ch and reduction_5 at
        320 ch) are fused by the Up block to yield a 512-channel output at the
        resolution of reduction_4.

        Args:
            x: Batch of images, shape (B, 3, H, W).

        Returns:
            Fused feature map of shape (B, 512, H/downsample, W/downsample).
        """
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate if self.trunk._global_params is not None else 0.0
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        """Encode a batch of camera images into lifted 3-D feature volumes.

        Args:
            x: Batch of images, shape (B, 3, H, W).

        Returns:
            Lifted feature volume of shape (B, C, D, fH, fW).
        """
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    """BEV (Bird's-Eye View) feature encoder built on a ResNet-18 backbone.

    Takes the collapsed BEV feature map produced by voxel pooling and refines it
    through three ResNet stages followed by two upsampling stages (the "Shoot"
    step of LSS) to produce a dense BEV segmentation map at the original
    BEV grid resolution.
    """

    def __init__(self, inC, outC):
        """Borrow the first three ResNet-18 residual stages and append two decoder
        stages to restore spatial resolution.

        Args:
            inC: Number of input channels in the BEV feature map (equals camC *
                number of Z slices after collapsing the Z dimension).
            outC: Number of output segmentation channels.
        """
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        """Encode a BEV feature map into a dense prediction map.

        Passes x through the ResNet stem and three residual stages, then decodes
        back to the original spatial resolution using two upsampling stages. The
        layer1 output is used as a skip connection in up1 to recover fine-grained
        spatial structure lost during downsampling.

        Args:
            x: BEV feature map of shape (B, inC, H, W).

        Returns:
            Dense BEV prediction map of shape (B, outC, H, W).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    """Full Lift-Splat-Shoot model for camera-only BEV perception.

    Implements the three-stage pipeline from Philion & Fidler (2020):
      1. Lift  – each image pixel is lifted into a 3-D feature cloud using a
                 predicted per-pixel depth distribution (CamEncode).
      2. Splat – the per-camera feature clouds are pooled into a shared BEV
                 voxel grid via voxel_pooling, then the Z dimension is collapsed.
      3. Shoot – the flat BEV grid is passed through BevEncode to produce the
                 final output (e.g. segmentation map).
    """

    def __init__(self, grid_conf, data_aug_conf, outC):
        """Build the voxel grid parameters, frustum, and sub-networks.

        Args:
            grid_conf: Dict with keys 'xbound', 'ybound', 'zbound', 'dbound'
                each a (min, max, step) tuple defining the BEV and depth grids.
            data_aug_conf: Dict containing at least 'final_dim' (H, W) — the
                spatial size of the input images after augmentation.
            outC: Number of output channels for the BEV segmentation head.
        """
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        """Build a 3-D frustum grid in pixel + depth space at the downsampled
        image resolution.

        For each combination of (depth bin, row, column) in the downsampled
        image grid, stores the corresponding (x_pixel, y_pixel, depth) triplet.
        This grid is fixed across the whole training run and is used in
        get_geometry to back-project every frustum point into the ego frame.

        Returns:
            nn.Parameter of shape (D, fH, fW, 3) where D is the number of depth
            bins and fH, fW are the downsampled image dimensions. Not a trainable
            parameter (requires_grad=False).
        """
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Map every frustum point from pixel space to the ego (vehicle) frame.

        Each frustum point starts as (u, v, d) in augmented pixel coordinates.
        The transform chain is applied in reverse order:
          1. Undo post-augmentation (post_rots, post_trans) to get back to the
             original, un-augmented pixel coordinates.
          2. Convert from homogeneous pixel coords to a 3-D camera-space ray by
             multiplying by the depth d (i.e. x_cam = K^{-1} * [u*d, v*d, d]).
          3. Rotate and translate from camera frame to ego frame using (rots, trans).

        Args:
            rots:       Camera-to-ego rotation matrices,    (B, N, 3, 3).
            trans:      Camera-to-ego translation vectors,  (B, N, 3).
            intrins:    Camera intrinsic matrices,           (B, N, 3, 3).
            post_rots:  Post-augmentation rotation matrices, (B, N, 3, 3).
            post_trans: Post-augmentation translations,      (B, N, 3).

        Returns:
            Ego-frame 3-D coordinates for every frustum point,
            shape (B, N, D, H/downsample, W/downsample, 3).
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Encode all camera images and reshape the output into a per-camera
        feature volume indexed by (batch, camera, depth, row, col, channel).

        Flattens the batch and camera dimensions so all images can be processed
        in a single CamEncode forward pass, then reshapes and permutes the result
        back to (B, N, D, fH, fW, C) for compatibility with voxel_pooling.

        Args:
            x: Multi-camera image tensor of shape (B, N, 3, imH, imW).

        Returns:
            Per-camera lifted feature volumes, shape (B, N, D, fH, fW, C).
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """Splat lifted camera features into the BEV voxel grid (the "Splat" step).

        Each frustum point carries a C-dimensional feature and an ego-frame 3-D
        coordinate. This method:
          1. Converts continuous ego coordinates to integer voxel indices.
          2. Discards points that fall outside the configured BEV bounds.
          3. Sorts all remaining points by a rank that groups same-voxel points
             together.
          4. Sums features within each voxel using either the differentiable
             cumsum trick or the faster QuickCumsum CUDA kernel.
          5. Scatters the summed features into a dense (B, C, Z, X, Y) grid,
             then collapses the Z dimension by concatenating Z slices along
             the channel axis, yielding a flat (B, C*Z, X, Y) BEV map.

        Args:
            geom_feats: Ego-frame voxel coordinates, shape (B, N, D, H, W, 3).
            x:          Lifted feature volumes,       shape (B, N, D, H, W, C).

        Returns:
            Flat BEV feature map of shape (B, C * nZ, nX, nY).
        """
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        final = torch.zeros((B, C, int(self.nx[2]), int(self.nx[0]), int(self.nx[1])), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """Run the Lift and Splat stages to produce a flat BEV feature map.

        Combines get_geometry, get_cam_feats, and voxel_pooling in sequence:
        camera images are encoded into frustum feature volumes, the geometry
        transform maps each frustum point to the ego frame, and voxel_pooling
        accumulates the features into the BEV grid.

        Args:
            x:          Multi-camera images, shape (B, N, 3, imH, imW).
            rots:       Camera-to-ego rotations,          (B, N, 3, 3).
            trans:      Camera-to-ego translations,        (B, N, 3).
            intrins:    Camera intrinsics,                 (B, N, 3, 3).
            post_rots:  Post-augmentation rotations,       (B, N, 3, 3).
            post_trans: Post-augmentation translations,    (B, N, 3).

        Returns:
            Flat BEV feature map of shape (B, C * nZ, nX, nY).
        """
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """Run the complete Lift-Splat-Shoot pipeline end to end.

        Calls get_voxels (Lift + Splat) to produce a flat BEV feature map, then
        passes it through BevEncode (Shoot) to generate the final dense output.

        Args:
            x:          Multi-camera images, shape (B, N, 3, imH, imW).
            rots:       Camera-to-ego rotations,          (B, N, 3, 3).
            trans:      Camera-to-ego translations,        (B, N, 3).
            intrins:    Camera intrinsics,                 (B, N, 3, 3).
            post_rots:  Post-augmentation rotations,       (B, N, 3, 3).
            post_trans: Post-augmentation translations,    (B, N, 3).

        Returns:
            Dense BEV output map of shape (B, outC, nX, nY).
        """
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    """Instantiate and return a LiftSplatShoot model.

    Args:
        grid_conf: Dict with 'xbound', 'ybound', 'zbound', 'dbound' tuples
            (min, max, step) defining the BEV and depth grids.
        data_aug_conf: Dict containing at least 'final_dim' (H, W).
        outC: Number of output channels for the BEV head.

    Returns:
        A LiftSplatShoot nn.Module ready for training or inference.
    """
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
