from torch import nn

class SSD(nn.Module):
    """
    From https://www.geeksforgeeks.org/computer-vision/how-single-shot-detector-ssd-works/
    """
    def __init__(self, in_channels: int=384, num_classes: int=3, num_anchors: int=2, num_attrs: int=8):
        """
        C:            feature channels from PointNet encoder
        in_channels:  backbone multiplier (default 6, giving 6*C input channels)
        num_classes:  number of object categories (e.g. car, pedestrian, cyclist)
        num_anchors:  anchors per BEV cell (default 2 for 0° and 90° rotations)
        num_attrs:    size of the shared attribute vocabulary (nuScenes ATTR_NAMES);
                      each anchor's fixed home class picks out a slice of this at
                      decode time (see common.ATTR_CLASS_RANGES) — not one head per class

        reg_head predicts 9 values per anchor: (x, y, z, w, l, h, θ, vx, vy)
        cls_head predicts num_classes scores per anchor
        attr_head predicts num_attrs scores per anchor
        """
        super().__init__()
        self.num_classes = num_classes

        self.reg_head  = nn.Conv2d(in_channels, num_anchors * 9, kernel_size=3, padding=1)
        self.cls_head  = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.attr_head = nn.Conv2d(in_channels, num_anchors * num_attrs, kernel_size=3, padding=1)

    def forward(self, x):
        cls  = self.cls_head(x)   # (B, num_anchors * num_classes, H, W)
        reg  = self.reg_head(x)   # (B, num_anchors * 9, H, W)
        attr = self.attr_head(x)  # (B, num_anchors * num_attrs, H, W)
        return cls, reg, attr