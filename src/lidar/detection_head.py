from torch import nn

class SSD(nn.Module):
    """
    From https://www.geeksforgeeks.org/computer-vision/how-single-shot-detector-ssd-works/
    """
    def __init__(self, in_channels: int=384, num_classes: int=3, num_anchors: int=2):
        """
        C:            feature channels from PointNet encoder
        in_channels:  backbone multiplier (default 6, giving 6*C input channels)
        num_classes:  number of object categories (e.g. car, pedestrian, cyclist)
        num_anchors:  anchors per BEV cell (default 2 for 0° and 90° rotations)

        reg_head predicts 7 values per anchor: (x, y, z, w, l, h, θ)
        cls_head predicts num_classes scores per anchor
        """
        super().__init__()
        self.num_classes = num_classes

        self.reg_head = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        cls = self.cls_head(x)  # (B, num_anchors * num_classes, H, W)
        reg = self.reg_head(x)  # (B, num_anchors * 7, H, W)
        return cls, reg