import torch


def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D IoU between two boxes used for anchor-to-ground-truth matching.

    From: https://colab.research.google.com/github/semilleroCV/deep-learning-notes/blob/main/notebooks/metrics/intersection-over-union.ipynb#scrollTo=FWOemCQ8g0P4


    box1, box2: (4,) tensors in (x1, y1, x2, y2) format
    returns: scalar IoU value
    """
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection

    return intersection / union
if __name__ == "__main__":
    import torch
    print(torch.version.cuda)