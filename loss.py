import torch
import torch.nn as nn
from torchvision.ops import box_iou



class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        iou = box_iou(pred, target).diag()
        return 1 - iou.mean()


class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Pred & target: (B, 4) - [x1, y1, x2, y2]
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        # Convert to box coords
        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        iw = (ix2 - ix1).clamp(0)
        ih = (iy2 - iy1).clamp(0)
        inter = iw * ih

        # Areas
        pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou = inter / union

        # Center distance
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        center_dist = (pcx - tcx)**2 + (pcy - tcy)**2

        # Enclosing box
        ex1 = torch.min(px1, tx1)
        ey1 = torch.min(py1, ty1)
        ex2 = torch.max(px2, tx2)
        ey2 = torch.max(py2, ty2)
        c_diag = (ex2 - ex1)**2 + (ey2 - ey1)**2 + 1e-7

        # Aspect ratio consistency
        pw = (px2 - px1).clamp(min=1e-6)
        ph = (py2 - py1).clamp(min=1e-6)
        tw = (tx2 - tx1).clamp(min=1e-6)
        th = (ty2 - ty1).clamp(min=1e-6)

        v = (4 / (3.14159 ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph))**2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - (center_dist / c_diag) - alpha * v
        loss = 1 - ciou.mean()
        return loss




class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Clamp to [0, 1]
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        iw = (ix2 - ix1).clamp(0)
        ih = (iy2 - iy1).clamp(0)
        inter = iw * ih

        # Areas
        pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou = inter / union

        # Enclosing box
        ex1 = torch.min(px1, tx1)
        ey1 = torch.min(py1, ty1)
        ex2 = torch.max(px2, tx2)
        ey2 = torch.max(py2, ty2)
        ew = (ex2 - ex1).clamp(0)
        eh = (ey2 - ey1).clamp(0)
        enclosing_area = ew * eh + 1e-7

        giou = iou - (enclosing_area - union) / enclosing_area
        loss = 1 - giou.mean()
        return loss


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return nn.MSELoss()(pred, target)


class SmoothL1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return nn.SmoothL1Loss()(pred, target)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return nn.L1Loss()(pred, target)


class WingLoss(nn.Module):
    def __init__(self, w=10, eps=2):
        super().__init__()
        self.w = w
        self.eps = eps
        self.c = w * (1.0 - torch.log(torch.tensor(1.0 + w / eps)))

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1.0 + abs_x / self.eps),
            abs_x - self.c.to(abs_x.device)  # make sure it's on the right device
        )
        return loss.mean()


def init_loss_fn(loss_fn):
    if loss_fn == "iou":
        criterion = IoULoss()
    elif loss_fn == "giou":
        criterion = GIoULoss()
    elif loss_fn == "ciou":
        criterion = CIoULoss()
    elif loss_fn == "mse":
        criterion = MSE()
    elif loss_fn == "smoothl1":
        criterion = SmoothL1()
    elif loss_fn == "WingLoss":
        criterion = WingLoss()
    elif loss_fn == "L1Loss":
        criterion = L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")
    return criterion