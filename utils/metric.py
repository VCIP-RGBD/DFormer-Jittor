"""
Evaluation metrics for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import numpy as np
from collections import OrderedDict


class SegmentationMetric(object):
    """Segmentation evaluation metrics."""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """Update metrics with new predictions and targets."""
        pred = pred.astype(np.int64)
        target = target.astype(np.int64)
        
        # Mask out ignore index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum((target == i) & (pred == j))
    
    def get_results(self):
        """Calculate and return all metrics."""
        results = OrderedDict()
        
        # Per-class IoU
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            if tp + fp + fn == 0:
                iou = 0.0
            else:
                iou = tp / (tp + fp + fn)
            iou_per_class.append(iou)
        
        results['IoU_per_class'] = iou_per_class
        results['mIoU'] = np.mean(iou_per_class)
        
        # Per-class accuracy
        acc_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total = np.sum(self.confusion_matrix[i, :])
            
            if total == 0:
                acc = 0.0
            else:
                acc = tp / total
            acc_per_class.append(acc)
        
        results['Acc_per_class'] = acc_per_class
        results['mAcc'] = np.mean(acc_per_class)
        
        # Overall accuracy
        total_correct = np.sum(np.diag(self.confusion_matrix))
        total_pixels = np.sum(self.confusion_matrix)
        results['Overall_Acc'] = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        # Frequency weighted IoU
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        results['FWIoU'] = np.sum(freq * iou_per_class)
        
        return results
    
    def get_confusion_matrix(self):
        """Get confusion matrix."""
        return self.confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    """Calculate intersection and union for IoU computation."""
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """GPU version of intersection and union calculation."""
    # Convert to numpy for computation
    if hasattr(output, 'numpy'):
        output = output.numpy()
    if hasattr(target, 'numpy'):
        target = target.numpy()
    
    return intersectionAndUnion(output, target, K, ignore_index)


class Evaluator(object):
    """Evaluator for semantic segmentation."""
    
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        """Calculate pixel accuracy."""
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        """Calculate per-class pixel accuracy."""
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        """Calculate mean IoU."""
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        """Calculate frequency weighted IoU."""
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """Generate confusion matrix."""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """Add batch to evaluation."""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def calculate_iou(pred, target, num_classes, ignore_index=255):
    """Calculate IoU for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        if target_inds.long().sum().item() == 0:
            ious.append(float('nan'))
        else:
            intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
            union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
            ious.append(float(intersection) / float(max(union, 1)))
    
    return ious


def fast_hist(a, b, n):
    """Fast histogram computation for confusion matrix."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """Calculate per-class IoU from histogram."""
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    """Map labels according to mapping dictionary."""
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)
