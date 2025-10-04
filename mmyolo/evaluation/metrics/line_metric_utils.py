"""
Created on Tue Jan 16 16:42:29 2024

@author: zcf
"""
import numpy as np
import sys

THRESHOLDS_FRECHET = [float(x) for x in range(10, 201, 5)]
# copy from openlanev2/centerline/evaluation/evaluate.py: https://github.com/OpenDriveLab/OpenLane-V2
def _pr_curve(recalls, precisions):
    r"""
    Calculate average precision based on given recalls and precisions.

    Parameters
    ----------
    recalls : array_like
        List in shape (N, ).
    precisions : array_like
        List in shape (N, ).

    Returns
    -------
    float
        average precision
    
    Notes
    -----
    Adapted from https://github.com/Mrmoore98/VectorMapNet_code/blob/mian/plugin/datasets/evaluation/precision_recall/average_precision_gen.py#L12

    """
    recalls = np.asarray(recalls)[np.newaxis, :]
    precisions = np.asarray(precisions)[np.newaxis, :]

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)

    for i in range(num_scales):
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[i, recalls[i, :] >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap[i] += prec
    ap /= 11

    return ap[0]

# copy from openlanev2/centerline/evaluation/evaluate.py: https://github.com/OpenDriveLab/OpenLane-V2
def _tpfp(gts, preds, confidences, distance_matrix, distance_threshold):
    r"""
    Generate lists of tp and fp on given distance threshold.

    Parameters
    ----------
    gts : List
        List of groud truth in shape (G, ).
    preds : List
        List of predictions in shape (P, ).
    confidences : array_like
        List of float in shape (P, ).
    distance_matrix : array_like
        Distance between every pair of instances.
    distance_threshold : float
        Predictions are considered as valid within the distance threshold.

    Returns
    -------
    (array_like, array_like, array_like)
        (tp, fp, match) both in shape (P, ).

    Notes
    -----
    Adapted from https://github.com/Mrmoore98/VectorMapNet_code/blob/mian/plugin/datasets/evaluation/precision_recall/tgfg.py#L10.

    """
    assert len(preds) == len(confidences)

    num_gts = len(gts)
    num_preds = len(preds)

    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)
    idx_match_gt = np.ones((num_preds), dtype=int) * np.nan

    if num_gts == 0:
        fp[...] = 1
        return tp, fp, idx_match_gt
    if num_preds == 0:
        return tp, fp, idx_match_gt

    dist_min = distance_matrix.min(0)
    dist_idx = distance_matrix.argmin(0)

    confidences_idx = np.argsort(-np.asarray(confidences))
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in confidences_idx:
        if dist_min[i] < distance_threshold:
            matched_gt = dist_idx[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
                idx_match_gt[i] = matched_gt
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    return tp, fp, idx_match_gt
# copy from openlanev2/centerline/evaluation/evaluate.py: https://github.com/OpenDriveLab/OpenLane-V2
def _inject(num_gt, pred, tp, idx_match_gt, confidence, distance_threshold, object_type):
    r"""
    Inject tp matching into predictions.

    Parameters
    ----------
    num_gt : int
        Number of ground truth.
    pred : dict
        Dict storing predictions for all samples,
        to be injected.
    tp : array_like
    idx_match_gt : array_like
    confidence : array_lick
    distance_threshold : float
        Predictions are considered as valid within the distance threshold.
    object_type : str
        To filter type of object for evaluation.

    """
    if tp.tolist() == []:
        pred[f'{object_type}_{distance_threshold}_idx_match_gt'] = []
        pred[f'{object_type}_{distance_threshold}_confidence'] = []
        pred[f'{object_type}_{distance_threshold}_confidence_thresholds'] = []
        return

    confidence = np.asarray(confidence)
    sorted_idx = np.argsort(-confidence)
    sorted_confidence = confidence[sorted_idx]
    tp = tp[sorted_idx]

    tps = np.cumsum(tp, axis=0)
    eps = np.finfo(np.float32).eps
    recalls = tps / np.maximum(num_gt, eps)

    taken = np.percentile(recalls, np.arange(10, 101, 10), method='closest_observation') # zhou numpy should >=1.22, that require Python >= 3.8.
    taken_idx = {r: i for i, r in enumerate(recalls)}
    confidence_thresholds = sorted_confidence[np.asarray([taken_idx[t] for t in taken])]

    pred[f'{object_type}_{distance_threshold}_idx_match_gt'] = idx_match_gt
    pred[f'{object_type}_{distance_threshold}_confidence'] = confidence
    pred[f'{object_type}_{distance_threshold}_confidence_thresholds'] = confidence_thresholds
# copy from openlanev2/centerline/evaluation/evaluate.py: https://github.com/OpenDriveLab/OpenLane-V2
def _AP(gts, preds, distance_matrixs, distance_threshold, object_type, filter, inject):
    r"""
    Calculate AP on given distance threshold.

    Parameters
    ----------
    gts : dict
        Dict storing ground truth for all samples.
    preds : dict
        Dict storing predictions for all samples.
    distance_matrixs : dict
        Dict storing distance matrix for all samples.
    distance_threshold : float
        Predictions are considered as valid within the distance threshold.
    object_type : str
        To filter type of object for evaluation.
    filter : callable
        To filter objects for evaluation.
    inject : bool
        Inject or not.

    Returns
    -------
    float
        AP over all samples.

    """
    tps = []
    fps = []
    confidences = []
    num_gts = 0
    for token in gts.keys():
        gt = [gt['points'] for gt in gts[token][object_type] if filter(gt)]
        pred = [pred['points'] for pred in preds[token][object_type] if filter(pred)]
        confidence = [pred['confidence'] for pred in preds[token][object_type] if filter(pred)]
        filtered_distance_matrix = distance_matrixs[token].copy()
        filtered_distance_matrix = filtered_distance_matrix[[filter(gt) for gt in gts[token][object_type]], :]
        filtered_distance_matrix = filtered_distance_matrix[:, [filter(pred) for pred in preds[token][object_type]]]
        tp, fp, idx_match_gt = _tpfp(
            gts=gt, 
            preds=pred, 
            confidences=confidence, 
            distance_matrix=filtered_distance_matrix, 
            distance_threshold=distance_threshold,
        )
        tps.append(tp)
        fps.append(fp)
        confidences.append(confidence)
        num_gts += len(gt)
        # 获取当前Python版本
        python_version = sys.version_info
        # 检查是否大于等于 Python 3.8
        if python_version >= (3, 8):
            if inject:
                _inject(
                    num_gt=len(gt),
                    pred=preds[token],
                    tp=tp,
                    idx_match_gt=idx_match_gt,
                    confidence=confidence,
                    distance_threshold=distance_threshold,
                    object_type=object_type,
                )

    confidences = np.hstack(confidences)
    sorted_idx = np.argsort(-confidences)
    tps = np.hstack(tps)[sorted_idx]
    fps = np.hstack(fps)[sorted_idx]

    if len(tps) == num_gts == 0:
        return np.float32(1)

    tps = np.cumsum(tps, axis=0)
    fps = np.cumsum(fps, axis=0)
    eps = np.finfo(np.float32).eps
    recalls = tps / np.maximum(num_gts, eps)
    precisions = tps / np.maximum((tps + fps), eps)
    return _pr_curve(recalls=recalls, precisions=precisions)
# copy from openlanev2/centerline/evaluation/evaluate.py: https://github.com/OpenDriveLab/OpenLane-V2
def _mAP_over_threshold(gts, preds, distance_matrixs, distance_thresholds, object_type, filter, inject):
    r"""
    Calculate mAP over distance thresholds.

    Parameters
    ----------
    gts : dict
        Dict storing ground truth for all samples.
    preds : dict
        Dict storing predictions for all samples.
    distance_matrixs : dict
        Dict storing distance matrix for all samples.
    distance_thresholds : list
        Distance thresholds.
    object_type : str
        To filter type of object for evaluation.
    filter : callable
        To filter objects for evaluation.
    inject : bool
        Inject or not.

    Returns
    -------
    list
        APs over all samples.

    """
    return np.asarray([_AP(
        gts=gts, 
        preds=preds, 
        distance_matrixs=distance_matrixs,
        distance_threshold=distance_threshold, 
        object_type=object_type,
        filter=filter,
        inject=inject,
    ) for distance_threshold in distance_thresholds])    

