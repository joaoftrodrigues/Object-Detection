import numpy as np

def is_between(value, min, max):
    """
    Check if a number is between 2 given numbers,
    including
    :param value: number to test
    :param min: min value of interval
    :param max: max value of interval
    :return:
    """

    # Value comparison between min and max
    if min <= value <= max:
        return True

    # In case previous condition fails,
    # value is outside of range
    return False


def axis_intercept(pr_min, pr_max, gt_min, gt_max):

    # Minimal value of Grand Truth between predicted interval
    if pr_min <= gt_min <= pr_max:
        return True

    # Minimal value of Predicted between Grand Truth interval
    elif gt_min <= pr_min <= gt_max:
        return True

    else:
        return False


def boxes_intercept(previsioned_box, gt_box):
    """
    Check if boxes intercept, based on x and y values
    :param previsioned_box:
    :param gt_box:
    :return: Boolean result for boxes interception
    """

    # Decomposal of predicted box
    pr_x1 = previsioned_box[0]
    pr_x2 = previsioned_box[2]
    pr_y1 = previsioned_box[1]
    pr_y2 = previsioned_box[3]

    # Decomposal of ground truth box
    gt_x1 = gt_box[0]
    gt_x2 = gt_box[2]
    gt_y1 = gt_box[1]
    gt_y2 = gt_box[3]

    # In case there is no interception, area won't be calculated as is 0
    #return x_intercepts(previsioned_box, gt_box) and y_intercepts(previsioned_box, gt_box)

    x_intercepts = axis_intercept(pr_x1, pr_x2, gt_x1, gt_x2)
    y_intercepts = axis_intercept(pr_y1, pr_y2, gt_y1, gt_y2)

    return x_intercepts and y_intercepts


def calc_iou(previsioned_box, gt_box):
    """
    Calculates Interception of Union between box previsioned
    and ground truth box
    :param previsioned_box:
    :param gt_box:
    :return:
    """

    # In case there is no interception, area won't be calculated as is 0
    if not boxes_intercept(previsioned_box, gt_box):
        return 0

    # coordinates of the area of intersection.
    ix1 = np.maximum(gt_box[0], previsioned_box[0])
    iy1 = np.maximum(gt_box[1], previsioned_box[1])
    ix2 = np.minimum(gt_box[2], previsioned_box[2])
    iy2 = np.minimum(gt_box[3], previsioned_box[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = gt_box[3] - gt_box[1] + 1
    gt_width = gt_box[2] - gt_box[0] + 1

    # Prediction dimensions.
    pd_height = previsioned_box[3] - previsioned_box[1] + 1
    pd_width = previsioned_box[2] - previsioned_box[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


def calc_iou_with_gt_boxes(previsioned_box, img_gt_boxes):
    """

    :param previsioned_box:
    :param img_gt_boxes:
    :return:
    """

    iou_results = [calc_iou(previsioned_box, gt_box) for gt_box in img_gt_boxes]

    return iou_results


def get_precision(tp, fp):
    """
    Calculus of precision, given number of True Positives
    and False Positives
    :param tp: number of True Positives
    :param fp: number of False Positives
    :return:
    """

    # Sum all previsions
    total = tp + fp

    if total == 0:
        return 0

    # Calculus of precision
    precision = tp / total

    return precision


def get_recall(tp, gt_boxes):
    """
    Calculate Recall bases on number of True Positives
    and Ground Truth boxes
    :param tp: Number of True Positives
    :param gt_boxes: List of bounding boxes
    :return:
    """

    # Quantity of bounding boxes
    n_gt_boxes = len(gt_boxes)

    # Calculus of Recall
    recall = tp / n_gt_boxes

    return recall
