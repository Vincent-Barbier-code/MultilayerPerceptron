import numpy as np

def F1_score(y_true, y_pred):
	"""
	F1 score metric.
	"""

	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	return 2 * (precision * recall) / (precision + recall)

def precision_score(y_true, y_pred):
    """
    Precision score metric.
    """

    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)
    if true_positives + false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)

def recall_score(y_true, y_pred):
    """
    Recall score metric.
    """
    
    true_positives = np.sum(y_true * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))
    if true_positives + false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)