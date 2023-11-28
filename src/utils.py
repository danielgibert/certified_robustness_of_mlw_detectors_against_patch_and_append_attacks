import numpy as np

def count_benign_and_malicious_predictions(predictions: list):
    benign_predictions = sum([1 for pred in predictions if pred < 0.5])
    malicious_predictions = sum([1 for pred in predictions if pred >= 0.5])
    return benign_predictions, malicious_predictions

def get_predicted_label(y_preds: list, min_threshold: float = 0.5):
    if np.sum(y_preds) / len(y_preds) >= min_threshold:
        return 1
    else:
        return 0