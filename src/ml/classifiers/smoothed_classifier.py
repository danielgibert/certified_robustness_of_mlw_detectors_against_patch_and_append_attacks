import torch
from torch import nn
from src.ml.ablation_schemes.ablations_generator import AblatedEnd2EndGenerator
import numpy as np


class SmoothedClassifier(nn.Module):
    def __init__(self, base_detector: nn.Module, ablation_scheme: AblatedEnd2EndGenerator):
        super(SmoothedClassifier, self).__init__()

        self.base_detector = base_detector
        self.ablation_scheme = ablation_scheme
        self.out_size = self.base_detector.out_size

    def forward(self, x: torch.Tensor):
        return self.base_detector.forward(x)

    def predict_from_outputs(self, x: torch.Tensor, min_threshold=None, max_threshold=None):
        probs = self.base_detector.get_prob(x)
        labels = self.get_labels(probs, min_threshold=min_threshold, max_threshold=max_threshold)
        return probs, labels

    def get_labels(self, probs, min_threshold=None, max_threshold=None):
        if min_threshold is None and max_threshold is None:
            try:
                labels = torch.IntTensor([1 if prob >= self.base_detector.thresh else 0 for prob in probs])
            except TypeError as te:
                labels = torch.IntTensor([1 if probs.item() >= self.base_detector.thresh else 0])
        else:
            labels = []
            for prob in probs:
                if prob > max_threshold:
                    labels.append(1)
                elif prob < min_threshold:
                    labels.append(0)
                else:
                    labels.append(-1)
            labels = torch.IntTensor(labels)
        return labels


    def predict(self, x: torch.Tensor):
        x_ablated, _ = self.ablation_scheme.generate_ablated_versions(x)
        outputs, _, _ = self.base_detector(x_ablated)
        probs, labels = self.predict_from_outputs(outputs)
        y_score, y_pred, n_counts = self.majority_vote(labels)
        return y_score, y_pred, {"n_counts": n_counts, "probs": probs, "labels": labels}


    def majority_vote(self, y_preds: torch.Tensor, default_vote: int = 1):
        y_preds = y_preds.detach().cpu().numpy()
        counts = np.bincount([x for x in y_preds if x != -1])  # I remove the not accepted predictions
        #print("Number of counts: ", counts)
        y_score = counts[1]/(counts[0]+counts[1])
        ypred = np.argmax(counts)
        return y_score, ypred, counts

    def get_predicted_label(self, y_preds: torch.Tensor, min_threshold: float = 0.5):
        y_preds = y_preds.numpy()
        if np.sum(y_preds) / y_preds.shape[0] >= min_threshold:
            return np.sum(y_preds) / y_preds.shape[0], 1
        else:
            return np.sum(y_preds) / y_preds.shape[0], 0