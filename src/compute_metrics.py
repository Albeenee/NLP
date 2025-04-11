import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calcul de l'accuracy
    accuracy = np.sum(predictions == labels) / len(labels)

    # Calcul du F1 score
    unique_labels = np.unique(labels)
    f1 = 0
    for label in unique_labels:
        tp = np.sum((predictions == label) & (labels == label))
        fp = np.sum((predictions == label) & (labels != label))
        fn = np.sum((predictions != label) & (labels == label))
        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 += 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    f1 /= len(unique_labels)

    return {
        "accuracy": accuracy,
        "f1": f1,
    }