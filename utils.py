# Import packages
import os
import warnings

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def evaluate(predict, validation_labels, format='report'):
    """
    Evaluate predicted labels and true label through reporting about some metrics

    Args:
        predict (RDD): labels are predicted from model
        validation_labels (RDD): true labels
    """
    y_true, y_pred = validation_labels.collect(), predict.collect()
    if format == 'report':
        return classification_report(y_true, y_pred)
    else:   # metric f1 score
        return f1_score(y_true, y_pred)

def show_confusion_matrix(predict, validation_labels):
    """Plot confusion matrix."""
    # Retrieve label from RDD.
    y_true, y_pred = validation_labels.collect(), predict.collect()

    # Get confusion matrix
    cf = np.array(confusion_matrix(y_true, y_pred))
    fig, ax = plt.subplots()
    im = ax.imshow(cf)

    # We want to show all ticks...
    ax.set_xticks(np.arange(cf.shape[1]))
    ax.set_yticks(np.arange(cf.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['True 0', 'True 1'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            text = ax.text(j, i, cf[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()
