from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


def plot_confusion_matrix_reduced(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function is used to print and plot the confusion matrix. Normalization can be applied by setting `normalize=True`

    Parameters
    ----------
    cm : confusion matrix
    classes : list of str
        classes names
    normalize : boolean, optional
        If True, apply normalization. The default is False.
    title : str, optional
        title. The default is 'confusion matrix'.
    cmap : optional
        color map. The default is plt.cm.Blues.

    Returns
    -------
    None.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print('confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    
    
    

    
def plot_confusion_matrix(true_labels,preds_labels):
    """
    This function to prints and plots the confusion matrix and sets its graphical parameters

    Parameters
    ----------
    true_labels : array of str
        the ground_truth labels. We must use arrays with the label in text (not encoded or binarized)
    preds_labels : array of str
        Same as previously, for the predicted labels

    Returns
    -------
    None.

    """
    # let's provide an order for the labels in the futur confusion matrix
    labels_names = list(set(np.unique(preds_labels)).union(set(np.unique(true_labels))))
    
    # Compute the matrix
    cnf_matrix = confusion_matrix(true_labels,preds_labels, labels = labels_names)
    
    # Some parameters :
    np.set_printoptions(precision=2)
    
    plt.figure(figsize=(6, 6))
    
    # plot it
    plot_confusion_matrix_reduced(cnf_matrix, classes=labels_names, normalize=True, title='normalized confusion matrix')

    plt.show()
        


def get_scores(y_true, y_pred):
    """
    This functions provides some metrics for a set of predicted/true lebels

    Parameters
    ----------
    y_true : list/array of str
        the ground_truth labels   
    y_pred : list/array of str
        the predicted labels

    Returns
    -------
    res : dic
        a dictionnary with the metrics

    """
    # we store the metrics in res
    res = {}
    res["Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    # We consider the F1-Score , thus we set beta = 1
    # average = 'weighted' in order to limit the unbalanced dataset
    precision,recall, fbeta_scor, support = precision_recall_fscore_support(y_true, y_pred,beta=1, average='weighted')
    
    # save it
    res["Precision"] = precision
    res["Recall"] = recall
    res["F1_Score"] = fbeta_scor
    res["support"] = support
    
    return res 
        
        
        
def plot_feature_importance(rf, columns):
    """
    plots the importance of each feature/column for a given instance of random forest

    Parameters
    ----------
    rf : instance of RandomForest
        The random forest classificator that we want to investigate on
    columns : list/array of str
        Name of the feature/columns in the training/testing dataset

    Returns
    -------
    None.

    """
    # get it
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=columns)
    
    # plot
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    
    
    