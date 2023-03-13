from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def plot_confusion_matrix_reduced(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
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
    # must receive arrays with the labels in full text
    labels_names = list(set(np.unique(preds_labels)).union(set(np.unique(true_labels))))
    
    cnf_matrix = confusion_matrix(true_labels,preds_labels, labels = labels_names)
    np.set_printoptions(precision=2)
    
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix_reduced(cnf_matrix, classes=labels_names, normalize=True, title='normalized confusion matrix')

    plt.show()
        


def get_scores(y_true, y_pred):
    res = {}
    res["Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    precision,recall, fbeta_scor, support = precision_recall_fscore_support(y_true, y_pred,beta=1, average='weighted')
    # beta = 1 for the F1_score
    res["Precision"] = precision
    res["Recall"] = recall
    res["F1_Score"] = fbeta_scor
    res["support"] = support
    
    return res 
        
        
        
        
    