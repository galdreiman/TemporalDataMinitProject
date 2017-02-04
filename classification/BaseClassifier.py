import csv
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class BaseClassifier(object):

    def __init__(self, input_filename,disct_num_symb):
        print('init BaseClassifier')

        self.clf = None
        self.disct_num_symb = disct_num_symb

        with open(input_filename, 'r') as input_file:
            reader = csv.reader(input_file)
            self.csv_file = [row for row in reader]

        #split into x/y:
        self.x_train = np.array([row[1:-1] for row in self.csv_file if 'ID' not in row])
        self.y_train = np.array([row[-1] for row in self.csv_file if 'ID' not in row])



    def train(self):
        raise NotImplementedError

    def classify(self, x_test):
        raise NotImplementedError

    def classify_with_CV(self, folds, summary_filename):
        scores = cross_val_score(self.clf, self.x_train, self.y_train, cv=folds)
        print (scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        X_train, X_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=.2, random_state=0)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        kappa_score = cohen_kappa_score(y_test, y_pred)
        print("Cohen Kappa's Score: %f" % kappa_score)

        precision = precision_score(y_test, y_pred, average='macro')
        print("precision: %f" % precision)

        recall = recall_score(y_test, y_pred, average='macro')
        print("recall: %f" % recall)

        F1 = 2 * (precision * recall) / (precision + recall)

        # Accuracy  |  std  |  kappa_score  |  precision  |  recall  |  F1_Score
        return [scores.mean(), scores.std() * 2,kappa_score, precision,  recall, F1]









    # def create_ROC(self,scores):
    #     X_train, X_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=.2, random_state=0)
    #
    #     y_score = self.clf.fit(X_train, y_train).decision_function(X_test)
    #
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #
    #     for i in range(self.disct_num_symb):
    #         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #     # Compute micro-average ROC curve and ROC area
    #     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    #     # ---------------------------------------------------------------------------
    #     # Compute macro-average ROC curve and ROC area
    #
    #     # First aggregate all false positive rates
    #     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.disct_num_symb)]))
    #
    #     # Then interpolate all ROC curves at this points
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(self.disct_num_symb):
    #         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    #     # Finally average it and compute AUC
    #     mean_tpr /= self.disct_num_symb
    #
    #     fpr["macro"] = all_fpr
    #     tpr["macro"] = mean_tpr
    #     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    #     # Plot all ROC curves
    #     plt.figure()
    #     plt.plot(fpr["micro"], tpr["micro"],
    #              label='micro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["micro"]),
    #              color='deeppink', linestyle=':', linewidth=4)
    #
    #     plt.plot(fpr["macro"], tpr["macro"],
    #              label='macro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["macro"]),
    #              color='navy', linestyle=':', linewidth=4)
    #
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #
    #     lw = 2
    #     for i, color in zip(range(self.disct_num_symb), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                  label='ROC curve of class {0} (area = {1:0.2f})'
    #                        ''.format(i, roc_auc[i]))
    #
    #     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Some extension of Receiver operating characteristic to multi-class')
    #     plt.legend(loc="lower right")
    #     plt.show()