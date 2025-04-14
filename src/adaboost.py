import numpy as np
from math import log
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

class AdaBoost:
    def __init__(self, n_rounds, learning_rate=0.5):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.classifiers = []   # store weak learners (decision stumps) 
        self.alphas = []        # store alphas (weights) for each weak learner, calculated based on its error

    # boosting rounds
    def fit(self, X, y, X_val=None, y_val=None):
        n_samples = X.shape[0]
        # initialize weights uniformly for all training samples
        weights = np.full(n_samples, 1 / n_samples)

        for t in tqdm(range(self.n_rounds), desc="Training AdaBoost", unit="round"):
            # create a decision tree using DecisionTreeClassifier with max_depth=1
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X, y, sample_weight=weights)

            # get predictions and compute weighted error
            predictions = clf.predict(X)
            # calculate training error as misclassification rate
            error = np.sum(weights * (predictions != y))

            # calculate alpha (weight) for the classifier
            alpha = self.learning_rate * 0.5 * log((1 - error) / (error + 1e-10))

            # update weights
            # misclassified samples get increased in weight
            weights *= np.exp(-alpha * y * predictions)
            # normalize weights to sum to 1
            weights /= np.sum(weights)

            # store classifier and alpha (its weight)
            self.classifiers.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        aggregated_predictions = np.zeros(X.shape[0])
        # sum the weighted predictions from each classifier
        for clf, alpha in zip(self.classifiers, self.alphas):
            # force predictions to be either 1 or -1
            pred = clf.predict(X)
            pred = np.where(pred >= 0, 1, -1)
            aggregated_predictions += alpha * pred
        # the final prediction is the sign of the aggregated result
        preds = np.sign(aggregated_predictions)
        # force any 0 values to +1
        preds[preds == 0] = 1
        return preds
    
    def predict_scores(self, X):
        aggregated_predictions = np.zeros(X.shape[0])
        for clf, alpha in zip(self.classifiers, self.alphas):
            aggregated_predictions += alpha * clf.predict(X)
        return aggregated_predictions