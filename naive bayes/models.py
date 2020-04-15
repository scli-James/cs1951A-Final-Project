import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a n_examples x n_attributes numpy array
            y_train: a n_examples numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        n_examples = len(y_train)

        self.prior = np.array([0.0] * self.n_classes)
        # calculate prior by counting number of examples in each class
        for i in range(self.n_classes):
            self.prior[i] = (len(y_train[y_train == i]) + 1) / (n_examples + self.n_classes)

        self.attr_distr = np.zeros((len(X_train[0]), self.n_classes))
        # calculate attribute distribution by counting number of attributes in each class
        for i in range(len(X_train[0])):
            for j in range(self.n_classes):
                # for each attribute and for each class, find how many examples fit both, divide by number of examples with that label
                self.attr_distr[i][j] = (np.count_nonzero(y_train[X_train[:, i] == 1] == j) + 1) / (len(y_train[y_train == j]) + 2)

        return self.attr_distr, self.prior

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        output = []
        # for each input, calculate the prediction
        for input in inputs:
            attr_copy = np.copy(self.attr_distr)
            # flip attributes that are not met
            #print(np.size(input))
            #print(np.size(attr_copy, 1))
            attr_copy[input == 0] = 1 - attr_copy[input == 0]
            # multiply all attribute probabilities, then multiply by prior, then find the maximum value
            output.append(np.argmax(np.multiply(np.prod(attr_copy, axis=0), self.prior)))
        return np.array(output)


    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        prediction = self.predict(X_test)
        # return the number of correct predictions divided by number of examples
        print(prediction)
        print(y_test)
        return len(prediction[prediction == y_test]) / len(y_test)

    def print_fairness(self, X_test, y_test, x_sens):
        """
        ***DO NOT CHANGE what we have implemented here.***

        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 1 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit.

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
            x_sens: numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged lasses are
        # grouped together as values of 0 and all privileged classes are given
        # the class 1. . Given data set D = (X,Y, C), with protected
        # attribute X (e.g., race, sex, religion, etc.), remaining attributes Y,
        # and binary class to be predicted C (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8).
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group

        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr
        unpr_fpr = 1 - unpr_tnr

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))


        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
