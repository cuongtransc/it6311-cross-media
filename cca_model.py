from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.stats import entropy
class CCA_Model:
    def __init__(self,n_components):
        self.n_components = n_components
        self.cca = CCA(n_components=n_components)


    def learn_model(self,X_chanel, Y_chanel,Y_Distinct=None):
        """

        :param X_chanel: array-like for X chanel
        :param Y_chanel: array-line for Y chanel
        :return:

        """
        print "Start learning..."

        self.x_dim  = len(X_chanel[0])
        self.y_dim = len(Y_chanel[0])
        self.cca.fit(X_chanel,Y_chanel)
        if Y_Distinct == None:
            self.X_transform ,self.Y_transform = self.cca.transform(X_chanel,Y_chanel)
        else:
            self.X_transform ,self.Y_transform = self.cca.transform(X_chanel,Y_Distinct)

        print "Learning completed"


    def get_bet_match_index_transform_x2y(self,x_transform):
        shape = self.Y_transform.shape
        scores = np.ndarray(shape[0],dtype=float)
        for i in xrange(shape[0]):
            scores[i] = np.dot(self.Y_transform[i],x_transform)
            #scores[i] = entropy(x_transform,self.Y_transform[i])
        return [np.argmax(scores), np.max(scores)]


    def get_bet_match_index_transform_y2x(self,y_transform):
        shape = self.X_transform.shape
        scores = np.ndarray(shape[0], dtype=float)
        for i in xrange(shape[0]):
            scores[i] = np.dot(self.X_transform[i], y_transform)
            #scores[i] = entropy(y_transform,self.X_transform[i])
        return [np.argmax(scores), np.max(scores)]

    def get_best_match_cross_indices_x2y(self,x_inputs):
        x_transformes = self.cca.transform(x_inputs)
        results = []
        for x_transform in x_transformes:
            results.append(self.get_bet_match_index_transform_x2y(x_transform))
        return results

    def get_best_match_cross_indices_y2x(self,y_inputs):
        _, y_transformes = self.cca.transform([[0 for i in xrange(self.x_dim)]],y_inputs)
        results = []
        for y_transform in y_transformes:
            results.append(self.get_bet_match_index_transform_y2x(y_transform))
        return results





