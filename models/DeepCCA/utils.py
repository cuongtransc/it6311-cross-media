import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import theano
from keras.utils.data_utils import get_file


def load_data(data_file, url):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def get_best_match_index_transform_x2y(X = x_transform, Y = Y_transform):
    shape = Y.shape
    scores = np.ndarray(shape[0],dtype=float)
    for i in xrange(shape[0]):
        scores[i] = np.dot(Y[i],x_transform)
        #scores[i] = entropy(x_transform,self.Y_transform[i])
    return [np.argmax(scores), np.max(scores)]


def get_best_match_index_transform_y2x(self,y_transform):
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
        results.append(self.get_best_match_index_transform_x2y(x_transform))
    return results

def get_best_match_cross_indices_y2x(self,y_inputs):
    _, y_transformes = self.cca.transform([[0 for i in xrange(self.x_dim)]],y_inputs)
    results = []
    for y_transform in y_transformes:
        results.append(self.get_best_match_index_transform_y2x(y_transform))
    return results
