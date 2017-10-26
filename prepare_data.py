import numpy as np
import os.path

NUM_TAGS = 81

IMAGE_FEATURE_TRAIN_PATH = "NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_WT_Lite_Train.dat"
IMAGE_FEATURE_TEST_PATH = "NUS-WIDE-Lite/NUS-WIDE-Lite_features/Normalized_WT_Lite_Test.dat"

TAG_TRAIN_PATH = "NUS-WIDE-Lite/NUS-WIDE-Lite_tags/Lite_Tags{}_Train.txt".format(NUM_TAGS)
TAG_TEST_PATH = "NUS-WIDE-Lite/NUS-WIDE-Lite_tags/Lite_Tags{}_Test.txt".format(NUM_TAGS)

TAG81_NAMES_PATH = "NUS-WIDE-Lite/Concepts81.txt"
def load_train_img_feature_vectors(IMAGE_FEATURE_TRAIN_PATH):
    """
    Loading features of images from file
    :return:
    """
    print "Loading train features vectors ..."
    features =  np.loadtxt(IMAGE_FEATURE_TRAIN_PATH,dtype=float)
    return features

def load_tag_concepts81():
    """
    Loading tag's names from file
    :return:
    """
    print "Loading concepts 81"
    f = open(TAG81_NAMES_PATH,"r")
    tag_2_id = dict()
    id_2_tag = dict()
    id = 0
    while True:
        line = f.readline()
        if line == "":
            break
        tag = line.strip()
        tag_2_id[tag] = id
        id_2_tag[id] = tag
        id += 1
    f.close()
    assert id == NUM_TAGS
    return tag_2_id, id_2_tag


def load_glove_from_file(path,tag_2_id):
    """
    Loading map tag_to_vector from pre-train w2v glove data and valid_tags
    :param path:
    :param tag_2_id:
    :return:
    """
    f = open(path,"r")
    word_2_vec = dict()
    while True:
        line = f.readline()
        if line == "":
            break
        parts = line.strip().split(" ")
        word = parts[0]
        try:
            word_id = tag_2_id[word]
        except:
            continue
        word_vector_s =  parts[1:]
        x = np.array(word_vector_s)
        x = np.asfarray(x, float)
        word_2_vec[word] = x

        if len(word_2_vec) == NUM_TAGS:
            break
    f.close()
    assert len(word_2_vec) == NUM_TAGS
    return word_2_vec

def convert_wvec_2_widvec(word2vec, word2id):
    """
    Converting word_to_vec to wordid_to_vec
    :param word2vec: map word_to_vec
    :param word2id: map word_to_word_id
    :return: wordid_to_vec
    """
    wordid_2_vec = dict()
    for word,vec in word2vec.iteritems():
        wordid_2_vec[word2id[word]] = vec
    return wordid_2_vec
def get_one_hot_w2vec():
    wordid_2_vec = dict()
    for i in xrange(NUM_TAGS):
        ar = [0 for j in xrange(NUM_TAGS)]
        ar[i] = 1
        wordid_2_vec[i] = ar
    return wordid_2_vec

def load_glove_w2v(tag_2_id):
    if os.path.isfile(GLOVE_PATH_SHORT):
        return load_glove_from_file(GLOVE_PATH_SHORT,tag_2_id)
    else:
        word_2_vec = load_glove_from_file(GLOVE_PATH,tag_2_id)
        f = open(GLOVE_PATH_SHORT,"w")
        for w,v in word_2_vec.iteritems():
            f.write("%s"%w)
            for vi in v:
                f.write(" %s"%vi)
            f.write("\n")
        f.close()
        return word_2_vec


def get_label_train_array(wordid_2_vec):
    
    label_vectors = []
    label_ids = []
    valid_ids = []
    DIM = len(wordid_2_vec[0])
    zeros = [0 for i in xrange(DIM)]
    f_in = open(TAG_TRAIN_PATH,"r")
    id = 0
    while True:
        line = f_in.readline()
        if line == "":
            break
        line = line.strip()
        array_ids = np.fromstring(line,dtype=int,sep=" ")
        label_id = array_ids.argmax()

        if array_ids[label_id] == 0:
            label_vectors.append(zeros)
            label_ids.append(-1)
        else:
            label_vectors.append(wordid_2_vec[label_id])
            label_ids.append(label_id)
            valid_ids.append(id)
        id += 1

    f_in.close()
    return label_vectors,label_ids,valid_ids


def load_train_label_vectors():

    print "Loading train label vectors"
    tag_2_id, id_2_tag =load_tag_concepts81()
    #word_2_vec = load_glove_w2v(tag_2_id)
    #wordid_2_vec = convert_wvec_2_widvec(word_2_vec, tag_2_id)
    wordid_2_vec = get_one_hot_w2vec()
    distinct_wordvec_arrays = []

    for i in xrange(NUM_TAGS):
        distinct_wordvec_arrays.append(wordid_2_vec[i])
    label_vectors, label_ids, valid_ids = get_label_train_array(wordid_2_vec)
    return np.array(label_vectors), np.array(label_ids), np.array(valid_ids), np.array(distinct_wordvec_arrays)

def load_feature_vec(file_path,dtype=float):

    label_vectors = []
    f_in = open(file_path,"r")
    while True:
        line = f_in.readline()
        if line == "":
            break
        line = line.strip()
        array = np.fromstring(line,dtype=dtype,sep=" ")
        label_vectors.append(array)

    f_in.close()
    return np.array(label_vectors)

def get_best_match_index_transform_x2y(X_predict=None, Y_transform=None):
    shape = Y_transform.shape
    scores = np.ndarray(shape[0],dtype=float)
    for i in xrange(shape[0]):
        scores[i] = np.dot(Y_transform[i],X_predict)
        #scores[i] = entropy(x_transform,self.Y_transform[i])
    return [np.argmax(scores), np.max(scores)]

def get_best_match_cross_indices_x2y(transform_fn=None,X_raw=None,Y_transform=None):
    x_transformes = transform_fn(X_raw)
    results = []
    for x_transform in x_transformes:
        results.append(get_best_match_index_transform_x2y(X_predict=x_transform,
                                                          Y_transform=Y_transform))
    return results

def get_n_match_index_transform_x2y(X_predict=None, Y_transform=None,n=5):
    shape = Y_transform.shape
    scores = np.ndarray(shape[0],dtype=float)
    for i in xrange(shape[0]):
        scores[i] = np.dot(Y_transform[i],X_predict)
        #scores[i] = entropy(x_transform,self.Y_transform[i])
    
    indices = (-scores).argsort()[:n]
    return [indices, scores[indices]]

def get_n_match_cross_indices_x2y(transform_fn=None,X_raw=None,Y_transform=None,top_n=5):
    x_transformes = transform_fn(X_raw)
    results = []
    for x_transform in x_transformes:
        results.append(get_n_match_index_transform_x2y(X_predict=x_transform,
                                                          Y_transform=Y_transform,
                                                          n=top_n))
    return results
