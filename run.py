from data import *
from cca_model import CCA_Model
import cPickle as pickle
import os.path
import os
import optparse
N_DIM = 50
MODEL_PATH = "Data/models/cca.model"
def gen_model(opts=None):
    if opts!= None and opts.retrain != "":
        os.remove(MODEL_PATH)
    if not os.path.isfile(MODEL_PATH):
        img_features = load_train_img_feature_vectors()
        print img_features.shape

        label_features,label_ids,valid_ids,distinct_labels = load_train_label_vectors()
        print "Valid size: ",len(valid_ids)
        print "Total size: ",len(label_features)

        cca_model = CCA_Model(N_DIM)
        x_features = [img_features[i] for i in valid_ids ]
        y_features = [label_features[i] for i in valid_ids]
        cca_model.learn_model(x_features,y_features,distinct_labels)
        print "Saving model..."
        pickle.dump(cca_model, open("%s" % (MODEL_PATH), "wb"))
        return cca_model
    else:
        print "Load pre-train model..."
        cca_model = pickle.load(open("%s" % (MODEL_PATH), "rb"))
        #print cca_model.x_dim, cca_model.y_dim
        return cca_model


def test(opts=None):
    cca_model = gen_model()
    #print cca_model.X_transform.shape
    #print cca_model.Y_transform.shape

    label_features, label_ids,valid_ids, distinct_labels = load_train_label_vectors()
    img_features = load_train_img_feature_vectors()


    feature_ins = []
    label_ins = []
    test_ids = []
    TEST_SIZE = 5000
    if opts !=None:
        TEST_SIZE = opts.testsize

    cc = 0
    for i in valid_ids:
        feature_ins.append(img_features[i])
        label_ins.append(label_ids[i])
        test_ids.append(i)
        cc += 1
        if cc >= TEST_SIZE:
            break
    #print "Input: %s"%label_in
    labels_pre = cca_model.get_best_match_cross_indices_x2y(feature_ins)

    #print label_ins,labels_pre
    print "-----------------------Stats-----------------------------"
    cc = 0
    for i in xrange(TEST_SIZE):
        for lb in labels_pre[i][0]:
            if label_ins[i] == lb :
                cc += 1
                break
    print "Acc image query: %.5f%% "%(cc*100.0/TEST_SIZE)



    image_pres = cca_model.get_best_match_cross_indices_y2x(distinct_labels)
    ctag = 0
    for i in xrange(NUM_TAGS):
        #if label_ids[image_pres[i][0]] == i:
        #    ctag += 1
        for jj in image_pres[i][0]:
            lb = label_ids[jj]
            if lb == i:
                ctag += 1
    print "Acc text query : %.5f%% "%(ctag*100.0/NUM_TAGS)










def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-C", "--train", help="training mode", default="")
    optparser.add_option("-R", "--retrain", help="Retrain mode", default="")
    optparser.add_option("-T", "--test", help="Test mode", default="")
    optparser.add_option("-S", "--testsize", help="Test size",type = int, default=15000)

    opts = optparser.parse_args()[0]

    if opts.train != "" or opts.retrain != "":
        gen_model(opts)
    else:
        test(opts)

if __name__ == "__main__":

    main()