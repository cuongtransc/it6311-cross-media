import logging
import numpy as np
import uuid
import optparse
import yaml

from keras.callbacks import ModelCheckpoint

from models.DeepCCA.models import create_model
from prepare_data import *

# Loading data from file config
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
image_train = load_feature_vec(IMAGE_FEATURE_TRAIN_PATH)
text_train = load_feature_vec(TAG_TRAIN_PATH)

image_test = load_feature_vec(IMAGE_FEATURE_TEST_PATH)
text_test = load_feature_vec(TAG_TEST_PATH)

INPUT_SHAPE1 = len(image_train[0])
INPUT_SHAPE2 = len(text_train[0])
OUTDIM_SIZE = 50

TOP_K = 1

# Loading neural network architecture
# number of layers with nodes in each one
layer_sizes1 = [2048, 1024, 1024, OUTDIM_SIZE]
layer_sizes2 = [2048, 1024, 1024, OUTDIM_SIZE]
# the parameters for training the network
learning_rate = 1e-2
epoch_num = 50
batch_size = 512
# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top OUTDIM_SIZE ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False
# Building, training, and producing the new features by DCCA
model = create_model(layer_sizes1, layer_sizes2, INPUT_SHAPE1, INPUT_SHAPE2,
                        learning_rate, reg_par, OUTDIM_SIZE, use_all_singular_values)
model.summary()
checkpointer = ModelCheckpoint(filepath="temp_weights_{}.h5".format(uuid.uuid1()), 
								verbose=1, 
								save_best_only=True, save_weights_only=True)

def test(opts=None):
    #print cca_model.X_transform.shape
    #print cca_model.Y_transform.shape
    filename = opts.filename
    model.load_weights(filename)
    image_model = model.layers[0].layers[0]
    text_model = model.layers[0].layers[1]

    label_features, label_ids,valid_ids, distinct_labels = load_train_label_vectors()
    img_features = load_train_img_feature_vectors(IMAGE_FEATURE_TEST_PATH)


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

    feature_img_arr = np.array(feature_ins).reshape(-1,INPUT_SHAPE1)

    labels_transform = text_model.predict(distinct_labels)
    LOGGER.info(labels_transform.shape)
    labels_pre = get_n_match_cross_indices_x2y(transform_fn=image_model.predict,
						    						X_raw=feature_img_arr,
						    						Y_transform=labels_transform)

    #print label_ins,labels_pre
    print "-----------------------Stats-----------------------------"
    cc = 0
    for i in xrange(TEST_SIZE):
        for lb in labels_pre[i][0]:
            if label_ins[i] == lb :
                cc += 1
                break
    print "Acc image query: %.5f%% "%(cc*100.0/TEST_SIZE)

    image_transform = image_model.predict(feature_img_arr)
    image_pres = get_n_match_cross_indices_x2y(transform_fn=text_model.predict,
						    						X_raw=distinct_labels,
						    						Y_transform=image_transform)
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
    optparser.add_option("-F", "--filename", help="Filename to load/save", default="")

    opts = optparser.parse_args()[0]

    if opts.train != "" or opts.retrain != "":
		LOGGER.info("---Training---")
		model.fit([image_train, text_train], np.zeros(len(image_train)),
		          batch_size=batch_size, epochs=epoch_num, shuffle=True,
		          validation_split=0.1,verbose = 2,
		          callbacks=[checkpointer])
		filepath = "deep_cca_OUTDIM_SIZE_{}.model".format(OUTDIM_SIZE)
		model.save(filepath)

    else:
    	LOGGER.info("---Test---")
        test(opts)

if __name__ == "__main__":
    main()

