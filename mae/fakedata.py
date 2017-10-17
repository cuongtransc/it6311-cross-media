
import numpy as np

import const


class FakeData():
    def __init__(self):

        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = const.N_EXAMPLE

        self.images= np.random.rand(const.N_EXAMPLE, const.INP_IMAGE_DIM)
        self.true_img_tags = np.random.randint(0,const.N_TAGS,const.N_EXAMPLE)

        self.texts= np.random.rand(const.N_EXAMPLE, const.INP_TXT_DIM)

        self.dtags = np.random.rand(const.N_TAGS,const.INP_TXT_DIM)
        self.true_txt_tags = np.arange(const.N_TAGS)


        self.test_images = np.random.rand(const.N_TEST,const.INP_IMAGE_DIM)
        self.true_test_img_tags = np.random.randint(0,const.N_TAGS,const.N_TEST)





    def get_all_train_images(self):
        return self.images,self.true_img_tags

    def get_all_txt_tags(self):
        return self.dtags,self.true_txt_tags

    def get_test_txt_tags(self):
        return self.dtags, self.true_txt_tags
    def get_test_images(self):
        return self.test_images,self.true_test_img_tags





    def tripple_data(self,data1,data2):
        data1_0 = np.zeros((data1.shape),dtype=float)
        data2_0 = np.zeros((data2.shape),dtype=float)
        new_data1 = np.concatenate((data1,data1_0,data1),axis=0)
        new_data2 = np.concatenate((data2,data2,data2_0),axis=0)

        true_data1 = np.concatenate((data1,data1,data1),axis=0)
        true_data2 = np.concatenate((data2,data2,data2), axis = 0)
        segment_size = data1.shape[0]
        perms = np.arange(segment_size*3)
        for i in xrange(segment_size*3):
            anchor = (i%3)*segment_size
            offset = i/3
            perms[i] = anchor + offset

        new_data1 = new_data1[perms]
        new_data2 = new_data2[perms]

        return [(new_data1,new_data2),(true_data1,true_data2)]







    def next_minibatch(self,batch_size):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 :
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._texts = self.texts[perm0]
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            texts_rest_part = self._texts[start:self._num_examples]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            texts_new_part = self._texts[start:end]
            return self.tripple_data(np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (texts_rest_part, texts_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.tripple_data(self._images[start:end], self._texts[start:end])

