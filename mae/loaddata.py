import numpy as np
import const

class TrueData():
    def __init__(self, path_features="", path_label=""):

        assert path_label!= "" and path_features != ""





        self._index_in_epoch = 0
        self._epochs_completed = 0
        #self._num_examples = const.N_EXAMPLE
        self._rate = 0.8

        n_current = 0

        fLabel = open(path_label,"r")
        fFea = open(path_features, "r")

        list_features = []
        list_labels = []
        list_labelids = []
        list_lineids = []


        while True:
            line_label = fLabel.readline()
            line_feature = fFea.readline()

            if line_label == "":
                break
            n_current += 1
            if line_feature == "":
                print "Unmatching line at ",n_current
                exit(-1)



            line_label = line_label.strip()
            label_vec = np.fromstring(line_label, dtype=int, sep=" ")
            label_id = label_vec.argmax()
            v_label_max = label_vec[label_id]
            if v_label_max == 0:
                continue


            line_feature == line_feature.strip()

            feature_vec = np.fromstring(line_feature, dtype=float, sep=" ")
            #print feature_vec.shape

            list_lineids.append(n_current)

            list_features.append(feature_vec)
            list_labels.append(label_vec)
            list_labelids.append(label_id)
        fLabel.close()
        fFea.close()

        print "Length: ",len(list_features),len(list_labels)
        self._n_total = len(list_features)

        #self.all_images = np.concatenate(list_features,axis=0)
        #self.all_labels = np.concatenate(list_labels,axis=0)
        self.all_images = np.asarray(list_features)
        self.all_labels = np.asarray(list_labels)
        self.all_labelids = np.array(list_labelids,dtype=int)
        self.all_lineids =np.asarray(list_lineids,dtype=int)

        print "All shape: ",self.all_images.shape,self.all_labels.shape,self.all_labelids.shape


        perm0 = np.arange(self._n_total)
        print perm0
        np.random.shuffle(perm0)

        i_spliter = int(self._n_total * self._rate)
        self._num_examples = i_spliter

        train_indices = perm0[:i_spliter]
        test_indices = perm0[i_spliter:]

        lineid_test= self.all_lineids[test_indices]

        lineid_train = self.all_lineids[train_indices]

        flineid = open("line_ids_testi.dat","w")
        for idc in lineid_test:
            flineid.write("%s\n"%idc)
        flineid.close()

        ftrainid =open("line_ids_traini.dat","w")
        for idc in lineid_train:
            ftrainid.write("%s\n"%idc)
        ftrainid.close()






        self.images = self.all_images[train_indices]
        self.true_img_tags = self.all_labelids[:i_spliter]

        self.texts = self.all_labels[train_indices]

        self.dtags = np.ndarray((const.N_TAGS,const.INP_TXT_DIM))
        self.dtags.fill(0)
        for i in xrange(const.N_TAGS):
            self.dtags[i][i] = 1




        self.true_txt_tags = np.arange(const.N_TAGS)

        self.test_images = self.all_images[test_indices]
        self.true_test_img_tags = self.all_labelids[test_indices]



        #
        # self.images= np.random.rand(const.N_EXAMPLE, const.INP_IMAGE_DIM)
        # self.true_img_tags = np.random.randint(0,const.N_TAGS,const.N_EXAMPLE)
        #
        # self.texts= np.random.rand(const.N_EXAMPLE, const.INP_TXT_DIM)
        #
        # self.dtags = np.random.rand(const.N_TAGS,const.INP_TXT_DIM)
        # self.true_txt_tags = np.arange(const.N_TAGS)
        #
        #
        # self.test_images = np.random.rand(const.N_TEST,const.INP_IMAGE_DIM)
        # self.true_test_img_tags = np.random.randint(0,const.N_TAGS,const.N_TEST)





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
            #self._images = self.images
            #self._texts = self.texts

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

