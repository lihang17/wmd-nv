# -*- coding:utf-8 -*-
import sys

from nltk.corpus import stopwords

from WMD.wmd import wmd_sim
import pickle
sys.path.append("..")
from helpers.util import cos_sim
import numpy as np
from keras.layers.core import Dropout, Reshape
from keras.layers import Lambda, Concatenate, Add
from keras.layers import Dense, Input, AveragePooling2D, concatenate
from keras.models import Model
from keras import backend as K
from process_text.processing_data import process_data, get_mashup_api_allCategories
import tensorflow as tf
from embedding.encoding_padding_texts import encoding_padding
from recommend_models.recommend_Model import recommend_Model
from LDA_Similarity.ApiDetailedDescription import *
from LDA_Similarity.calculate_similarity import *

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format () == 'channels_first' else 3
categories_size = 3


class gx_text_tag_only_MLP_model (recommend_Model):
    def __init__(self, base_dir, remove_punctuation, embedding_name,
                 u_factors_matrix, i_factors_matrix,
                 m_id2index,i_id2index,
                 max_ks, num_feat, topK,CF_self_1st_merge,
                 cf_unit_nums,text_weight=0.5,
                 predict_fc_unit_nums=[],
                 if_co=1, if_pop=True,co_unit_nums=[1024,256,64,16],
                 sims_path='../data/sims.info'):

        super (gx_text_tag_only_MLP_model, self).__init__ (base_dir, remove_punctuation, embedding_name)

        self.u_factors_matrix = u_factors_matrix
        self.i_factors_matrix = i_factors_matrix
        self.m_index2id = {index: id for id, index in m_id2index.items ()}
        self.i_id2index = i_id2index



        self.max_ks = max_ks
        self.max_k = max_ks[-1]  # 一定要小于num——users！
        self.predict_fc_unit_nums = predict_fc_unit_nums

        self.num_feat = num_feat
        self._map = None  # pair->x
        self.x_feature_dim = None
        self.mashup_id2CFfeature = None  # mashup-> text,tag 100D
        self.topK = topK
        self.text_weight=text_weight

        self.CF_self_1st_merge=CF_self_1st_merge
        self.cf_unit_nums=cf_unit_nums
        self.model = None

        self.if_co = if_co # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
        self.if_pop = if_pop
        self.co_unit_nums=co_unit_nums
        self.api_id2covec,self.api_id2pop = self.pd.get_api_co_vecs()
        self.mashup_id_pair = self.pd.get_mashup_api_pair('dict')

        self.mashup_id2neighbors={}
        self.mashup_id2CFfeature = {}
        self.sims_path = sims_path


    def get_name(self):
        name = super (gx_text_tag_only_MLP_model, self).get_name ()
        cf_='_cf_unit' if self.CF_self_1st_merge  else ''
        name=name+cf_

        co_= '_coInvoke_' + str(self.if_co)
        pop_='_pop_' if self.if_pop else ''

        return 'gx_text_tag_MLP_only_model:' + name+ '_KNN_'+str(self.topK)+'_textWeight_'+str(self.text_weight)+co_+pop_ # ***

    #在这里完成了CI部分的模型 *
    #在initialize和get_model中寻找ci和ini的连接处
    def initialize(self,neighbors,
                   test_mashup_id_list, feature_train_mashup_ids):
        #prod = len (test_mashup_id_list) * len (test_mashup_id_list[0])
        #D1_test_mashup_id_list = tuple (np.array (test_mashup_id_list).reshape (prod, ))  # 将二维的test数据降维

        #feature_test_mashup_ids = sorted (list (set (D1_test_mashup_id_list)))  # 测试用mashup的升序排列



        # 先train，后test mashup id
        #all_feature_mashup_ids = feature_train_mashup_ids + feature_test_mashup_ids
        all_feature_mashup_ids = feature_train_mashup_ids
        # CNN提取的文本特征和tag的embedding大小不一样，所以无法直接拼接计算sim;需要单独计算sim，然后加权求和!!!
        
        if os.path.exists(self.sims_path):
            if_calculate = False
            all_sims_file = open(self.sims_path,'rb')
            all_sims_dict = pickle.load(all_sims_file)
            all_sims_file.close()
        else:
            Souce_mashup = {}
            read_APIDetailedDescription(Souce_mashup)
            #features = LDA_getFeatures(Souce_mashup)
            instance = wmd_sim(Souce_mashup)
            #print("MAX of i: ",len (all_feature_mashup_ids),"MAX of j:",len (feature_train_mashup_ids))
            all_sims_dict = {}
            if_calculate = True
        for i in range (len (all_feature_mashup_ids)):  # 为所有mashup找最近
            #id2sim = {}
            if if_calculate:
                sent_w = list(jieba.cut(Souce_mashup[all_feature_mashup_ids[i]][1]))
                english_stopwords = stopwords.words('english')
                query = [w for w in sent_w if not w in english_stopwords]
                index_sims = instance[query]
                #topk_index_sims = index_sims[1:self.topK+1]
                topk_index_sims = index_sims[:neighbors]
                all_sims_dict[i] = index_sims[:101]
            else:
                topk_index_sims = all_sims_dict[i][:neighbors]
            # for j in range (len (feature_train_mashup_ids)):  # 从所有train中找,存放的是内部索引
            #     if i != j:
            #         pass
                    #text_sim = cos_sim (features.iloc[[all_feature_mashup_ids[i]]], features.iloc[[j]])
                    #id2sim[j] = text_sim
            #topK_indexes, topK_sims = zip (*(sorted (id2sim.items (), key=lambda x: x[1], reverse=True)[:self.topK]))
            topK_indexes = [index[0] for index in topk_index_sims]
            topK_sims = [sim[1] for sim in topk_index_sims]
            self.mashup_id2neighbors[all_feature_mashup_ids[i]]=[self.m_index2id[index] for index in topK_indexes] #每个mashup距离最近的mashup的id list
            topK_sims = np.array (topK_sims) / sum (topK_sims)
            cf_feature = np.zeros ((self.num_feat))
            for z in range (len (topK_indexes)):
                #将邻接矩阵加权相加 *
                cf_feature += topK_sims[z] * self.u_factors_matrix[topK_indexes[z]]
            self.mashup_id2CFfeature[all_feature_mashup_ids[i]] = cf_feature
        with open(self.sims_path,'wb+') as f:
            pickle.dump(all_sims_dict,f)

    def get_model(self):
        # 搭建简单模型
        mashup_cf = Input (shape=(self.num_feat,), dtype='float32')
        api_cf = Input (shape=(self.num_feat,), dtype='float32')
        pair_x = Input (shape=(self.x_feature_dim,), dtype='float32')

        co_dim= self.topK if self.if_co == 3 else len(self.api_id2covec) # 3：最近邻是否调用 50D
        co_invoke=Input (shape=(co_dim,), dtype='float32')
        pop=Input (shape=(1,), dtype='float32')

        predict_vector=None
        # CF_self_1st_merge = True
        if self.CF_self_1st_merge:
            predict_vector = concatenate ([mashup_cf, api_cf])
            for unit_num in self.cf_unit_nums:
                predict_vector = Dense (unit_num, activation='relu') (predict_vector)
            #predict_vector = concatenate ([predict_vector, pair_x])
        else:
            predict_vector = concatenate ([mashup_cf, api_cf, pair_x])  # 整合文本和类别特征，尽管层次不太一样

        if self.if_co:
            predict_vector1 = Dense (self.co_unit_nums[0], activation='relu') (co_invoke)
            for unit_num in self.co_unit_nums[1:]:
                predict_vector1=Dense (unit_num, activation='relu') (predict_vector1)
            predict_vector = concatenate ([predict_vector,predict_vector1])

        for unit_num in self.predict_fc_unit_nums:
            predict_vector = Dense (unit_num, activation='relu') (predict_vector)
        predict_vector = Dropout (0.5) (predict_vector)
        predict_result = Dense (1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction") (
            predict_vector)

        if not (self.if_co or self.if_pop):
            #if_co == 0
            self.model = Model (inputs=[mashup_cf, api_cf],outputs=[predict_result])
        elif self.if_co and not self.if_pop:
            #if_co == 1,2,3
            self.model = Model (inputs=[mashup_cf, api_cf,co_invoke], outputs=[predict_result])
        elif self.if_pop and not self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,pop], outputs=[predict_result])
        elif self.if_pop and  self.if_co:
            self.model = Model (inputs=[mashup_cf, api_cf, pair_x,co_invoke,pop], outputs=[predict_result])
        return self.model

    #返回模型中INI组建里的mashup加权相加特征和api的特征，
    def get_instances(self, mashup_id_list, api_id_list):
        mashup_cf_features, api_cf_features, x_features, api_co_vecs,api_pops = [], [], [],[],[]
        api_zeros = np.zeros ((self.num_feat))
        num_api=len(self.api_id2covec[0])
        for i in range (len (mashup_id_list)):
            mashup_id = mashup_id_list[i]
            api_id = api_id_list[i]
            mashup_cf_features.append (self.mashup_id2CFfeature[mashup_id])
            api_i_feature = self.i_factors_matrix[self.i_id2index[api_id]] if api_id in self.i_id2index.keys() else api_zeros
            api_cf_features.append (api_i_feature)

            if self.if_co:
                if self.if_co==1:
                    api_co_vecs.append(self.api_id2covec[api_id])
                elif self.if_co==2:
                    api_co_vec=np.zeros((num_api))
                    for m_neigh_id in self.mashup_id2neighbors:
                        for _api_id in self.mashup_id_pair[m_neigh_id]: # 邻居mashup调用过的api
                            api_co_vec[_api_id]=self.api_id2covec[api_id][_api_id]
                    api_co_vecs.append (api_co_vec)
                elif self.if_co == 3: # 是否被最近邻调用
                    api_co_vec=[1 if api_id in self.mashup_id_pair[m_neigh_id] else 0 for m_neigh_id in self.mashup_id2neighbors[mashup_id]]
                    api_co_vecs.append (api_co_vec)
            if self.if_pop:
                api_pops.append(self.api_id2pop[api_id])

        if not (self.if_co or self.if_pop):
            return np.array (mashup_cf_features), np.array (api_cf_features)
        elif self.if_co and not self.if_pop:
            return np.array (mashup_cf_features), np.array (api_cf_features),  np.array (api_co_vecs)
        elif self.if_pop and not self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features), np.array (api_pops)
        elif self.if_pop and  self.if_co:
            return np.array (mashup_cf_features), np.array (api_cf_features), np.array (x_features),np.array(api_co_vecs),np.array(api_pops)