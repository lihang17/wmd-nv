# -*- coding:utf-8 -*-

import numpy as np
from keras.layers.core import Dropout, Flatten
from keras.layers import Lambda, Concatenate, MaxPooling2D, \
    LSTM, merge
from keras.layers import Dense, Input, Conv2D, Embedding, concatenate, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import Constant
from keras import initializers
from keras import backend as K
import os

from embedding.embedding import get_embedding_matrix
from process_text.processing_data import process_data
from recommend_models.simple_inception import inception_layer
import tensorflow as tf

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 30000
channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
categories_size=3


class recommend_Model(object):
    """
    共同基类
    """
    def __init__(self,base_dir,mf_embedding_dim,mf_fc_unit_nums):
        """
        :param base_dir:
        :param mf_embedding_dim: 因为DHSR NCF都包含MF部分，所以这里作为属性（尽管只处理文本部分的模型不需要这部分,此时默认为空即可
        :param mf_fc_unit_nums:
        """
        self.base_dir = base_dir
        self.pd = process_data(self.base_dir, False)
        self.num_users = len(self.pd.get_mashup_api_index2name('mashup'))
        self.num_items = len(self.pd.get_mashup_api_index2name('api'))
        self.mf_embedding_dim=mf_embedding_dim
        self.mf_fc_unit_nums=mf_fc_unit_nums

    def get_name(self):
        name=''
        name += 'mf_embedding_dim:{} '.format(self.mf_embedding_dim)
        name += 'mf_fc_unit_nums:{} '.format(self.mf_fc_unit_nums).replace(',', ' ')
        return 'GX:'+name  # *** 用于区别每个模型  应包含选用的embedding，是否使用tag，inception结构，MF结构，总体结构（FC nums）

    # 类别如何处理？增加一部分？
    def get_model(self):
        """
        **TO OVERIDE**
        :return:  a model
        """
        pass

    def get_merge_MLP(self,input1,input2,MLP_layers):
        """
        难点在于建立model的话，需要设定Input，其中要用到具体形状
        """
        pass

    def get_mf_MLP(self,input_dim1,input_dim2,output_dim,MLP_layers):
        """
        返回id-embedding-merge-mlp的model
        """
        # Input Layer
        user_input = Input(shape=(1,), dtype='int32')
        item_input = Input(shape=(1,), dtype='int32')

        # Embedding layer
        MF_Embedding_User = Embedding(input_dim=input_dim1, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        MF_Embedding_Item = Embedding(input_dim=input_dim2, output_dim=output_dim,
                                      embeddings_initializer=initializers.VarianceScaling(scale=0.01,distribution='normal'),
                                      embeddings_regularizer=l2(0.01), input_length=1)
        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))  # why Flatten？
        mf_vector = concatenate([mf_user_latent, mf_item_latent])

        for idx in range(len(MLP_layers)):   # 学习非线性关系
            layer = Dense(MLP_layers[idx],  activation='relu')
            mf_vector = layer(mf_vector)
        model = Model(inputs=[user_input,item_input],outputs=mf_vector)
        return model

    def get_instances(self):
        """
        **TO OVERIDE**
        """
        pass

    def save_sth(self):
        pass


