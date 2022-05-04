# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
import random
import numpy as np
from process_text.processing_data import process_data
from helpers.util import list2dict


class split_dataset (object):
    def __init__(self, data_dir, split_mannner, train_ratio=0.7, num_negatives=6):  # ,valid_ratio
        """
        交叉验证？？？mashup划分时容易，但是按关系对比例划分时很难
        :param data_dir: 存取文件路径 /data
        :param split_mannner: 受划分方式的影响：按mashup划分还是划分一定比例的关系对
        :param train_ratio: 训练集比例：按mashup划分时 是mashup比例；按关系对划分时是关系对比例
        """

        self.data_dir = data_dir
        self.pd = process_data (self.data_dir)  # 未划分的数据集对象

        self.split_mannner = split_mannner
        self.result_path = self.data_dir + r'/split_data/' + self.split_mannner
        self.train_radio = train_ratio
        # self.valid_ratio = valid_ratio
        self.num_negatives = num_negatives
        self.train_mashup_api_list = []
        self.test_mashup_api_list = []



    def split_dataset(self):
        """
        首先得到train和test和划分,只包含实际调用的，不包含负采样的（在get_instances中）
        :return:
        """
        train_path = self.result_path + r'/train_set.data'
        test_path = self.result_path + r'/test_set.data'
        if os.path.exists (train_path):
            self.train_mashup_api_list = read_split_train (train_path, False)
            self.test_mashup_api_list = read_split_train (test_path, False)
            print ('read split data,done!')
        else:
            if self.split_mannner == 'mashup':
                self.train_mashup_api_list, self.test_mashup_api_list = self.split_dataset_by_mashup ()
            elif self.split_mannner == 'left_one_spilt':
                self.train_mashup_api_list, self.test_mashup_api_list = self.left_one_spilt ()
            elif self.split_mannner == 'cold_start':
                self.train_mashup_api_list, self.test_mashup_api_list = self.split_dataset_for_ColdStart ()
            else:
                raise ValueError ('Wrong split_mannner!')

            # 先求后存
            if not os.path.exists (self.result_path):
                os.makedirs (self.result_path)
            save_split_train (train_path, self.train_mashup_api_list)
            save_split_train (test_path, self.test_mashup_api_list)
            print ('split data and save,done!')

    def left_one_spilt(self):
        """
        留一法 ：每个train 只使用一个api 作为test
        :return:
        """
        mashup_api_dict = self.pd.get_mashup_api_pair ('dict')
        test_mashup_api_list = []
        train_mashup_api_list = []
        for mashup_id, api_ids in mashup_api_dict.items ():
            if len(api_ids) == 3:
                test_api = api_ids.pop()
                test_mashup_api_list.append((mashup_id, test_api))
            for train_api in api_ids:
                train_mashup_api_list.append ((mashup_id, train_api))
        return train_mashup_api_list, test_mashup_api_list

    def split_dataset_for_ColdStart(self):

        mashup_api_dict = self.pd.get_mashup_api_pair('dict')
        mashup_ids = list(mashup_api_dict.keys())
        random.shuffle(mashup_ids)

        mid = int(self.train_radio * len(mashup_ids))
        train_mashups = mashup_ids[:mid]

        test_mashup_api_list = []
        train_mashup_api_list = []

        for mashup_id, api_ids in mashup_api_dict.items():
            if mashup_id in train_mashups:
                for train_api in api_ids:
                    train_mashup_api_list.append((mashup_id, train_api))
            else:
                for train_api in api_ids:
                    test_mashup_api_list.append((mashup_id, train_api))
        return train_mashup_api_list, test_mashup_api_list

    def split_dataset_by_mashup(self):
        """
        按照论文的方法划分：按比例将mashup划分，分别用于训练和测试；其中测试用mashup也拿出一个api用于训练
        将数据分为训练集和测试集，测试集中，每个mashup要拿出一个api放到训练集中[将所有的边打掉只剩下1条边，剩下的拿来测试]
        :param :
        :return:
        """
        mashup_api_dict = self.pd.get_mashup_api_pair ('dict')
        mashup_ids = list (mashup_api_dict.keys ())
        random.shuffle (mashup_ids)

        mid = int (self.train_radio * len (mashup_ids))
        train_mashups = mashup_ids[:mid]
        test_mashups = mashup_ids[mid:]

        train_mashup_api_list = []  # (mashup,api)list
        test_mashup_api_list = []

        for train_mashup_id in train_mashups:
            train_apis = mashup_api_dict[train_mashup_id]
            for api_id in train_apis:  # 所有
                train_mashup_api_list.append ((train_mashup_id, api_id))

        # 每个test mashup随机选一个正例
        for test_mashup_id in test_mashups:
            test_apis = mashup_api_dict[test_mashup_id]

            random_api_id = test_apis.pop ()  # 随机一个正例作为训练集
            train_mashup_api_list.append ((test_mashup_id, random_api_id))

            # test
            for api_id in test_apis:
                test_mashup_api_list.append ((test_mashup_id, api_id))
        return train_mashup_api_list, test_mashup_api_list



    def get_model_instances(self, candidates_manner, candidates_ratio=99, candidates_num=100):
        """
        通用于 一般的数据集和完全冷启动项目数据集
        根据划分的train，test，从unobserved中选择train的负例，用于test的candidates
        :param candidates_manner: 'ratio':  # 留一法中的1:99；  'num':  # eg:算上ground只要100个 100-x   'all':  # 除去...,所有未知的
        :param candidates_ratio:  candidates_manner='ratio' 时需设置该比例
        :param candidates_num:   candidates_manner='num'  时需设置该数量
        :return train_mashup_api_list：  (train_mashup_id_instances,train_api_id_instances)
                                        每个mashup有n（其调用api的个数）个正例，6*n个负例
        :return train_labels:           训练数据的标签
        :return test_api_id_instances:  100个候选api 用于测试 其中包括了split_dataset_by_mashup()中的测试数据，剩下的是随机选出
        :return grounds:                split_dataset_by_mashup()中选出的测试数据
        """

        s = ''  # 名称中的取值字段
        if candidates_manner == 'ratio':
            s = candidates_ratio
        elif candidates_manner == 'num':
            s = candidates_num

        data_dir = self.result_path + '/{}_{}_neg_{}'.format (candidates_manner, s, self.num_negatives)  # 某种选择下的写入路径
        if not os.path.exists (data_dir):
            os.makedirs (data_dir)
        train_instance_path = data_dir + r'/train.instance'
        test_instance_path = data_dir + r'/test.instance'
        grounds_path = data_dir + r'/grounds'

        train_mashup_id_instances, train_api_id_instances, train_labels, test_mashup_id_instances, test_api_id_instances, grounds = [], [], [], [], [], []

        if os.path.exists (train_instance_path):
            train_mashup_api_list, train_labels = read_split_train (train_instance_path, True)
            test_mashup_id_instances, test_api_id_instances = read_test_instance (test_instance_path)
            grounds = read_2D_list (grounds_path)
            print ('read model instances,done!')
        else:
            train_mashup_api_list = self.train_mashup_api_list

            test_mashup_id_instances = []  # 二维
            test_api_id_instances = []
            grounds = []  # 二维

            mashup_num = len (self.pd.get_mashup_api_id2info ('mashup'))
            api_num = len (self.pd.get_mashup_api_id2info ('api'))

            train_dict = list2dict (self.train_mashup_api_list)  # 每个mashup的正例 字典 -》set
            train_labels = [1] * len (self.train_mashup_api_list)  # 一维
            ground_set = list2dict (self.test_mashup_api_list)  # 测试集实际是当标准

            for mashup_id in range (mashup_num):
                all_apis = {api_id for api_id in range (api_num)}

                temp_train_apis = train_dict.get (mashup_id)  # 每个mashup的train不可能为none
                if temp_train_apis is None:
                    temp_train_apis = set ()

                temp_test_apis = ground_set.get (mashup_id)  # 真实调用的作为ground
                if temp_test_apis is None:
                    temp_test_apis = set ()

                # 可观测的：train/ground  剩余为未观测
                all_unboserved = list (all_apis - temp_train_apis - temp_test_apis)
                temp_negative_train_apis_num = self.num_negatives * len (temp_train_apis)

                random.shuffle (all_unboserved)
                # 根据训练正例为train找负例
                temp_negative_train_apis = all_unboserved[:temp_negative_train_apis_num]
                for negative_train_api in temp_negative_train_apis:
                    train_mashup_api_list.append ((mashup_id, negative_train_api))
                train_labels = train_labels + [0] * len (temp_negative_train_apis)

                # 根据测试ground寻找candidates_apis：首先确定数目，然后刨去负例
                if len (temp_test_apis) > 0:
                    if candidates_manner == 'ratio':  # 留一法中的1:99
                        temp_candidates_num = candidates_ratio * len (temp_test_apis)
                    elif candidates_manner == 'num':  # eg:算上ground只要100个 100-x
                        temp_candidates_num = candidates_num - len (temp_test_apis)
                    elif candidates_manner == 'all':  # 除去...,所有未知的
                        temp_candidates_num = len (all_unboserved) - temp_negative_train_apis_num

                    candidates_apis = all_unboserved[
                                      temp_negative_train_apis_num:temp_negative_train_apis_num + temp_candidates_num] + list (
                        temp_test_apis)  # 一部分unboserved和ground的和
                    test_mashup_id_instances.append ([mashup_id] * len (candidates_apis))
                    test_api_id_instances.append (candidates_apis)
                    grounds.append (list (temp_test_apis))

            save_split_train (train_instance_path, train_mashup_api_list, train_labels)
            save_test_instance (test_instance_path, test_mashup_id_instances, test_api_id_instances)
            save_2D_list (grounds_path, grounds)
            print ('get model instances and save them,done!')

        train_mashup_id_instances, train_api_id_instances = zip (*train_mashup_api_list)

        return train_mashup_id_instances, train_api_id_instances, train_labels, test_mashup_id_instances, test_api_id_instances, grounds


class transfer_UI_matrix(object):
    """
    将不连续的mashup和api的交互矩阵转化为连续，重新编码：id从小到大
    """
    def __init__(self,train_mashup_api_list):
        self.train_mashup_api_list=train_mashup_api_list
        self.train_mashup_ids, self.train_api_ids = zip (*train_mashup_api_list)
        self.train_mashup_ids = np.unique (self.train_mashup_ids)  # 有序排列
        self.train_api_ids = np.unique (self.train_api_ids)

        self.train_mashup_num = len (self.train_mashup_ids)
        self.train_mashup_id2index = {self.train_mashup_ids[i]: i for i in
                                      range (self.train_mashup_num)}  # api的id到local index的映射
        self.train_api_num = len (self.train_api_ids)
        self.train_api_id2index = {self.train_api_ids[i]: i for i in
                                   range (self.train_api_num)}  # api的id到local index的映射

    def get_transfered_UI(self, mode, contain1=True):
        # 将U-I id数据转化为内部映射的 id，id，1
        if mode == 'array':
            UI_array = []
            for mashup_id, api_id in self.train_mashup_api_list:
                if contain1:
                    UI_array.append (
                        (self.train_mashup_id2index[mashup_id], self.train_api_id2index[api_id], 1))  # 显示 隐式
                else:
                    UI_array.append ((self.train_mashup_id2index[mashup_id], self.train_api_id2index[api_id]))
            return np.array (UI_array)
        elif mode == 'matrix':
            UI_matrix = np.zeros ((self.train_mashup_num, self.train_api_num), dtype='int32')
            for mashup_id, api_id in self.train_mashup_api_list:
                UI_matrix[self.train_mashup_id2index[mashup_id]][self.train_api_id2index[api_id]] = 1
            return UI_matrix

def save_split_train(path, train_test_mashup_api_list, train_labels=None):
    """
    工具类，用于存储list
    用在保存split得到的 train，test结构（list)；也可用于保存据此生成的有label的实例
    格式：mashup_id api_id label(可选）
    :param path:
    :param train_test_mashup_api_list:
    :param train_labels:
    :return:
    """
    with open (path, 'w') as f:
        if train_labels is None:
            for mashup_id, api_id in train_test_mashup_api_list:
                f.write ('{} {}\n'.format (mashup_id, api_id))
        else:
            assert len (train_test_mashup_api_list) == len (train_labels)
            for i in range (len (train_test_mashup_api_list)):
                a_pair = train_test_mashup_api_list[i]
                f.write ('{} {} {}\n'.format (a_pair[0], a_pair[1], train_labels[i]))


def read_split_train(path, have_label):
    mashup_api_list = []
    labels = []
    with open (path, 'r') as f:
        line = f.readline ()
        while line is not None and len(line) != 0:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                continue
            mashup_api_list.append ((ids[0], ids[1]))
            if have_label:
                labels.append (ids[2])
            line = f.readline ()
    if have_label:
        return mashup_api_list, labels
    else:
        return mashup_api_list


def save_test_instance(path, test_mashup_id_instances, test_api_id_instances):
    """
    将实际使用的测试样例 写入文件 ： mashup_id api_id1 api_id2
    :param path:
    :param test_mashup_id_instances:
    :param test_api_id_instances:
    :return:
    """
    with open (path, 'w') as f:
        assert len (test_mashup_id_instances) == len (test_api_id_instances)
        for index in range (len (test_mashup_id_instances)):
            mashup_id = test_mashup_id_instances[index][0]
            api_ids = test_api_id_instances[index]
            f.write (str (mashup_id) + ' ')
            f.write (' '.join ([str (api_id) for api_id in api_ids]))
            f.write ('\n')


def read_test_instance(test_instance_path):
    test_mashup_id_instances = []
    test_api_id_instances = []
    with open (test_instance_path, 'r') as f:
        line = f.readline ()
        while line is not None:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                break
            mashup_id = ids[0]
            api_ids = ids[1:]
            test_mashup_id_instances.append ([mashup_id] * len (api_ids))
            test_api_id_instances.append (api_ids)

            line = f.readline ()
    return test_mashup_id_instances, test_api_id_instances


def read_2D_list(path):
    _list = []
    with open (path, 'r') as f:
        line = f.readline ()
        while line is not None:
            ids = [int (str_id) for str_id in line.split ()]
            if len (ids) == 0:
                break
            _list.append (ids)
            line = f.readline ()
    return _list


def save_2D_list(path, _list):
    with open (path, 'w') as f:
        for index in range (len (_list)):
            f.write (' '.join ([str (id) for id in _list[index]]))
            f.write ('\n')


"""
执行两步操作
1、运行split_dataset_by_mashup()【在split_dataset()中调用】
    将mashup按一定比例分为训练集和测试集
    训练集中的mashup，其调用的所有的api都进训练集
    而测试集中的mashup，随机选一个api进入训练集，剩下的api当作正例
    并保存到../data/spilit_data/mashup/train_set.data
2、运行get_model_instances
    将前面处理好的数据进行二次处理
    将split_dataset_by_mashup()处理好的数据拿出来，分批处理
    对于每个mashup，先拿出所有的api，然后减去它所调用的api
    然后为mashup选择n（6）*调用api个数的负例，加入训练集
    之后再为测试集中的数据选出100-x（测试集中api的个数）作为负例，加入测试集
"""
if __name__ == '__main__':

    ds = split_dataset ('../data', 'left_one_spilt', 0.7, 6)  # 'mashup','manner_ratio'
    ds.split_dataset ()
    instance_result = ds.get_model_instances ('all',
                                              candidates_num=100)  # 'num', candidates_num=100  # 'ratio', candidates_ratio=99 # 'num'
