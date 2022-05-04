# -*- coding:utf-8 -*-
import sys
sys.path.append("..")
from main.para_setting import Para
from main.train_predict_methods import get_preTrained_text_tag_model, train_test_text_tag_CF_model
from main.evalute import test_evalute, load_model_evalute

from recommend_models.text_tag_model import  \
    gx_text_tag_only_MLP_model

def tst_inceptionCF(text_extracter_mode,Category_type, tag_manner, merge_manner, para_mode,CF_self_1st_merge,text_weight,if_co,if_pop):
    GX_base_parameters = Para.data_dir, Para.remove_punctuation, Para.embedding_name


    max_ks = [5, 10, 20, 30, 40, 50]  # ***
    max_ks = [n+1 for n in range(20)] + [30,40,50]
    pmf_01='01'

    # 获得text_tag_model
    # gx_text_tag_model只有MLP *

    #para_mode = 'MLP_only' *

    # 预训练 text_tag_model，得到mashup的features；和预训练的模型参数；是否重新训练
    #ini_features_array, text_tag_model=get_preTrained_text_tag_model(text_tag_recommend_model,text_tag_model,train_new=True)

    if para_mode == 'MLP_only':
        test_times=1
        for i in range(test_times):
            # trian_MLP_only(text_tag_recommend_model, text_tag_model, ini_features_array, max_ks[-1], predict_fc_unit_nums)
            #only_MLP比text_tag_model 多了一层MLP用于整合MLP部分和MF部分 *
            text_tag_MLP_only_recommend_model=gx_text_tag_only_MLP_model(*GX_base_parameters,
                                                                         Para.u_factors_matrix,Para.i_factors_matrix,
                                                                         Para.m_id2index,Para.i_id2index,
                                                                         max_ks,Para.num_feat,max_ks[-1],CF_self_1st_merge,
                                                                         Para.cf_unit_nums,text_weight,
                                                                         Para.predict_fc_unit_nums,
                                                                         if_co,if_pop,
                                                                         Para.shadow_co_fc_unit_nums if if_co==3 else Para.deep_co_fc_unit_nums
                                                                         )
            text_tag_MLP_only_recommend_model.initialize(Para.neighbors,
                                                         Para.test_mashup_id_list,Para.feature_train_mashup_ids)

            #推测text_tag_MLP_only_model中包含了真实的tensor，而text_tag_MLP_only_recommend_model中包含了所需的数据，参数等
            text_tag_MLP_only_model=text_tag_MLP_only_recommend_model.get_model()
            # 是否重新训练上层的MLP：是
            test_evalute(text_tag_MLP_only_recommend_model, text_tag_MLP_only_model, text_tag_MLP_only_recommend_model.get_name()+'_'+Para.mf_mode, Para.num_epochs, train=True)
            #load_model_evalute(text_tag_MLP_only_recommend_model, '../data/model/0/3', text_tag_MLP_only_recommend_model.get_name()+'_'+Para.mf_mode, Para.num_epochs)
            

if __name__ == '__main__':
    #model_names=['DHSR_noMF'] # 'DHSR','DHSR_noMF','NCF'  'text_only', 'text_only_MF','text_tag','text_tag_MF','text_tag_CF'
    text_extracter_modes=['inception'] # 'inception','LSTM','textCNN'
    Category_types=['all'] # ,'first'
    tag_manners=['old_average'] #'old_average'
    merge_manners=['direct_merge'] #'final_merge','direct_merge'
    #times=1

    """
    for i in range(times):
        for model_name in model_names:
            tst_para(model_name,text_extracter_modes,Category_types, tag_manners, merge_manners)
    """
    CF_self_1st_merge = True
    text_weights = [0.0] #0.0+0.02*i for i in range(4)
    mf_modes = ['Node2vec'] # 'listRank','BPR'
    # ? *
    # 0是没有，1是跟所有api的共现次数向量；2是跟最近邻mashup调用过的api的共现次数；3是最近邻mashup是否调用过该api，50D
    if_cos = [0,1,2,3]
    # ?
    if_pops = [False,False,False,False]
    #baseline_model = 'binary_keyword' #'Samanta'
    model_name = 'MLP_only'

    Para.set_MF_mode (mf_modes[0])
    for i in range(len(if_cos)):
        tst_inceptionCF(text_extracter_modes[0],
                        Category_types[0],
                        tag_manners[0],
                        merge_manners[0],
                        model_name,
                        CF_self_1st_merge,
                        text_weights[0],
                        if_cos[i],if_pops[i]) #
        break