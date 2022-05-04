# -*- coding: utf-8 -*-
from LDA_Similarity.ApiDetailedDescription import read_APIDetailedDescription
from LDA_Similarity.calculate_similarity import LDA_similarity

Source_APIDD_Filename = "data/Source_APIDetailedDescription/APIDetailedDescription_Test.txt"
Result_Net_Filename = "../data/Result_Net/API_Similarity_LDAS01.net"

Source_API = {}
index2id = {}
Result_Net = open(Result_Net_Filename, 'a+', encoding='UTF-8')

if __name__ == '__main__':
    print("- 读取API描述文件 -")
    read_APIDetailedDescription(Source_API)
    length = len(Source_API)

    print("- 生成结果文件.Net -")
    #create_Result_Net(Result_Net, Source_API, length , index2id)

    # 编辑距离相似度
    # print("- Edit Distance Similarity -")
    # edit_distance_similarity(Source_API, length, Result_Net)

    # LDA文本相似度
    print("- Latent Dirichlet Allocation Similarity -")
    LDA_similarity(Source_API, length, Result_Net)

    Result_Net.close()
