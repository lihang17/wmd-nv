import operator

class cf:
    def __init__(self, data):
        data_dic = {}
        id = 0
        for line in data.itertuples():
            if not line[1] in data_dic.keys():
                data_dic[line[1]] = {line[3]: line[2]}
        else:
            data_dic[line[1]][line[4]] = line[2]
        self.data = data_dic

    ########找到与当前用户最近的n个邻居
    def nearstUser(self,username,type,n=1):
        distances={};#用户，相似度
        for otherUser,items in self.data.items():#遍历整个数据集
            if otherUser != username:#非当前的用户
                distance=self.getDistance(self.data[username],self.data[otherUser],type)#计算两个用户的相似度
                distances[otherUser]=distance
        sortedDistance=sorted(distances.items(),key=operator.itemgetter(1),reverse=True);#最相似的N个用户
        #print ("排序后的用户为：",sortedDistance)
        return sortedDistance[:n]

    #给用户推荐电影
    def Recomand(self,username,tp = 'Pearson',n=1):
        recommand={};#待推荐的电影
        for user,score in dict(self.nearstUser(username,tp,n)).items():#最相近的n个用户
            #print ("推荐的用户：",(user,score))
            for movies,scores in self.data[user].items():#推荐的用户的电影列表
                if movies not in self.data[username].keys():#当前username没有看过
                    #print ("%s为该用户推荐的电影：%s"%(user,movies))
                    if movies not in recommand.keys():#添加到推荐列表中
                        recommand[movies]=scores
        return sorted(recommand.items(),key=operator.itemgetter(1),reverse=True);

if __name__ == '__main__':
    data_dir = r"..\data\split_data\left_one_spilt\all__neg_6\train.instance"
    with open(data_dir) as f:
        print(f.readline())