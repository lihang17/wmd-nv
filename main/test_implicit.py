import implicit
import pandas as pd
from scipy import sparse

if __name__ == '__main__':
    data_dir = "../data/split_data/left_one_spilt/all__neg_6/train.instance"
    parser_m = pd.read_csv(data_dir,sep=' ')
    parser_m.columns = ['user_id','item_id','confidence']
    item_user_data = sparse.csr_matrix((parser_m['confidence'],(parser_m['item_id'],parser_m['user_id'])))

    model = implicit.als.AlternatingLeastSquares(factors=25,iterations=15)
    model.fit(item_user_data)
    user_items = item_user_data.T.tocsr()
    testList = [2,3,18,24,28,45,73,75,89,96,123,125,129,134,145,159,175,208,210,215]
    nList = [2,1,28,14,51,62,28,8,1,99,124,9,8,77,143,154,8,8,1,9]
    count = 0
    for n in range(len(testList)):
        recommendations = model.recommend(testList[n], user_items, N=500)
        for j in recommendations:
            if nList[n] in j:
                print("Yes")
                count+=1
    print(count,'/',len(testList))
    #print(type(recommendations))