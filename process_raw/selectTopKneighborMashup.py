import pandas as pd
import numpy as np

class selectTopKneighborMashup():
    def __init__(self,base_path):
        self.base_path = base_path;
        self.neighbor = {}

    def proceess(self,topK):
        with open(self.base_path) as f:
            temp = f.readlines()
            self.neighbor = {}
            id = '1'
            start = 0
            self.neighbor[id] = []
            for line in temp:
                start += 1
                if line.strip() == "*Edges":
                    break
            for line in temp[start:-1]:
                line = line.strip().split(' ')
                if line[0] != id:
                    self.neighbor[id].sort(key=lambda x:x[1],reverse=True)
                    self.neighbor[id] = self.neighbor[id][:topK]
                    id = line[0]
                    self.neighbor[id] = []
                self.neighbor[id].append(line[1:])




if __name__=="__main__":
    kit = selectTopKneighborMashup("../data/Similarity_result_net/API_Similarity_LDAS01.net")
    kit.proceess(50)
    print(kit.neighbor)

