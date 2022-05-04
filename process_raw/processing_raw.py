import os
import pandas as pd
import numpy as np

class process_raw():
    def __init__(self, base_path,target_path, process_new=False):
        self.base_path = base_path
        self.target_path = target_path
        self.process()

    def process(self):
        dirs = [os.path.join(self.base_path,'Mashup.txt'),os.path.join(self.base_path,'API.txt'),os.path.join(self.base_path,'mashup_api.txt'),os.path.join(self.base_path,'mashup_developer.txt')]
        mashupf = pd.read_csv(dirs[0],sep='\t')
        apif = pd.read_csv(dirs[1],sep='^')
        mashup_apif = pd.read_csv(dirs[2], sep='\t')
        developerf = pd.read_csv(dirs[3], sep='\t')
        dirs = {'Mashup':mashupf,'API':apif}

        data_type = ['Mashup','API']
        for count,type in enumerate(data_type):
            for index, data in dirs[type].iterrows():
                with open(os.path.join(self.target_path, type,str(index)), 'w',encoding='utf-8') as f:
                    SkipNUllDescription = False
                    id = data['id']
                    temp = ''
                    for row in dirs[type].columns:
                        if row == 'id':
                            continue
                        #取出数据，对于description为null的，为空
                        line = str(data[row]).strip()
                        if line.lower() != 'null' and len(line) != 0 :
                            temp += row + " >>\t"
                            temp += str(data[row])+'\n'
                        else:
                            temp = '-1'
                            break
                    if type == 'Mashup' and temp != '-1':
                        temp += "Related APIs" + " >>\t"
                        Related_API_ID = mashup_apif.query('mashupnumber==@id')
                        #遍历相关API,根据id找到name
                        for _,ConnectedAPI in Related_API_ID.iterrows():
                            #apid = re[2]
                            API_ID = ConnectedAPI["apinumber"]
                            Related_API_ROW = apif.query("id==@API_ID")
                            for _,API_name in Related_API_ROW.iterrows():
                                if Related_API_ROW.__len__() <= 0:
                                    print("related api == 0")
                                temp +=" "
                                temp += API_name["name"]
                                print()

                        temp += '\nDevelop' + " >>\t"
                        developer_rows = developerf.query('mushupnumber==@id')
                        for _,row in developer_rows.iterrows():
                            developer = row['developer']
                        temp += developer
                    f.writelines(temp)

if __name__ == '__main__':
    base_path = '../data/raw_data'
    target_path = '../data'
    pr = process_raw(base_path,target_path)