# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:57:42 2019

@author: cyang26
"""

import csv
import numpy as np

def input_data():
  csvFile = open("sandy_resource_tweet_embeddings.csv", "r")
  reader = csv.reader(csvFile)

  result =[]
  for item in reader:
    line={}
    line['label']=item[0]
    line['data']=item[2:202]
    result.append(line)
  csvFile.close()

  name=['Clothing','Food','Medical','Money','Shelter','Volunteer']

  def group_data(name,result,idx):
    data=[]
    for item in result:
        if item['label']==name[idx]:
            data.append(np.array(item['data']))
    return np.array(data).reshape(-1,200,1).astype(np.float)

  data1=group_data(name,result,0)
  data2=group_data(name,result,1)
  data3=group_data(name,result,2)
  data4=group_data(name,result,3)
  data5=group_data(name,result,4)
  data6=group_data(name,result,5)
  label1=np.ones(len(data1))*1
  label2=np.ones(len(data2))*2
  label3=np.ones(len(data3))*3
  label4=np.ones(len(data4))*4
  label5=np.ones(len(data5))*5
  label6=np.ones(len(data6))*6
  return data1,data2,data3,data4,data5,data6,label1,label2,label3,label4,label5,label6