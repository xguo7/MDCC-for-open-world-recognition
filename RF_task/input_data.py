# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:49:32 2018

@author: gxjco
"""

from __future__ import print_function
import numpy as np
from six.moves import range
from math import sqrt
from cmath import phase

def read_data():
  def amp_pha(real,imag):
    return sqrt(real**2+imag**2),phase(complex(real,imag))
  lines1 = open('1_2943_Wired_TX1.csv','r').readlines()
  lines2 = open('1_2943_Wired_TX2.csv','r').readlines()
  lines3 = open('2_2943_Wired_TX1.csv','r').readlines()
  lines4 = open('2_2943_Wired_TX2.csv','r').readlines()
  lines5 = open('1_N210_Wired_TX.csv','r').readlines()
  lines6 = open('2_N210_Wired_TX.csv','r').readlines()
  lines7 = open('3_N210_Wired_TX.csv','r').readlines()
  lines8 = open('4_N210_Wired_TX.csv','r').readlines()

  # lines1 = lines11+lines12
  # lines2 = lines21+lines22
  
  lines1 = [_k[:-1].replace(',','\t') for _k in lines1]
  lines2 = [_k[:-1].replace(',','\t') for _k in lines2]
  lines3 = [_k[:-1].replace(',','\t') for _k in lines3]
  lines4 = [_k[:-1].replace(',','\t') for _k in lines4]
  lines5 = [_k[:-1].replace(',','\t') for _k in lines5]
  lines6 = [_k[:-1].replace(',','\t') for _k in lines6]
  lines7 = [_k[:-1].replace(',','\t') for _k in lines7]
  lines8 = [_k[:-1].replace(',','\t') for _k in lines8]
  row_size = 512
  col_size = 1
  sam_size1 = 8000
  sam_size2 = 8000
  sam_size3 = 8000
  sam_size4 = 8000
  sam_size5 = 8000
  sam_size6 = 8000
  sam_size7 = 8000
  sam_size8 = 8000
  num_channels = 4
  data_org1 = np.zeros((sam_size1 * row_size * col_size, num_channels))
  data_org2 = np.zeros((sam_size2 * row_size * col_size, num_channels))
  data_org3 = np.zeros((sam_size3 * row_size * col_size, num_channels))
  data_org4 = np.zeros((sam_size4 * row_size * col_size, num_channels))
  data_org5 = np.zeros((sam_size5 * row_size * col_size, num_channels))
  data_org6 = np.zeros((sam_size6 * row_size * col_size, num_channels))
  data_org7 = np.zeros((sam_size7 * row_size * col_size, num_channels))
  data_org8 = np.zeros((sam_size8 * row_size * col_size, num_channels))
  for i in range(sam_size1*row_size*col_size):
        line = lines1[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org1[i]=items
  for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org2[i] = items
  for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org3[i]=items
  for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org4[i] = items
  for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org5[i]=items
  for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org6[i]=items
  for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org7[i]=items
  for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org8[i]=items
  data1  = data_org1.reshape([sam_size1,row_size,num_channels])
  data2 = data_org2.reshape([sam_size2, row_size, num_channels])
  data3  = data_org3.reshape([sam_size3,row_size,num_channels])
  data4 = data_org4.reshape([sam_size4, row_size, num_channels])
  data5  = data_org5.reshape([sam_size5,row_size,num_channels])
  data6  = data_org6.reshape([sam_size6,row_size,num_channels])
  data7 = data_org7.reshape([sam_size7, row_size, num_channels])
  data8  = data_org8.reshape([sam_size8,row_size,num_channels]) 
  np.random.shuffle(data1)
  np.random.shuffle(data2)
  np.random.shuffle(data3)
  np.random.shuffle(data4)
  np.random.shuffle(data5)
  np.random.shuffle(data6)
  np.random.shuffle(data7)
  np.random.shuffle(data8)
  
  label1=np.empty((sam_size1))
  label1[:]=1
  label2=np.empty((sam_size1))
  label2[:]=2
  label3=np.empty((sam_size1))
  label3[:]=3
  label4=np.empty((sam_size1))
  label4[:]=4
  label5=np.empty((sam_size1))
  label5[:]=5
  label6=np.empty((sam_size1))
  label6[:]=6
  label7=np.empty((sam_size1))
  label7[:]=7
  label8=np.empty((sam_size1))
  label8[:]=8
  
  return data1,data2,data3,data4,data5,data6,data7,data8,label1,label2,label3,label4,label5,label6,label7,label8