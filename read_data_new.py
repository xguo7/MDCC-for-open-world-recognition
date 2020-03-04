# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:04:45 2018

@author: gxjco
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from math import sqrt
from numpy import arctan
from cmath import phase
import os

# In[ ]:
def amp_pha(real,imag):
    #return sqrt(real**2+imag**2), arctan(real/imag)
    return sqrt(real**2+imag**2),phase(complex(real,imag))

def readdata(type_data,class_num):
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
  sam_size1 = 2000#int(len(lines1)/(row_size*col_size))
  sam_size2 = 2000#int(len(lines2)/(row_size*col_size))
  sam_size3 = 2000#int(len(lines3)/(row_size*col_size))
  sam_size4 = 2000#int(len(lines4)/(row_size*col_size))
  sam_size5 = 2000#int(len(lines5)/(row_size*col_size))
  sam_size6 = 2000#int(len(lines6)/(row_size*col_size))
  sam_size7 = 2000#int(len(lines7)/(row_size*col_size))
  sam_size8 = 2000#int(len(lines8)/(row_size*col_size))

  if type_data == 'original+pure':
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
        items = [items[0],items[1],items[2],items[3]]
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[2],items[3]]
        data_org8[i]=items

  elif type_data == 'original':
      num_channels = 2
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
        items =[items[0],items[1]]
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1]]
        data_org8[i]=items

  elif type_data == 'pure+noise':
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
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[2],items[3],items[0]-items[2],items[1]-items[3]]
        data_org8[i]=items
  elif type_data == 'noise':
      num_channels = 2
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
        items =[items[0]-items[2],items[1]-items[3]]
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0]-items[2],items[1]-items[3]]
        data_org8[i]=items

  elif type_data == 'original+noise':
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
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =[items[0],items[1],items[0]-items[2],items[1]-items[3]]
        data_org8[i]=items

  elif type_data == 'ap_original+pure':
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
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])+amp_pha(items[2],items[3])
        data_org8[i]=items

  elif type_data == 'ap_original':
      num_channels = 2
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
        items = amp_pha(items[0],items[1])
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0],items[1])
        data_org8[i]=items

  elif type_data == 'ap_original+noise':
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

        
  elif type_data == 'ap_pure+noise':
      num_channels=4
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
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[2],items[3])+amp_pha(items[0]-items[2],items[1]-items[3])
        data_org8[i]=items

  elif type_data == 'ap_noise':
      num_channels=2
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
        items = amp_pha(items[0]-items[2],items[1]-items[3])
        data_org1[i]=items
      for i in range(sam_size2*row_size*col_size):
        line = lines2[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[0]-items[2],items[1]-items[3])
        data_org2[i] = items
      for i in range(sam_size3*row_size*col_size):
        line = lines3[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0]-items[2],items[1]-items[3])
        data_org3[i]=items
      for i in range(sam_size4*row_size*col_size):
        line = lines4[i]
        items = [float(_k) for _k in line.split('\t')]
        items = amp_pha(items[0]-items[2],items[1]-items[3])
        data_org4[i] = items
      for i in range(sam_size5*row_size*col_size):
        line = lines5[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[0]-items[2],items[1]-items[3])
        data_org5[i]=items
      for i in range(sam_size6*row_size*col_size):
        line = lines6[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[0]-items[2],items[1]-items[3])
        data_org6[i]=items
      for i in range(sam_size7*row_size*col_size):
        line = lines7[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[0]-items[2],items[1]-items[3])
        data_org7[i]=items
      for i in range(sam_size8*row_size*col_size):
        line = lines8[i]
        items = [float(_k) for _k in line.split('\t')]
        items =amp_pha(items[0]-items[2],items[1]-items[3])
        data_org8[i]=items


  data1  = data_org1.reshape([sam_size1,row_size,num_channels])
  data2 = data_org2.reshape([sam_size2, row_size, num_channels])
  data3  = data_org3.reshape([sam_size3,row_size,num_channels])
  data4 = data_org4.reshape([sam_size4, row_size, num_channels])
  data5  = data_org5.reshape([sam_size5,row_size,num_channels])
  data6  = data_org6.reshape([sam_size6,row_size,num_channels])
  data7 = data_org7.reshape([sam_size7, row_size, num_channels])
  data8  = data_org8.reshape([sam_size8,row_size,num_channels])
  tr_size1 = int(sam_size1/2)
  tr_size2 = int(sam_size2/2)
  tr_size3 = int(sam_size3/2)
  tr_size4 = int(sam_size4/2)
  tr_size5 = int(sam_size5/2)
  tr_size6 = int(sam_size6/2)
  tr_size7 = int(sam_size7/2)
  tr_size8 = int(sam_size8/2)


  if class_num==2:
      data_tr = np.concatenate((data3[:tr_size7],data8[:tr_size8]),axis=0)
      data_te = np.concatenate((data3[tr_size7:],data8[tr_size8:]), axis=0)
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), 
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), 
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:]), axis=0)
      new_data=data1 #np.concatenate((data1,data4,data5,data6,data7,data2), axis=0)
  elif class_num==3:
      data_tr = np.concatenate((data3[:tr_size1],data8[:tr_size2],data2[:tr_size3]),axis=0)
      data_te = np.concatenate((data3[tr_size1:], data8[tr_size2:],data2[tr_size3:]), axis=0)  
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:]), axis=0)
      new_data=np.concatenate((data1,data5,data6,data7,data4), axis=0)
  elif class_num==4:
      data_tr = np.concatenate((data5[:tr_size1],data6[:tr_size2],data7[:tr_size3],data8[:tr_size4]),axis=0)
      data_te = np.concatenate((data5[tr_size1:], data6[tr_size2:],data7[tr_size3:],data8[tr_size4:]), axis=0) 
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),np.zeros((sam_size3, 1))
                           ), axis=1)
      label4 = np.concatenate((np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)),np.ones((sam_size4, 1))
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3],label4[:tr_size4]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:], label4[tr_size4:]), axis=0)     
      new_data=np.concatenate((data3,data4,data1,data2), axis=0)
  elif class_num==5:
      data_tr = np.concatenate((data1[:tr_size1],data2[:tr_size2],data3[:tr_size3],data4[:tr_size4],data5[:tr_size5]),axis=0)
      data_te = np.concatenate((data1[tr_size1:], data2[tr_size2:],data3[tr_size3:],data4[tr_size4:],data5[tr_size5:]), axis=0)
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1))
                           ), axis=1)
      label4 = np.concatenate((np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)),np.ones((sam_size4, 1)),np.zeros((sam_size4, 1))
                           ), axis=1)
      label5 = np.concatenate((np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1)),np.ones((sam_size5, 1))
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3],label4[:tr_size4],label5[:tr_size5]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:], label4[tr_size4:],label5[tr_size5:]), axis=0)
      new_data=np.concatenate((data6,data7,data8), axis=0)
  elif class_num==6:
      data_tr = np.concatenate((data1[:tr_size1],data2[:tr_size2],data3[:tr_size3],data4[:tr_size4],data5[:tr_size5],data6[:tr_size6]),axis=0)
      data_te = np.concatenate((data1[tr_size1:], data2[tr_size2:],data3[tr_size3:],data4[tr_size4:],data5[tr_size5:],data6[tr_size6:]), axis=0)    
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1))
                           ), axis=1)
      label4 = np.concatenate((np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)),np.ones((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1))
                           ), axis=1)
      label5 = np.concatenate((np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1)),np.ones((sam_size5, 1)),np.zeros((sam_size5, 1))
                           ), axis=1)
      label6 = np.concatenate((np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.ones((sam_size6, 1))
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3],label4[:tr_size4],label5[:tr_size5],label6[:tr_size6]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:], label4[tr_size4:],label5[tr_size5:],label6[tr_size6:]), axis=0)
      new_data=np.concatenate((data7,data8), axis=0)
  elif class_num==7:
      data_tr = np.concatenate((data1[:tr_size1],data2[:tr_size2],data3[:tr_size3],data4[:tr_size4],data5[:tr_size5],data6[:tr_size6],data7[:tr_size7]),axis=0)
      data_te = np.concatenate((data1[tr_size1:], data2[tr_size2:],data3[tr_size3:],data4[tr_size4:],data5[tr_size5:],data6[tr_size6:],data7[tr_size7:]), axis=0)    
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1))
                           ), axis=1)
      label4 = np.concatenate((np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)),np.ones((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1))
                           ), axis=1)
      label5 = np.concatenate((np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1)),np.ones((sam_size5, 1)),np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1))
                           ), axis=1)
      label6 = np.concatenate((np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.ones((sam_size6, 1)),np.zeros((sam_size6, 1))
                           ), axis=1)
      label7 = np.concatenate((np.zeros((sam_size7, 1)), np.zeros((sam_size7, 1)), np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.ones((sam_size7, 1))
                           ), axis=1)   
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3],label4[:tr_size4],label5[:tr_size5],label6[:tr_size6],label7[:tr_size7]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:], label4[tr_size4:],label5[tr_size5:],label6[tr_size6:],label7[tr_size7:]), axis=0)  
      new_data=data8
  elif class_num==8:
      data_tr = np.concatenate((data1[:tr_size1],data2[:tr_size2],data3[:tr_size3],data4[:tr_size4],data5[:tr_size5],data6[:tr_size6],data7[:tr_size7],data8[:tr_size8]),axis=0)
      data_te = np.concatenate((data1[tr_size1:], data2[tr_size2:],data3[tr_size3:],data4[tr_size4:],data5[tr_size5:],data6[tr_size6:],data7[tr_size7:],data8[tr_size8:]), axis=0)    
      label1 = np.concatenate((np.ones((sam_size1, 1)), np.zeros((sam_size1, 1)), np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1)),np.zeros((sam_size1, 1))
                           ), axis=1)
      label2 = np.concatenate((np.zeros((sam_size2, 1)), np.ones((sam_size2, 1)), np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1)),np.zeros((sam_size2, 1))
                           ), axis=1)
      label3 = np.concatenate((np.zeros((sam_size3, 1)), np.zeros((sam_size3, 1)), np.ones((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1)),np.zeros((sam_size3, 1))
                           ), axis=1)
      label4 = np.concatenate((np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)), np.zeros((sam_size4, 1)),np.ones((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1)),np.zeros((sam_size4, 1))
                           ), axis=1)
      label5 = np.concatenate((np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)), np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1)),np.ones((sam_size5, 1)),np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1)),np.zeros((sam_size5, 1))
                           ), axis=1)
      label6 = np.concatenate((np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)), np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1)),np.ones((sam_size6, 1)),np.zeros((sam_size6, 1)),np.zeros((sam_size6, 1))
                           ), axis=1)
      label7 = np.concatenate((np.zeros((sam_size7, 1)), np.zeros((sam_size7, 1)), np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.zeros((sam_size7, 1)),np.ones((sam_size7, 1)),np.zeros((sam_size7, 1))
                           ), axis=1)
      label8 = np.concatenate((np.zeros((sam_size8, 1)), np.zeros((sam_size8, 1)), np.zeros((sam_size8, 1)),np.zeros((sam_size8, 1)),np.zeros((sam_size8, 1)),np.zeros((sam_size8, 1)),np.zeros((sam_size8, 1)),np.ones((sam_size8, 1))
                           ), axis=1)
      label_tr = np.concatenate((label1[:tr_size1],label2[:tr_size2],label3[:tr_size3],label4[:tr_size4],label5[:tr_size5],label6[:tr_size6],label7[:tr_size7],label8[:tr_size8]),axis=0)
      label_te = np.concatenate((label1[tr_size1:], label2[tr_size2:], label3[tr_size3:], label4[tr_size4:],label5[tr_size5:],label6[tr_size6:],label7[tr_size7:],label8[tr_size8:]), axis=0)


  return data_tr,data_te, label_tr, label_te, row_size, col_size, class_num, num_channels,new_data