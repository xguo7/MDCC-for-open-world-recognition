# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:25:39 2019

@author: gxjco
"""

from retrain_model import cnn_model_train,cnn_model_test
from read_data import input_data  
import numpy as np

def find_new_data(pred_new,new,new_label):
    new_data=[]
    for i in range(len(new)):
        if pred_new[i]==new_label: new_data.append(new[i])    
    return np.array(new_data)
'''
data1,data2,data3,data4,data5,data6,lab1,lab2,lab3,lab4,lab5,lab6=input_data()
seqtest=[data1[80:160],data2[80:160],data3[80:160],data4[80:160],data5[80:160],data6[80:160]]
seqtest_lab=[lab1[80:160],lab2[80:160],lab3[80:160],lab4[80:160],lab5[80:160],lab6[80:160]]
seq=[data1[0:80],data2[0:80],data3[0:80],data4[0:80],data5[0:80],data6[0:80]]
seq_label=[1,2,3,4,5,6]

#phase 1:
mean_vec1,weibull_model1,old_data1,old_label1=cnn_model_train(np.concatenate((seqtest[0],seqtest[1]),axis=0),'C:/Users/gxjco/Desktop/open_word_twitter/class1_re/class1.ckpt',50)
#pred_new1,pred_old1,score_new1,score_old1,acc_new1,acc_old1=cnn_model_test(seq[2],seq[2],'C:/Users/gxjco/Desktop/open_word_twitter/class1_0.00008/',weibull_model1,2,0.015,1000)
#phase 2:

#new_data1=find_new_data(pred_new1,seq[2],2)
mean_vec2,weibull_model2,old_data2,old_label2=cnn_model_train(np.concatenate((seqtest[0],seqtest[1],seqtest[2]),axis=0),'C:/Users/gxjco/Desktop/open_word_twitter/class2_re/class2.ckpt',50)
#pred_new2,pred_old2,score_new2,score_old2,acc_new2,acc_old2=cnn_model_test(seq[3],seq[3],'C:/Users/gxjco/Desktop/open_word_twitter/class2_0.00008/',weibull_model2,2,0.015,1000)
#phase 3:
#new_data2=find_new_data(pred_new1,seq[3],3)
mean_vec3,weibull_model3,old_data3,old_label3=cnn_model_train(np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3]),axis=0),'C:/Users/gxjco/Desktop/open_word_twitter/class3_re/class3.ckpt',50)
#pred_new3,pred_old3,score_new3,score_old3,acc_new3,acc_old3=cnn_model_test(seq[4],seq[4],'C:/Users/gxjco/Desktop/open_word_twitter/class3_0.00008/',weibull_model3,2,0.015,1000)
#phase 4:
#new_data3=find_new_data(pred_new1,seq[4],4)
mean_vec4,weibull_model4,old_data4,old_label4=cnn_model_train(np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4]),axis=0),'C:/Users/gxjco/Desktop/open_word_twitter/class4_re/class4.ckpt',50)
#pred_new4,pred_old3,score_new3,score_old3,acc_new3,acc_old3=cnn_model_test(seq[5],seq[5],'C:/Users/gxjco/Desktop/open_word_twitter/class4_0.00008/',weibull_model4,2,0.015,1000)
#phase 5:
#new_data4=find_new_data(pred_new1,seq[5],5)
mean_vec5,weibull_model5,old_data5,old_label5=cnn_model_train(np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4],seqtest[5]),axis=0),'C:/Users/gxjco/Desktop/open_word_twitter/class5_re/class5.ckpt',50)

'''

def test_phase1(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_word_twitter/class1_re/',weibull_model1,2,threhold,size,2)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     else:
        if pred[i]==1: pred[i]=seq_label[1]
        else:
           if pred[i]==2: pred[i]=0
   return pred

def test_phase2(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_word_twitter/class2_re/',weibull_model2,2,threhold,size,3)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     else:
        if pred[i]==1: pred[i]=seq_label[1]
        else:
           if pred[i]==2: pred[i]=seq_label[2]
           else:
             if pred[i]==3: pred[i]=0
   return pred

def test_phase3(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_word_twitter/class3_re/',weibull_model3,2,threhold,size,4)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     else:
        if pred[i]==1: pred[i]=seq_label[1]
        else:
           if pred[i]==2: pred[i]=seq_label[2]
           else:
             if pred[i]==3: pred[i]=seq_label[3]
             else:
               if pred[i]==4: pred[i]=0
   return pred

def test_phase4(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_word_twitter/class4_re/',weibull_model4,2,threhold,size,5)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     else:
        if pred[i]==1: pred[i]=seq_label[1]
        else:
           if pred[i]==2: pred[i]=seq_label[2]
           else:
             if pred[i]==3: pred[i]=seq_label[3]
             else:
               if pred[i]==4: pred[i]=seq_label[4]
               else:
                 if pred[i]==5: pred[i]=0
   return pred

def test_phase5(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_word_twitter/class5_re/',weibull_model5,2,threhold,size,6)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     else:
        if pred[i]==1: pred[i]=seq_label[1]
        else:
           if pred[i]==2: pred[i]=seq_label[2]
           else:
             if pred[i]==3: pred[i]=seq_label[3]
             else:
               if pred[i]==4: pred[i]=seq_label[4]
               else:
                 if pred[i]==5: pred[i]=seq_label[5]
                 else:
                   if pred[i]==6: pred[i]=0
   return pred



def acc_phase1(size,alpharank,threhold,dis_type,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1]),axis=0)
    new_data= np.concatenate((seqtest[2],seqtest[3],seqtest[4],seqtest[5]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase1(old_data,alpharank,threhold,dis_type,size,weibull_model1)
    new_pred=test_phase1(new_data,alpharank,threhold,dis_type,size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    return acc_old,acc_new

def acc_phase2(alpharank,threhold,dis_type,group_size,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2]),axis=0)
    new_data=np.concatenate((seqtest[3],seqtest[4],seqtest[5]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase2(old_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    new_pred=test_phase2(new_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    return acc_old,acc_new

def acc_phase3(alpharank,threhold,dis_type,group_size,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3]),axis=0)
    new_data=np.concatenate((seqtest[4],seqtest[5]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase3(old_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    new_pred=test_phase3(new_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    return acc_old,acc_new

def acc_phase4(alpharank,threhold,dis_type,group_size,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4]),axis=0)
    new_data=seqtest[5]#np.concatenate((seqtest[5]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase4(old_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    new_pred=test_phase4(new_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    return acc_old,acc_new

def acc_phase5(alpharank,threhold,dis_type,group_size,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4],seqtest[5]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4],seqtest_lab[5]),axis=0)
    old_pred=test_phase5(old_data,alpharank,threhold,dis_type,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    return acc_old




acc1={}
acc1['old'],acc1['new']=acc_phase1(1,2,0.1,'eucos',weibull_model1)
acc2={}
acc2['old'],acc2['new']=acc_phase2(2,0.1,'eucos',1,weibull_model2)
acc3={}
acc3['old'],acc3['new']=acc_phase3(2,0.1,'eucos',1,weibull_model3)
acc4={}
acc4['old'],acc4['new']=acc_phase4(2,0.1,'eucos',1,weibull_model4)
acc5={}
acc5['old']=acc_phase5(2,0.1,'eucos',1,weibull_model5)
