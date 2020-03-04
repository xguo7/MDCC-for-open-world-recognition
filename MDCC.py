# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:19:11 2019

@author: gxjco
"""

"""
Created on Mon Jan 29 23:22:25 2018

@author: gxjco
"""
from model import one_class_train,one_class_test,classify, cnn_model_train,cnn_model_test
from input_data import read_data  
import numpy as np

def find_new_data(pred_new,new,new_label):
    new_data=[]
    for i in range(len(new)):
        if pred_new[i]==new_label: new_data.append(new[i])    
    return np.array(new_data)

def class_acc(old_lab,new_lab,old_pred,new_pred):
    num_old=len(old_lab)/4000
    num_new=len(new_lab)/4000
    acc={}
    for i in range(int(num_old)):
       acc[i]=(100.0 * np.sum(old_pred[i*4000:i*4000+4000] ==old_lab[i*4000:i*4000+4000])
            /4000)
    for i in range(int(num_new)):
       acc[i+num_old]=(100.0 * np.sum(new_pred[i*4000:i*4000+4000] ==new_lab[i*4000:i*4000+4000])
            /4000)
    return acc
    
'''
data1,data2,data3,data4,data5,data6,data7,data8,lab1,lab2,lab3,lab4,lab5,lab6,lab7,lab8=read_data()
seqtest=[data3[1000:5000],data8[1000:5000],data1[1000:5000],data5[1000:5000],data4[1000:5000],data2[1000:5000],data6[1000:5000],data7[1000:5000]]
seqtest_lab=[lab3[1000:5000],lab8[1000:5000],lab1[1000:5000],lab5[1000:5000],lab4[1000:5000],lab2[1000:5000],lab6[1000:5000],lab7[1000:5000]]
seq=[data3[0:1000],data8[0:1000],data1[0:1000],data5[0:1000],data4[0:1000],data2[0:1000],data6[0:1000],data7[0:1000]]
seq_label=[3,8,1,5,4,2,6,7]

ref=[data3[0:500],data8[0:500]]
ref=np.array(ref)
ref=ref.reshape(ref.shape[0]*ref.shape[1],ref.shape[2],ref.shape[3])
num=2
def update_ref(ref_data,num,new_data):
    size=int(len(ref_data)/num)
    new_size=int(len(ref_data)/(num+1))
    new_ref_data=new_data[0:new_size]
    for i in range(num):
       new_ref_data=np.concatenate((new_ref_data,ref_data[i*size:i*size+new_size]),axis=0)
    num+=1
    return new_ref_data,num   
     
#phase 1:
mean_vec1,weibull_model1,old_data1,old_label1=cnn_model_train(seq[0],seq[1],'C:/Users/gxjco/Desktop/open_world2019/class1_0.00008/class1.ckpt',50)
pred_new1,pred_old1,score_new1,score_old1,acc_new1,acc_old1=cnn_model_test(seq[2],seq[2],'C:/Users/gxjco/Desktop/open_world2019/class1_0.00008/',weibull_model1,2,0.015,1000)
#phase 2:
new_data1=find_new_data(pred_new1,seq[2],2)

one_class_train(num,ref,new_data1,'C:/Users/gxjco/Desktop/open_world2019/class2_0.00008/class2.ckpt')
template_1=compute_template(num,new_data1,'C:/Users/gxjco/Desktop/open_world2019/class2_0.00008/')
ref,num=update_ref(ref,num,new_data1)
#phase 3:
pred,new_data2=one_class_test(2,seq[3],template_1,0.9,1000,'C:/Users/gxjco/Desktop/open_world2019/class2_0.00008/')
one_class_train(num,ref,new_data2,'C:/Users/gxjco/Desktop/open_world2019/class3_0.00008/class3.ckpt')
template_2=compute_template(num,new_data2,'C:/Users/gxjco/Desktop/open_world2019/class3_0.00008/')
ref,num=update_ref(ref,num,new_data2)
#phase 4:
pred,new_data3=one_class_test(3,seq[4],template_2,0.9,1000,'C:/Users/gxjco/Desktop/open_world2019/class3_0.00008/')
one_class_train(num,ref,new_data3,'C:/Users/gxjco/Desktop/open_world2019/class4_0.00008/class4.ckpt')
template_3=compute_template(num,new_data3,'C:/Users/gxjco/Desktop/open_world2019/class4_0.00008/')
ref,num=update_ref(ref,num,new_data3)
#phase 5:
pred,new_data4=one_class_test(4,seq[5],template_3,0.9,1000,'C:/Users/gxjco/Desktop/open_world2019/class4_0.00008/')
one_class_train(num,ref,new_data4,'C:/Users/gxjco/Desktop/open_world2019/class5_0.00008/class5.ckpt')
template_4=compute_template(num,new_data4,'C:/Users/gxjco/Desktop/open_world2019/class5_0.00008/')
ref,num=update_ref(ref,num,new_data4)
#phase 6:
pred,new_data5=one_class_test(5,seq[6],template_4,0.9,1000,'C:/Users/gxjco/Desktop/open_world2019/class5_0.00008/')
one_class_train(num,ref,new_data5,'C:/Users/gxjco/Desktop/open_world2019/class6_0.00008/class6.ckpt')
template_5=compute_template(num,new_data5,'C:/Users/gxjco/Desktop/open_world2019/class6_0.00008/')
ref,num=update_ref(ref,num,new_data5)
#phase 7:
pred,new_data6=one_class_test(6,seq[7],template_5,0.5,1000,'C:/Users/gxjco/Desktop/open_world2019/class6_0.00008/')
one_class_train(num,ref,new_data6,'C:/Users/gxjco/Desktop/open_world2019/class7_0.00008/class7.ckpt')
template_6=compute_template(num,new_data6,'C:/Users/gxjco/Desktop/open_world2019/class7_0.00008/')
ref,num=update_ref(ref,num,new_data6)

'''
def test_phase1(data,alpharank,threhold,dis_type,size,weibull_model1):
   pred,a,b,c,d,e,=cnn_model_test(data,data,'C:/Users/gxjco/Desktop/open_world2019/class1_0.00008/',weibull_model1,2,threhold,size)
   for i in range(len(pred)):
     if pred[i]==0: pred[i]=seq_label[0]
     if pred[i]==1: pred[i]=seq_label[1]
     if pred[i]==2: pred[i]=0
   return pred

def test_phase2(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase1(data,alpharank,threhold,dis_type,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(2,data[i*group_size:i*group_size+group_size],t1,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class2_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[2]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0 #new
   return pred

def test_phase3(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase2(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(3,data[i*group_size:i*group_size+group_size],t2,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class3_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[3]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0
   return pred

def test_phase4(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase3(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(4,data[i*group_size:i*group_size+group_size],t3,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class4_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[4]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0
   return pred

def test_phase5(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase4(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(5,data[i*group_size:i*group_size+group_size],t4,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class5_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[5]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0
   return pred

def test_phase6(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase5(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(6,data[i*group_size:i*group_size+group_size],t5,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class6_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[6]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0
   return pred

def test_phase7(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1):
   pred=test_phase6(data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
   for i in range(int(len(data)/group_size)):
    if pred[i*group_size]==0: 
       pred[i*group_size:i*group_size+group_size],new_data=one_class_test(7,data[i*group_size:i*group_size+group_size],t6,theta,group_size,'C:/Users/gxjco/Desktop/open_world2019/class7_0.00008/')
       if pred[i*group_size]==1: pred[i*group_size:i*group_size+group_size]=seq_label[7]
       if pred[i*group_size]==0: pred[i*group_size:i*group_size+group_size]=0
   return pred

def acc_phase1(size,alpharank,threhold,dis_type,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1]),axis=0)
    new_data= np.concatenate((seqtest[2],seqtest[3],seqtest[4],seqtest[5],seqtest[6],seqtest[7]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase1(old_data,alpharank,threhold,dis_type,size,weibull_model1)
    new_pred=test_phase1(new_data,alpharank,threhold,dis_type,size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase2(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2]),axis=0)
    new_data= np.concatenate((seqtest[3],seqtest[4],seqtest[5],seqtest[6],seqtest[7]),axis=0) #seqtest[3] 
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase2(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    new_pred=test_phase2(new_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase3(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3]),axis=0)
    new_data=np.concatenate((seqtest[4],seqtest[5],seqtest[6],seqtest[7]),axis=0) #seqtest[4]
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase3(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    new_pred=test_phase3(new_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase4(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4]),axis=0)
    new_data=np.concatenate((seqtest[5],seqtest[6],seqtest[7]),axis=0) #seqtest[5]
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase4(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    new_pred=test_phase4(new_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase5(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4],seqtest[5]),axis=0)
    new_data=np.concatenate((seqtest[6],seqtest[7]),axis=0)#seqtest[6] 
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4],seqtest_lab[5]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase5(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    new_pred=test_phase5(new_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase6(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4],seqtest[5],seqtest[6]),axis=0)
    new_data=seqtest[7] #np.concatenate((seqtest[7]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4],seqtest_lab[5],seqtest_lab[6]),axis=0)
    new_lab=np.zeros((len(new_data)))
    old_pred=test_phase6(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    new_pred=test_phase6(new_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_new=(100.0 * np.sum(new_pred ==new_lab)
            / new_lab.shape[0])
    acc_class=class_acc(old_lab,new_lab,old_pred,new_pred)
    return acc_old,acc_new,acc_class

def acc_phase7(alpharank,threhold,dis_type,group_size,theta,weibull_model1):
    old_data=np.concatenate((seqtest[0],seqtest[1],seqtest[2],seqtest[3],seqtest[4],seqtest[5],seqtest[6],seqtest[7]),axis=0)
    old_lab=np.concatenate((seqtest_lab[0],seqtest_lab[1],seqtest_lab[2],seqtest_lab[3],seqtest_lab[4],seqtest_lab[5],seqtest_lab[6],seqtest_lab[7]),axis=0)
    old_pred=test_phase7(old_data,alpharank,threhold,dis_type,theta,group_size,weibull_model1)
    acc_old=(100.0 * np.sum(old_pred ==old_lab)
            / old_lab.shape[0])
    acc_class=class_acc(old_lab,old_lab,old_pred,old_pred)
    return acc_old,acc_class

'''
acc1={}
acc1['old'],acc1['new'],acc_class1=acc_phase1(1,2,0.008,'eucos',weibull_model1) 
acc2={}
acc2['old'],acc2['new'],acc_class2=acc_phase2(2,0.008,'eucos',1,0.7,weibull_model1)

acc_class3={}
acc3={}
acc3['old'],acc3['new'],acc_class3['0.2']=acc_phase3(2,0.2,'eucos',1,0.7,weibull_model1)
acc3['old'],acc3['new'],acc_class3['0.1']=acc_phase3(2,0.1,'eucos',1,0.7,weibull_model1)
acc3['old'],acc3['new'],acc_class3['0.05']=acc_phase3(2,0.05,'eucos',1,0.7,weibull_model1)

acc_class4={}
acc4={}
acc4['old'],acc4['new'],acc_class4['0.2']=acc_phase4(2,0.2,'eucos',1,0.7,weibull_model1)
acc4['old'],acc4['new'],acc_class4['0.1']=acc_phase4(2,0.1,'eucos',1,0.7,weibull_model1)
acc4['old'],acc4['new'],acc_class4['0.05']=acc_phase4(2,0.05,'eucos',1,0.7,weibull_model1)

acc_class5={}
acc5={}
acc5['old'],acc5['new'],acc_class5['0.2']=acc_phase5(2,0.2,'eucos',1,0.7,weibull_model1)
acc5['old'],acc5['new'],acc_class5['0.1']=acc_phase5(2,0.1,'eucos',1,0.7,weibull_model1)
acc5['old'],acc5['new'],acc_class5['0.05']=acc_phase5(2,0.05,'eucos',1,0.7,weibull_model1)
'''
acc_class6={}
acc6={}
acc6['old'],acc6['new'],acc_class6['0.2']=acc_phase6(2,0.2,'eucos',1,0.5,weibull_model1)
acc6['old'],acc6['new'],acc_class6['0.1']=acc_phase6(2,0.1,'eucos',1,0.5,weibull_model1)
acc6['old'],acc6['new'],acc_class6['0.05']=acc_phase6(2,0.05,'eucos',1,0.5,weibull_model1)

acc_class7={}
acc7={}
acc7['old'],acc_class7['0.2']=acc_phase7(2,0.2,'eucos',1,0.5,weibull_model1)
acc7['old'],acc_class7['0.1']=acc_phase7(2,0.1,'eucos',1,0.5,weibull_model1)
acc7['old'],acc_class7['0.05']=acc_phase7(2,0.05,'eucos',1,0.5,weibull_model1)
'''  

#sensitivity
thr=[0.1]
acc=[]
for i in thr:
 acc3={}
 acc3['old'],acc3['new']=acc_phase3(2,0.008,'eucos',1,i,weibull_model1)
 acc.append(acc3)
'''











