# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:14:14 2018

@author: gxjco
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range

 

def compute_know_distances(mean_vector,feature,label):
   import scipy.spatial.distance as spd
   eucos_dist, eu_dist, cos_dist = [], [], []
   for i in range(len(mean_vector)):
     eucos, eu, cos= [], [], []
     for j in range(len(label)):
         if label[j]==i:
          eu+=[spd.euclidean(mean_vector[i], feature[j])]
          cos+=[spd.cosine(mean_vector[i], feature[j])]
          eucos+=[spd.euclidean(mean_vector[i], feature[j])/200. +
                               spd.cosine(mean_vector[i], feature[j])]
     eu_dist.append(eu)
     eucos_dist.append(eucos)
     cos_dist.append(cos)
   return eucos_dist, eu_dist, cos_dist

def weibull_tailfitting(meantrain_vec,distance_scores, tailsize = 20, distance_type = 'eucos'):   
   import libmr                        
   weibull_model = []
   for i in range(len(meantrain_vec)):
        model = {}
        model['distances_%s'%distance_type] = distance_scores[i]
        model['mean_vec'] = meantrain_vec[i]
        mr = libmr.MR()
        tailtofit = sorted(distance_scores[i])[-tailsize:]
        mr.fit_high(np.array(tailtofit), len(tailtofit))
        model['weibull_model'] = mr
        weibull_model.append(model)
   return weibull_model

def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    prob_scores, prob_unknowns = [], []
    for category in range(len(openmax_fc8)):
            prob_scores += [sp.exp(openmax_fc8[category])]
                    
    total_denominator = sp.sum(sp.exp(openmax_fc8)) + sp.exp(sp.sum(openmax_score_u))
    prob_scores = prob_scores/total_denominator
    prob_unknowns= [sp.exp(sp.sum(openmax_score_u))/total_denominator]
        
    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)
    modified_scores =  prob_scores.tolist() + [prob_unknowns]
    assert len(modified_scores) == len(openmax_fc8)+1
    return modified_scores

 
import scipy.spatial.distance as spd
import scipy as sp
def compute_distance(score, mean_vec, distance_type):
    query=score
    query_distance=0
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec, query)/200. + spd.cosine(mean_vec, query)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec, query)/200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec, query)
    return query_distance

def recalibrate_scores(weibull_model, scores,fc8, alpharank = 2, distance_type = 'eucos'):   
    ranked_list = scores.argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = sp.zeros(1000)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    openmax_fc8= []
    openmax_fc8_unknown = []
    for categoryid in range(len(fc8)):
            # get distance between current channel and mean vector
            category_weibull = weibull_model[distance_type][categoryid]
            distance = compute_distance(fc8, category_weibull['mean_vec'],distance_type)
            wscore = category_weibull['weibull_model'].w_score(distance)
            modified_fc8_score = fc8[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            openmax_fc8 += [modified_fc8_score]
            openmax_fc8_unknown += [fc8[categoryid] - modified_fc8_score ]
        # gather modified scores fc8 scores for each channel for the given image
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_fc8_unknown)    
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = scores.ravel() 
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)

def detect(dataset_score,dataset_fc8,weibull_model,alpharank):
    num=len(dataset_score)
    openmax_score={}
    softmax_score={}
    openmax_score['eucos'],softmax_score['eucos']=np.zeros((num,len(dataset_score[0])+1)),np.zeros((num,len(dataset_score[0])))
    openmax_score['eu'],softmax_score['eu']=np.zeros((num,len(dataset_score[0])+1)),np.zeros((num,len(dataset_score[0])))
    openmax_score['cos'],softmax_score['cos']=np.zeros((num,len(dataset_score[0])+1)),np.zeros((num,len(dataset_score[0])))
    for i in range(num):
        openmax_score['eucos'][i],softmax_score['eucos'][i]=recalibrate_scores(weibull_model, dataset_score[i],dataset_fc8[i], alpharank, distance_type = 'eucos')
        openmax_score['eu'][i],softmax_score['eu'][i]=recalibrate_scores(weibull_model, dataset_score[i],dataset_fc8[i], alpharank, distance_type = 'eu')
        openmax_score['cos'][i],softmax_score['cos'][i]=recalibrate_scores(weibull_model, dataset_score[i],dataset_fc8[i], alpharank, distance_type = 'cos')
    return openmax_score,softmax_score
  
def prediction(openmax_score_new,openmax_score_old,size,threhold,dis_type):
    label_new=np.empty(len(openmax_score_new[dis_type]))
    label_old=np.empty(len(openmax_score_old[dis_type])) 
    pred_new=np.empty(len(openmax_score_new[dis_type]))
    pred_old=np.empty(len(openmax_score_old[dis_type])) 
    new_label=len(openmax_score_new[dis_type][0])-1
    for i in range(len(openmax_score_new[dis_type])):
        if openmax_score_new[dis_type][i][new_label]>threhold: label_new[i]=new_label
        if openmax_score_new[dis_type][i][new_label]<=threhold: label_new[i]=np.argmax(openmax_score_new[dis_type][i])
    for i in range(len(openmax_score_old[dis_type])):
        if openmax_score_old[dis_type][i][new_label]>threhold: label_old[i]=new_label
        if openmax_score_old[dis_type][i][new_label]<=threhold: label_old[i]=np.argmax(openmax_score_old[dis_type][i])
    group_num_new=int(len(openmax_score_new['cos'])/size)
    for i in range(group_num_new):
         a=label_new[i*size:i*size+size].astype(int) 
         pred_new[i*size:i*size+size]=np.argmax(np.bincount(a))
    group_num_old=int(len(openmax_score_old['cos'])/size) 
    for i in range(group_num_old):
         b=label_old[i*size:i*size+size].astype(int) 
         pred_old[i*size:i*size+size]=np.argmax(np.bincount(b))
    accuracy_new=0
    accuracy_old=0
    for k in range(len(pred_new)):
        if pred_new[k]==new_label: accuracy_new+=1
    for k in range(len(pred_old)):
        if pred_old[k]!=new_label: accuracy_old+=1    
    acc_new=accuracy_new/len(pred_new)  
    acc_old=accuracy_old/len(pred_old)        
    return acc_new,acc_old,pred_new,pred_old

def pred_single(openmax_score,threhold,dis_type):
    label=np.empty(len(openmax_score[dis_type]))
    #pred=np.empty(len(openmax_score[dis_type])) 
    new_label=2
    for i in range(len(openmax_score[dis_type])):
        if openmax_score[dis_type][i][new_label]>threhold: label[i]=new_label
        if openmax_score[dis_type][i][new_label]<=threhold: label[i]=np.argmax(openmax_score[dis_type][i])   
    return label   


def cnn_model_train(data_a,data_b,path,tailsize):
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  num_labels=2
  data_size=min(len(data_a),len(data_b))
  image_size1=512
  num_channels=4
  
  batch_size =50  
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20
  label_a = np.concatenate((np.ones((len(data_a), 1)), np.zeros((len(data_a), 1)), 
                           ), axis=1)
  label_b = np.concatenate((np.zeros((len(data_b), 1)), np.ones((len(data_b), 1)), 
                           ), axis=1)
  train_dataset= np.concatenate((data_a[:data_size],data_b[:data_size]),axis=0)
  test_dataset= np.concatenate((data_a[:data_size],data_b[:data_size]), axis=0)
  train_labels= np.concatenate((label_a[:data_size],label_b[:data_size]),axis=0)
  test_labels= np.concatenate((label_a[:data_size],label_b[:data_size]), axis=0)
  with graph.as_default():
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(batch_size,image_size1, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_t_dataset = tf.constant(train_dataset)
    tf_t_dataset = tf.cast(tf_t_dataset, tf.float32)
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_dataset = tf.cast(tf_test_dataset, tf.float32)
    tf_valid_dataset = tf.constant(test_dataset)
    tf_valid_dataset = tf.cast(tf_valid_dataset, tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))   
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) 

    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.tanh(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.tanh(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.dropout(hidden3, 0.5)
      return tf.matmul(hidden4, layer4_weights) + layer4_biases,hidden3

    logits,features = model(tf_train_dataset)
    logits_test,features_test = model(tf_test_dataset)
    logits_valid,features_valid = model(tf_valid_dataset)
    logits_,features_ = model(tf_t_dataset)
   
    regularizer=tf.contrib.layers.l2_regularizer(0.000005)
    regularization=regularizer(layer1_weights)+regularizer(layer2_weights)+regularizer(layer3_weights)+regularizer(layer4_weights)
    
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))+regularization

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(logits_test)
    valid_prediction = tf.nn.softmax(logits_valid)
    t_prediction=tf.nn.softmax(logits_)

  # In[ ]:


  num_steps =200000

  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)     
   max_acc=0
   for step in range(num_steps):
       offset = (step * batch_size) % (train_labels.shape[0]-batch_size)
       batch_data = train_dataset[offset:(offset + batch_size), :, :]
       batch_labels = train_labels[offset:(offset + batch_size), :]
       feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
       _, l, predictions= session.run(
         [optimizer, loss, train_prediction],feed_dict=feed_dict)
       if (step % 50 == 0):
         print('Minibatch loss at step %d: %f' % (step, l))           
         print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
         pred=session.run(valid_prediction,feed_dict={tf_valid_dataset:test_dataset, keep_prob:1})
         m=accuracy(pred, test_labels)
         print('Validation accuracy: %.1f%%' % m)    
         if m>max_acc:
            max_acc=m
            saver.save(session,path,global_step=step)
   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
   print('Training accuracy: %.1f%%' % accuracy(t_prediction.eval(), train_labels))    
   feature_t=features_.eval()
   pred_t=np.argmax(t_prediction.eval(), 1)    
   score_t=t_prediction.eval()
   v_t=logits_.eval()
   final_data=[]
   final_label=[]
   final_x=[]
   final_score=[]
   for m in range(num_labels):
       s=[]
       p=[]
       f=[]
       l=[]
       final_data.append(s)
       final_label.append(p)
       final_x.append(f)
       final_score.append(l)      
   for i in range(len(v_t)):
        if pred_t[i]==np.argmax(train_labels[i]):
            final_data[pred_t[i]].append(v_t[i])
            final_label[pred_t[i]].append(pred_t[i])
            final_score[pred_t[i]].append(score_t[i])
            final_x[pred_t[i]].append(feature_t[i])
                   
            
   mean_vec=np.zeros((len(final_data),len(final_data)))
   for i in range(len(final_data)):
        mean_vec[i,:]=sum(final_data[i])/len(final_data[i])        
   con=[]   
   con_=[]
   con_x=[]
   con_score=[]
   for i in range(num_labels):        
       con.extend(final_data[i])      
       con_.extend(final_label[i])      
       con_x.extend(final_x[i])      
       con_score.extend(final_score[i])
    
   data=np.zeros((len(con),num_labels))
   label=np.zeros(len(con_))
   x=np.zeros((len(con_x),len(con_x[0])))
   score=np.zeros((len(con_score),len(con_score[0])))
   for i in range(len(con)):
        data[i]=con[i]
        label[i]=con_[i]
        x[i]=con_x[i]
        score[i]=con_score[i]
   fc8=data
   distance={}
   distance['eucos'], distance['eu'],distance['cos']= compute_know_distances(mean_vec,fc8,label)
   weibull_model={}
   weibull_model['eucos']= weibull_tailfitting(mean_vec,distance['eucos'],tailsize, distance_type = 'eucos')
   weibull_model['eu']= weibull_tailfitting(mean_vec,distance['eu'],tailsize, distance_type = 'eu')
   weibull_model['cos']= weibull_tailfitting(mean_vec,distance['cos'],tailsize, distance_type = 'cos')
   return mean_vec,weibull_model,test_dataset,test_labels
  
    

def cnn_model_test(data_new,data_old,path,weibull_model,alpharank,threhold,size):
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  num_labels=2
  image_size1=512
  num_channels=4
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20
  
  new_dataset= data_new
  old_dataset= data_old
  with graph.as_default():
    tf_old_dataset = tf.constant(old_dataset)
    tf_old_dataset = tf.cast(tf_old_dataset, tf.float32)
    tf_new_dataset = tf.constant(new_dataset)
    tf_new_dataset = tf.cast(tf_new_dataset, tf.float32)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
    
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) 

    # Model.
    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.tanh(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.tanh(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.dropout(hidden3, 0.5)
      return tf.matmul(hidden4, layer4_weights) + layer4_biases,hidden3

    # Training computation.
    logits_old,features_old = model(tf_old_dataset)
    logits_new,features_new = model(tf_new_dataset)    
    old_prediction=tf.nn.softmax(logits_old)
    new_prediction=tf.nn.softmax(logits_new)
  # In[ ]:
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)
  
   check_point_path = path
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
   saver.restore(session,ckpt.model_checkpoint_path)
   print("Model restored.") 
   score_old=old_prediction.eval()
   score_new=new_prediction.eval()
   score_old=old_prediction.eval()
   v_old=logits_old.eval()
   v_new=logits_new.eval()
   openmax_score_new,softmax_score_new=detect(score_new,v_new,weibull_model,alpharank)
   openmax_score_old,softmax_score_old=detect(score_old,v_old,weibull_model,alpharank) 
   acc_new,acc_old,pred_new,pred_old=prediction(openmax_score_new,openmax_score_old,size,threhold,'cos')   
   return pred_new,pred_old,openmax_score_new,openmax_score_old,acc_new,acc_old

def classify(data_new,data_old,path,weibull_model,alpharank,threhold,size):
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  num_labels=2
  image_size1=512
  num_channels=4
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20
  
  new_dataset= data_new
  old_dataset= data_old
  with graph.as_default():
    tf_old_dataset = tf.constant(old_dataset)
    tf_old_dataset = tf.cast(tf_old_dataset, tf.float32)
    tf_new_dataset = tf.constant(new_dataset)
    tf_new_dataset = tf.cast(tf_new_dataset, tf.float32)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
    
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) 

    # Model.
    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.tanh(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.tanh(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.dropout(hidden3, 0.5)
      return tf.matmul(hidden4, layer4_weights) + layer4_biases,hidden3

    # Training computation.
    logits_old,features_old = model(tf_old_dataset)
    logits_new,features_new = model(tf_new_dataset)    
    old_prediction=tf.nn.softmax(logits_old)
    new_prediction=tf.nn.softmax(logits_new)
  # In[ ]:
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)
  
   check_point_path = path
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
   saver.restore(session,ckpt.model_checkpoint_path)
   print("Model restored.") 
   score_old=old_prediction.eval()
   score_new=new_prediction.eval()
   score_old=old_prediction.eval()
   v_old=logits_old.eval()
   v_new=logits_new.eval()
   openmax_score_new,softmax_score_new=detect(score_new,v_new,weibull_model,alpharank)
   openmax_score_old,softmax_score_old=detect(score_old,v_old,weibull_model,alpharank) 
   acc_new,acc_old,pred_new,pred_old=prediction(openmax_score_new,openmax_score_old,size,threhold,'eucos')   
   return pred_new

def root(data,path):
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  num_labels=2
  image_size1=512
  num_channels=4
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20
  with graph.as_default():
    tf_dataset = tf.constant(data)
    tf_dataset = tf.cast(tf_dataset, tf.float32)


    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
    
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels])) 

    # Model.
    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.tanh(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.tanh(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.tanh(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.dropout(hidden3, 0.5)
      return tf.matmul(hidden4, layer4_weights) + layer4_biases,hidden3

    # Training computation.
    logits,features= model(tf_dataset)    
    prediction=tf.nn.softmax(logits)
  # In[ ]:
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)
  
   check_point_path = path
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
   saver.restore(session,ckpt.model_checkpoint_path)
   print("Model restored.") 
   score=prediction.eval()
   pred=np.argmax(score,1)
   pred[:]=np.argmax(np.bincount(pred.astype(int))) 
   return pred






def one_class_train(num,ref,new_data,model_path):
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  num_labels=num
  ref_size=int(len(ref)/num)
  image_size1=512
  num_channels=4
  size=len(new_data)
  ref_batch_size =100 #int(ref_size/800)  
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20

  if num==2:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), 
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), 
                           ), axis=1)
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.ones((size, 1)), 
                           ), axis=1)                         
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size]),axis=0)    
  if num==3:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), 
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), 
                           ), axis=1)
    label_3 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), 
                           ), axis=1)      
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.ones((size, 1)), 
                           ), axis=1)               
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size],label_3[:ref_size]),axis=0)  
  if num==4:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),
                           ), axis=1)
    label_3 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),
                           ), axis=1)      
    label_4 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)),
                           ), axis=1) 
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.ones((size, 1)),
                           ), axis=1)               
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size],label_3[:ref_size],label_4[:ref_size]),axis=0)     
  if num==5:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_3 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)      
    label_4 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1) 
    label_5 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)  
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)),np.ones((size, 1)),
                           ), axis=1)              
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size],label_3[:ref_size],label_4[:ref_size],label_5[:ref_size]),axis=0)      
  if num==6:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_3 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)      
    label_4 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1) 
    label_5 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)  
    label_6 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.ones((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)   
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)),np.zeros((size, 1)),np.ones((size, 1)),
                           ), axis=1)            
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size],label_3[:ref_size],label_4[:ref_size],label_5[:ref_size],label_6[:ref_size]),axis=0)        
  if num==7:
    label_1 = np.concatenate((np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_2 = np.concatenate((np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)
    label_3 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)      
    label_4 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1) 
    label_5 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.ones((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)  
    label_6 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.ones((ref_size, 1)),np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1)   
    label_7 = np.concatenate((np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)), np.zeros((ref_size, 1)),np.zeros((ref_size, 1)),np.ones((ref_size, 1)),np.zeros((ref_size, 1)),
                           ), axis=1) 
    label_t = np.concatenate((np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)), np.zeros((size, 1)),np.zeros((size, 1)),np.zeros((size, 1)),np.ones((size, 1))
                           ), axis=1)            
    ref_train_labels= np.concatenate((label_1[:ref_size],label_2[:ref_size],label_3[:ref_size],label_4[:ref_size],label_5[:ref_size],label_6[:ref_size],label_7[:ref_size]),axis=0)           
  ref_train_dataset= ref
  target_train_dataset=new_data
  ref_train_dataset=np.concatenate((ref_train_dataset,target_train_dataset[:ref_size]),axis=0)
  ref_train_labels=np.concatenate((ref_train_labels,label_t[:ref_size]),axis=0)
  
  with graph.as_default():
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(ref_batch_size,image_size1, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(ref_batch_size, num+1))
  
    
    tf_ref_dataset = tf.constant(ref_train_dataset)
    tf_ref_dataset = tf.cast(tf_ref_dataset, tf.float32)
    tf_new_dataset = tf.constant(target_train_dataset)
    tf_new_dataset = tf.cast(tf_new_dataset, tf.float32)
    #tf_test_dataset = tf.constant(test_dataset)
    #tf_test_dataset = tf.cast(tf_test_dataset, tf.float32)
    #tf_valid_dataset = tf.constant(test_dataset)
    #tf_valid_dataset = tf.cast(tf_valid_dataset, tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))   
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels+1], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels+1])) 

    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.elu(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.elu(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.elu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.elu(tf.matmul(hidden3, layer4_weights) + layer4_biases)
      return hidden4,hidden3

    logits,features = model(tf_train_dataset)
    logits_test,features_test = model(tf_new_dataset)
    logits_valid,features_valid = model(tf_ref_dataset)
    #logits_,features_ = model(tf_t_dataset)
   
    regularizer=tf.contrib.layers.l2_regularizer(0.000005)
    regularization=regularizer(layer1_weights)+regularizer(layer2_weights)+regularizer(layer3_weights)+regularizer(layer4_weights)
    def compute_target_loss(feature):
        l=0.0
        for i in range(feature.shape[0]):
            z=tf.subtract(feature[i],tf.reduce_mean(tf.subtract(tf.reduce_sum(feature,0),feature[i])))
            m=z*tf.ones([1, z.shape[0]])
            l=tf.add(l,tf.matmul(m,tf.transpose(m)))
        return l/(num_hidden*ref_batch_size)
            
    ref_loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))+0.001*regularization
    target_loss = compute_target_loss(features)+0.001*regularization
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.002)
    
    ref_gradient_all = optimizer.compute_gradients(ref_loss)  # gradient of network (with NoneType)
    ref_grads_vars = [v for (g, v) in ref_gradient_all if g is not None]  # all variable that has gradients
    ref_gradient = optimizer.compute_gradients(ref_loss, ref_grads_vars)

    target_gradient_all = optimizer.compute_gradients( target_loss)  # gradient of network (with NoneType)
    target_grads_vars = [v for (g, v) in target_gradient_all if g is not None]  # all variable that has gradients
    target_gradient = optimizer.compute_gradients(target_loss, target_grads_vars)
    
    grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                     for (g, v) in ref_gradient]
    apply = optimizer.apply_gradients(grads_holder)

  # In[ ]:
  num_steps =10000
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)     
   max_l=120
   for step in range(num_steps):
      for idx in range(19):
       ref_offset =int((ref_train_labels.shape[0]-100)/18) # 70 #(step * ref_batch_size) % (ref_train_labels.shape[0]-ref_batch_size)
       ref_batch_data = ref_train_dataset[idx*ref_offset:(idx*ref_offset + ref_batch_size), :, :]
       ref_batch_labels = ref_train_labels[idx*ref_offset:(idx*ref_offset + ref_batch_size), :]
       target_offset =50 # (step * ref_batch_size) % (target_train_dataset.shape[0]-ref_batch_size)
       target_batch_data = target_train_dataset[idx*target_offset:(idx*target_offset + ref_batch_size), :, :]  

       ref_feed_dict = {tf_train_dataset: ref_batch_data, tf_train_labels: ref_batch_labels}
       target_feed_dict = {tf_train_dataset: target_batch_data}
       ref_grad_ = session.run(ref_gradient, feed_dict=ref_feed_dict)
       target_grad_ = session.run(target_gradient, feed_dict=target_feed_dict) 
       def tuple_adds(a,b):
           return tuple(map(lambda x: (x[0]+x[1])/2, zip(a,b)))  
       grads = list(map(lambda x: tuple_adds(x[0],x[1]), zip(ref_grad_,target_grad_)))        
       grads.append(ref_grad_[-1])
       sum_ = {}
       for i in range(len(grads_holder)):
           k = grads_holder[i][0]
           sum_[k] =grads[i][0] 
       _= session.run(apply,feed_dict=sum_)
       ref_l=session.run(ref_loss,feed_dict= ref_feed_dict)
       target_l=session.run(target_loss,feed_dict= target_feed_dict)       
       if (step % 50 == 0):
         print('Minibatch reg_loss at step %d: %f' % (step, ref_l))       
         print('Minibatch target_loss at step %d: %f' % (step, target_l))    
         if ref_l+target_l<max_l:
            max_l=ref_l+target_l
            saver.save(session,model_path,global_step=step)
   return features_test.eval()       
            




def one_class_test(num_labels,new_data,templates,theta,batch_size,model_path): 
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  image_size1=512
  num_channels=4
 
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20          

  target_train_dataset=new_data
  
  with graph.as_default():
  
    tf_new_dataset = tf.constant(target_train_dataset)
    tf_new_dataset = tf.cast(tf_new_dataset, tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))   
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels+1], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels+1])) 

    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.elu(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.elu(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.elu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.elu(tf.matmul(hidden3, layer4_weights) + layer4_biases)
      return hidden4,hidden3

    logits,features = model(tf_new_dataset)
    
    def classifier(templates,feature,theta):
       mean=np.mean(templates,axis=0)
       distance=[]
       for i in range(len(templates)):
           distance.append(np.sqrt(np.sum(np.square(templates[i] - mean))))
       distance=sorted(distance) 
       max_distance=max(distance[0:int(len(templates)*theta)])
       if np.sqrt(np.sum(np.square(feature - mean)))<=max_distance:
           return 1
       else: return 0   
    def batch(feature,templates):
     class_result=np.zeros(len(feature))
     for i in range(len(feature)):
        class_result[i]=classifier(templates,feature[i],theta)
     if sum(class_result)>len(feature)*1/2:
        return np.ones(len(feature))
     else: return np.zeros(len(feature))
  # In[ ]:
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)  
   check_point_path = model_path
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
   saver.restore(session,ckpt.model_checkpoint_path)
   print("Model restored.") 
   feature=features.eval()
   pred=np.zeros(len(feature))
   for i in range(int(len(feature)/batch_size)):
        pred[i*batch_size:i*batch_size+batch_size]=batch(feature[i*batch_size:i*batch_size+batch_size],templates)
   new=np.zeros((int(tf_new_dataset.shape[0]-sum(pred)),512,4))
   n=0
   for i in range(len(feature)):       
       if pred[i]==0:
           new[n]=new_data[i]
           n+=1
   return pred,new
    
def compute_template(num_labels,new_data,model_path):
  image_size1=512
  num_channels=4
 
  patch_size1 =20
  patch_size2 = 20
  depth1 = 5
  depth2 =5
  num_hidden = 10
  graph = tf.Graph()
  stride1 = 1
  stride2 = 1
  p_stride1=8
  p_stride2=8
  pool1=30
  pool2=20          

  target_train_dataset=new_data
  
  with graph.as_default():
  
    tf_new_dataset = tf.constant(target_train_dataset)
    tf_new_dataset = tf.cast(tf_new_dataset, tf.float32)

    keep_prob = tf.placeholder(tf.float32)
    
    layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size2, depth1, depth2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))   
    
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size1 // stride1//stride2//p_stride1//p_stride2 * depth2, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels+1], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels+1])) 

    def model(data):
      conv1 = tf.nn.conv1d(data, layer1_weights, stride1, padding='SAME')      
      hidden10 = tf.nn.elu(conv1 + layer1_biases)
      hidden11=tf.nn.pool(hidden10, [pool1],'AVG',strides=[p_stride1], padding='SAME')
      hidden12 = tf.nn.dropout(hidden11, 0.5)
      
      conv2 = tf.nn.conv1d(hidden12, layer2_weights,stride2, padding='SAME')
      hidden20 = tf.nn.elu(conv2 + layer2_biases)
      hidden21=tf.nn.pool(hidden20, [pool2],'AVG',strides=[p_stride2], padding='SAME')
      hidden22 = tf.nn.dropout(hidden21, 0.5)
      
      shape = hidden21.get_shape().as_list()
      reshape = tf.reshape(hidden22, [shape[0], shape[1] * shape[2]])
    
      hidden3 = tf.nn.elu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      hidden4 = tf.nn.elu(tf.matmul(hidden3, layer4_weights) + layer4_biases)
      return hidden4,hidden3

    logits,features = model(tf_new_dataset)
    
  # In[ ]:
  with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print('Initialized')
   saver=tf.train.Saver(max_to_keep=1)  
   check_point_path = model_path
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
   saver.restore(session,ckpt.model_checkpoint_path)
   print("Model restored.") 
   feature=features.eval()
   return feature