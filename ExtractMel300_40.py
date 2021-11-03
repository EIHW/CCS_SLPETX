#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:01:12 2021

@author: tianhao
"""


import wave
import numpy as np
import python_speech_features as ps
import os
# import glob
import pickle
# import librosa
#import base
#import sigproc
eps = 1e-5

train_path = "E:\\samnew\\data\\train\\"
test_path = "E:\\samnew\\data\\test\\"
dev_path = "E:\\samnew\\data\\dev\\"

# root_path = "/home/tianhao/yth/dist_conf/data/"

negnum = 215
posnum = 71
train_emt = {'neg':0,'pos':0}
# train_label = np.empty((train_num,1), dtype = np.int8)

def generate_label(emotion):
    if (emotion == 'neg'):
        label = 0
    elif (emotion == 'pos'):
        label = 1

    return label

def read_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate

def abstract_line(line):
#    print(line)
    words = line.split('.')[0].split('_')[1]
#    print(len(words))
#    print(words)
    # path = words[1]
    label = words

    return label

def load_data():
    f = open('./zscoretra30040.pkl','rb')
    mean1,std1,mean2,std2,mean3,std3 = pickle.load(f)
    return mean1,std1,mean2,std2,mean3,std3

# def load_data1():
#     f = open('./zscoretes300.pkl','rb')
#     mean4,std4,mean5,std5,mean6,std6 = pickle.load(f)
#     return mean4,std4,mean5,std5,mean6,std6

# def load_data2():
#     f = open('./zscoreval300.pkl','rb')
#     mean7,std7,mean8,std8,mean9,std9 = pickle.load(f)
#     return mean7,std7,mean8,std8,mean9,std9
 
# def read_ComparE():
    # trnum = 286
# testactualnum = 208
# deactualnum = 231

tenum = 208
denum = 231


train_num = 654
test_num = 470
dev_num = 510

filter_num = 40

pernums_test = np.arange(tenum)    #remerber each utterance contain how many segments
pernums_dev = np.arange(denum)

mean1,std1,mean2,std2,mean3,std3 = load_data()
# mean4,std4,mean5,std5,mean6,std6 = load_data1()
# mean7,std7,mean8,std8,mean9,std9 = load_data2()

# negnum = 1000
# posnum = 1000

# train_label = np.empty((train_num,1), dtype = np.int8)
test_label = []
valid_label = []
# Test_label = np.empty((test_num,1), dtype = np.int8)
# Valid_label = np.empty((dev_num,1), dtype = np.int8)
train_data = np.empty((train_num,300,filter_num,3),dtype = np.float32)
test_data = np.empty((test_num,300,filter_num,3),dtype = np.float32)
valid_data = np.empty((dev_num,300,filter_num,3),dtype = np.float32)


train_emt = {'neg':0,'pos':0}
test_emt = {'neg':0,'pos':0}
dev_emt = {'neg':0,'pos':0}
    
# def train():   
train_num = 0
train_label = []
train_datalist = []
for speaker in os.listdir(train_path):
    path = train_path + speaker
    # print(speaker)
    label = abstract_line(speaker)
    if (label == 'neg') or (label == 'pos'):
        train_datalist.append([path,label])
# print (train_datalist)
for i, datalist in enumerate (train_datalist):
    path = datalist[0]
    label = datalist[1]
    data, time, rate = read_file(path)
    mel_spec = ps.logfbank(data,rate,nfilt = filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    
    time = mel_spec.shape[0] 
    if(time <= 300):
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
        
        part = np.pad(part,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta11 = np.pad(delta11,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta21 = np.pad(delta21,((0,300 - time),(0,0)),'constant',constant_values = 0)
        train_data[train_num,:,:,0] = part
        train_data[train_num,:,:,1] = delta11
        train_data[train_num,:,:,2] = delta21
        
        em = generate_label(label)
        train_label.append(em)
        train_emt[label] = train_emt[label] + 1
        train_num = train_num + 1
        
    else:
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
        
        frames = divmod(time,300)[0]
        frames_yu = divmod(time,300)[1]
        if frames_yu > 100:
            frames += 1
            part = np.pad(part,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta11 = np.pad(delta11,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta21 = np.pad(delta21,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
        for j in range(frames):
            begin = 300*j
            end = begin + 300
            part_seg = part[begin:end,:]
            delta11_seg = delta11[begin:end,:]
            delta21_seg = delta21[begin:end,:]
        
            train_data[train_num,:,:,0] = part_seg
            train_data[train_num,:,:,1] = delta11_seg
            train_data[train_num,:,:,2] = delta21_seg 
 
            em = generate_label(label)
            train_label.append(em)
            train_emt[label] = train_emt[label] + 1
            train_num = train_num + 1
        
print (train_num)
print (len(train_label))
#     return train_data,train_label

# def test():
    
test_num = 0
tenum = 0
test_datalist = []
Test_label = []
test_label = []
sumpath = []
total = []

for speaker in os.listdir(test_path):
    path = test_path + speaker
    # print(speaker)
    label = abstract_line(speaker)
    if (label == 'neg') or (label == 'pos'):
        test_datalist.append([path,label])
# print (test_datalist)
for i, datalist in enumerate (test_datalist):
    path = datalist[0]
    label = datalist[1]
    data, time, rate = read_file(path)
    mel_spec = ps.logfbank(data,rate,nfilt = filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    
    time = mel_spec.shape[0]
    em = generate_label(label)
    test_label.append(em)
    if(time <= 300):
        pernums_test[tenum] = 1
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
        
        part = np.pad(part,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta11 = np.pad(delta11,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta21 = np.pad(delta21,((0,300 - time),(0,0)),'constant',constant_values = 0)
        test_data[test_num,:,:,0] = part
        test_data[test_num,:,:,1] = delta11 
        test_data[test_num,:,:,2] = delta21 
        
        em = generate_label(label)
        Test_label.append(em)
        sumpath.append(path)
        total.append([path,em])
        # Test_label[test_num] = em
        test_num = test_num + 1
        tenum = tenum + 1
        
    else:
#        tenum = tenum + 1
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
        
        frames = divmod(time,300)[0]
        frames_yu = divmod(time,300)[1]
        if frames_yu > 100:
            frames += 1
            part = np.pad(part,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta11 = np.pad(delta11,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta21 = np.pad(delta21,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
        pernums_test[tenum] = frames
        tenum = tenum + 1
        for j in range(frames):
            begin = 300*j
            end = begin + 300
            part_seg = part[begin:end,:]
            delta11_seg = delta11[begin:end,:]
            delta21_seg = delta21[begin:end,:]
                
            test_data[test_num,:,:,0] = part_seg
            test_data[test_num,:,:,1] = delta11_seg 
            test_data[test_num,:,:,2] = delta21_seg 
            
            em = generate_label(label)
            Test_label.append(em)
            sumpath.append(path)
            total.append([path,em])
            # Test_label[test_num] = em
            test_num = test_num + 1  
            
print (test_num)
print (len(Test_label))
#     return test_num, Test_label

# def valid():
    
denum = 0
dev_num = 0
Valid_label = []
dev_datalist = []
for speaker in os.listdir(dev_path):
    path = dev_path + speaker
    # print(speaker)
    label = abstract_line(speaker)
    if (label == 'neg') or (label == 'pos'):
        dev_datalist.append([path,label])
# print (dev_datalist)
for i, datalist in enumerate (dev_datalist):
    path = datalist[0]
    label = datalist[1]
    data, time, rate = read_file(path)
    mel_spec = ps.logfbank(data,rate,nfilt = filter_num)
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)
    
    time = mel_spec.shape[0] 
    valid_label.append(em)
    if(time <= 300):
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
        
        part = np.pad(part,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta11 = np.pad(delta11,((0,300 - time),(0,0)),'constant',constant_values = 0)
        delta21 = np.pad(delta21,((0,300 - time),(0,0)),'constant',constant_values = 0)
        valid_data[dev_num,:,:,0] = part 
        valid_data[dev_num,:,:,1] = delta11 
        valid_data[dev_num,:,:,2] = delta21
        
        em = generate_label(label)
        dev_emt[label] = dev_emt[label] + 1
        Valid_label.append(em)
        dev_num = dev_num + 1
        denum = denum + 1
        
    else:
        denum = denum + 1
        part = mel_spec
        delta11 = delta1
        delta21 = delta2
        
        part = (part -mean1)/(std1+eps)
        delta11 = (delta11 - mean2)/(std2+eps)
        delta21 = (delta21 - mean3)/(std3+eps)
                
        frames = divmod(time,300)[0]
        frames_yu = divmod(time,300)[1]
        if frames_yu > 100:
            frames += 1
            part = np.pad(part,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta11 = np.pad(delta11,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
            delta21 = np.pad(delta21,((0,300*frames - time),(0,0)),'constant',constant_values = 0)
        for j in range(frames):
            begin = 300*j
            end = begin + 300
            part_seg = part[begin:end,:]
            delta11_seg = delta11[begin:end,:]
            delta21_seg = delta21[begin:end,:]
    
            valid_data[dev_num,:,:,0] = part_seg 
            valid_data[dev_num,:,:,1] = delta11_seg
            valid_data[dev_num,:,:,2] = delta21_seg
            
            em = generate_label(label)
            dev_emt[label] = dev_emt[label] + 1
            Valid_label.append(em)
            dev_num = dev_num + 1

print (dev_num)
print (len(Valid_label))
    # return dev_num, Valid_label
#     # neg_index = np.arange(negnum)
#     # pos_index = np.arange(posnum)
    
#     # n = 0
#     # p = 0
    
#     # for l in range(train_num):
#     #     if (train_label[l] == 0):
#     #         neg_index[n] = l
#     #         n = n + 1
#     #     else:
#     #         pos_index[p] = l
#     #         p = p + 1
#     # print(n,p)  
pernums_test = np.array(pernums_test)
train_label = np.array(train_label)
Test_label = np.array(Test_label)
Valid_label = np.array(Valid_label)
test_label = np.array(test_label)
valid_label = np.array(valid_label)


print(pernums_test)
output = './Compare2021_s300_40_fu.pkl'
f=open(output,'wb') 
pickle.dump((train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev),f)
f.close()

        
    
# if __name__=='__main__':
#     train_data,train_label = train()
#     test_data, Test_label = test()
#     dev_data, Valid_label = valid()
    
    
    
    
    
    