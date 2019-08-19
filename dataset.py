import os
# import pickle
import sys
import numpy as np
from time import time

import time as time0
from datetime import datetime
from tqdm import tqdm

class DataSet:

    def __init__(self, data_path="/home/zhengyi_ma/kaggle/ieee-fraud-detection/data/",
                    batch_size = 32, num_epoch = 10 ):
        self.data_path = data_path 
        self.batch_size = batch_size
        self.num_epoch = num_epoch
    
    def init_dataset(self):

        self.X_train = []
        self.X_train_identity = []
        self.Y_train = []

        self.X_valid = []
        self.X_valid_identity = []
        self.Y_valid = []

    def prepare_dataset(self):
        if not hasattr(self, 'X_train'):
            self.init_dataset()
        
        iden_path = os.path.join(self.data_path, "train_identity.csv")
        f_iden = open(iden_path)
        self.identity_dic = {}
        print("identity loading...")
        for line in tqdm(f_iden):

            line_cols = line.split(",")
            tid = line_cols[0]
            self.identity_len = len(line_cols) - 1
            for i in range(len(line_cols)):
                if line_cols[i] == "":
                    line_cols[i] = 0

                try:
                    fl = float(line_cols[i])
                    line_cols[i] = fl
                except:
                    line_cols[i] = 0

            self.identity_dic[tid] = np.array(line_cols[1:])
            
        print("transcation loading...")
        trans_path = os.path.join(self.data_path, "train_transaction.csv")
        f_trans = open(trans_path)
        for line in tqdm(f_trans):
            line_cols = line.split(",")
            tid = line_cols[0]
            self.trans_len = len(line_cols) - 2
            for i in range(len(line_cols)):

                if line_cols[i] == "":
                    line_cols[i] = 0
                try:
                    fl = float(line_cols[i])
                    line_cols[i] = fl
                except:
                    line_cols[i] = 0
            
            self.X_train.append(np.array(line_cols[2:]))
            self.Y_train.append(line_cols[1])

            if tid in self.identity_dic:
                self.X_train_identity.append(self.identity_dic[tid])
            else:
                self.X_train_identity.append(np.zeros((self.identity_len)))

        # print(self.X_train)
        # print(self.X_train_identity)

        self.divide_dataset()
    
        print("the dim of X_train is ",np.array(self.X_train).shape)
        print("the dim of X_train_iden is ",np.array(self.X_train_identity).shape)
        print("the dim of Y_train is ",np.array(self.Y_train).shape)

    def divide_dataset(self):
        shuf_a = np.arange(len(self.X_train))
        np.random.shuffle(shuf_a)

        num_valid = int(len(self.X_train)/ 5)
        num_sum = len(self.X_train)
        num_train = num_sum - num_valid

        for i in range(0,num_valid):
            idx = shuf_a[i]
            self.X_valid.append(self.X_train[idx])
            self.X_valid_identity.append(self.X_train_identity[idx])
            self.Y_valid.append(self.Y_train[idx])

        X_train_new = []
        X_train_identity_new = []
        Y_train_new = []

        for i in range(0,num_train):
            idx = shuf_a[i + num_valid]
            X_train_new.append(self.X_train[idx])
            X_train_identity_new.append(self.X_train_identity[idx])
            Y_train_new.append(self.Y_train[idx])
        
        self.X_train = X_train_new
        self.X_train_identity = X_train_identity_new
        self.Y_train = Y_train_new


    def gen_epochs(self):
        pair_train = len(self.X_train)
        for i in range(self.num_epoch):
            perm = np.random.permutation(pair_train) #generate an array range from 1 to pair_train then random
            yield self.gen_batchs(perm)
    
    def gen_batchs(self, perm):
        pair_train = len(self.X_train)
        pair_valid = len(self.X_valid)
        perm2 = [int(item%pair_valid) for item in perm]
        for j in range(int(pair_train/self.batch_size)): #num of example in one batch
            train_start = j*self.batch_size
            train_end = (j+1)*self.batch_size

            X_t = np.zeros((self.batch_size,self.trans_len),dtype=np.float64)
            X_ti = np.zeros((self.batch_size,self.identity_len),dtype=np.float64)
            X_v = np.zeros((self.batch_size,self.trans_len),dtype=np.float64)
            X_vi = np.zeros((self.batch_size,self.identity_len),dtype=np.float64)

            for i in range(self.batch_size):
                X_t[i] =  self.X_train[perm[train_start+i]]
                X_ti[i] = self.X_train_identity[perm[train_start + i]]
            
            for i in range(self.batch_size):
                X_v[i] =  self.X_valid[perm2[train_start+i]]
                X_vi[i] = self.X_valid_identity[perm2[train_start + i]]

            yield [[self.X_train[item] for item in perm[train_start:train_end]], 
                    [self.X_train_identity[item] for item in perm[train_start:train_end]],
                    [self.Y_train[item] for item in perm[train_start:train_end]],
                    [self.X_valid[item] for item in perm2[train_start:train_end]], 
                    [self.X_valid_identity[item] for item in perm2[train_start:train_end]], 
                    [self.Y_valid[item] for item in perm2[train_start:train_end]]
                    ]



    


            




            
                




        

        


        
