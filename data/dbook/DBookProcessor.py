# %%
import os
import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
from tqdm import tqdm
random.seed(13)

# %%
input_dir = 'original/'
output_dir = './'
melu_output_dir = '../../../MeLU/dbook/'
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]

if not os.path.exists("{}/meta_training/".format(output_dir)):
    os.mkdir("{}/log/".format(output_dir))
    for state in states:
        os.mkdir("{}/{}/".format(output_dir, state))
        os.mkdir("{}/{}/".format(melu_output_dir, state))
        if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
            os.mkdir("{}/{}/{}".format(output_dir, "log", state))

# %%
ui_data = pd.read_csv(input_dir+'user_book.dat', names=['user','item','rating'], sep='\t',engine='python')
ul = pd.read_csv(input_dir+'user_location.dat', names=['user','location'], sep='\t',engine='python')

ba = pd.read_csv(input_dir+'book_author.dat', names=['book','author'], sep='\t',engine='python')
bp = pd.read_csv(input_dir+'book_publisher.dat', names=['book','publisher'], sep='\t',engine='python')

by = pd.read_csv(input_dir+'book_year.dat', names=['book','year'], sep='\t',engine='python')

# %%
max(ui_data.item)

# %%
user_list = list(set(ui_data.user) & set(ul.user))
item_list = list(set(ui_data.item) & ((set(ba.book) & set(bp.book))) & set(by.book))
len(user_list), len(item_list)

# %%
"""
### 1. user and item featur
"""

# %%
location_list = list(set(ul[ul.user.isin(user_list)].location))
publisher_list = list(set(bp[bp.book.isin(item_list)].publisher))
author_list = list(set(ba[ba.book.isin(item_list)].author))
len(location_list), len(publisher_list), len(author_list)

# %%
from tqdm import tqdm
import torch
user_fea = {}
for i in tqdm(user_list):
    location_idx = location_list.index(list(ul[ul['user']==i].location)[0])
    location = torch.tensor([[location_idx]]).long()
    user_fea[i] = location
len(user_fea)

# %%
item_fea_homo = {}
item_fea_hete = {}
for i in tqdm(item_list):
    publisher_idx = publisher_list.index(list(bp[bp['book']==i].publisher)[0])
    publisher = torch.tensor([[publisher_idx]]).long()
        
    author_idx = author_list.index(list(ba[ba['book']==i].author)[0])
    author = torch.tensor([[author_idx]]).long()
    
    item_fea_hete[i] = publisher
    item_fea_homo[i] = torch.cat((publisher, author), 1)
len(item_fea_hete), len(item_fea_homo)

# %%
"""
### 2. mp data
"""

# %%
states = ["warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]

# %%
import collections
def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)
    return dict(re_d)

# %%
b_authors =  {k: g["author"].tolist() for k,g in ba[ba.book.isin(item_list)].groupby("book")}
a_books = reverse_dict(b_authors)
len(b_authors), len(a_books)

# %%
def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

# %%
state = 'meta_training'

support_u_books = json.load(open(output_dir+state+'/support_u_books.json','r'), object_hook=jsonKeys2int)
query_u_books = json.load(open(output_dir+state+'/query_u_books.json','r'), object_hook=jsonKeys2int)
support_u_books_y = json.load(open(output_dir+state+'/support_u_books_y.json','r'), object_hook=jsonKeys2int)
query_u_books_y = json.load(open(output_dir+state+'/query_u_books_y.json','r'), object_hook=jsonKeys2int)
if support_u_books.keys() == query_u_books.keys():
    u_id_list = support_u_books.keys()
print(len(u_id_list))

train_u_books = {}
if support_u_books.keys() == query_u_books.keys():
    u_id_list = support_u_books.keys()
print(len(u_id_list))
for idx, u_id in tqdm(enumerate(u_id_list)):
    train_u_books[int(u_id)] = []
    train_u_books[int(u_id)] += support_u_books[u_id]+query_u_books[u_id]
len(train_u_books)

# %%
train_u_id_list = list(u_id_list).copy()
len(train_u_id_list)

# %%
# get mp data 
print(state)

u_b_u_books = {}
u_b_a_books= {}

support_b_users = reverse_dict(support_u_books)

for u in tqdm(u_id_list):
    u_b_u_books[u] = {}
    u_b_a_books[u] = {}
    for b in support_u_books[u]:
        u_b_a_books[u][b] = set([b])
        for _a in b_authors[b]:
            cur_bs = a_books[_a]
            u_b_a_books[u][b].update(cur_bs)
        
        u_b_u_books[u][b] = set([b])
        u_b_u_books[u][b].update(support_u_books[u].copy())  # add itself to avoid empty tensor when build the support set
        if b in support_b_users:
            for _u in support_b_users[b]:  #  only include user in training set !!!!
                cur_bs = support_u_books[_u]  # list
                u_b_u_books[u][b].update(cur_bs)
    
    for b in query_u_books[u]:
        if b in u_b_a_books[u] or b in u_b_u_books[u]:
            print('error!!!')
            break
        u_b_a_books[u][b] = set([b])
        for _a in b_authors[b]:
            cur_bs = a_books[_a]
            u_b_a_books[u][b].update(cur_bs)
        
        u_b_u_books[u][b] = set([b])
        u_b_u_books[u][b].update(support_u_books[u].copy())  # add itself to avoid empty tensor when build the support set
        if b in support_b_users:
            for _u in support_b_users[b]:  #  only include user in training set !!!!
                cur_bs = support_u_books[_u]  # list
                u_b_u_books[u][b].update(cur_bs)
        
print(len(u_b_u_books), len(u_b_a_books))

# %%
import pickle
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_ub_app = []
    support_ubub_app = []
    support_ubab_app = []
        
    for b_id in support_u_books[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
        except:
            support_x_app = tmp_x_converted

        # meta-paths
        # UB
        support_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], support_u_books[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UBUB
        support_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_books[u_id][b_id])), dim=0))
        # UBAB
        support_ubab_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_a_books[u_id][b_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_books_y[u_id])

    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ub_app, open("{}/{}/support_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubub_app, open("{}/{}/support_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubab_app, open("{}/{}/support_ubab_{}.pkl".format(output_dir, state, idx), "wb"))
    
    query_x_app = None
    query_ub_app = []
    query_ubub_app = []
    query_ubab_app = []
        
    for b_id in query_u_books[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
        except:
            query_x_app = tmp_x_converted

        # meta-paths
        # UM
        query_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], support_u_books[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        query_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_books[u_id][b_id])), dim=0))
        # UMAM
        query_ubab_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_a_books[u_id][b_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_books_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ub_app, open("{}/{}/query_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubub_app,open("{}/{}/query_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubab_app,open("{}/{}/query_ubab_{}.pkl".format(output_dir, state, idx), "wb"))
    
print(idx)

# %%
# state = 'warm_up'
# state = 'user_cold_testing'
# state = 'item_cold_testing'
state = 'user_and_item_cold_testing'

support_u_books = json.load(open(output_dir+state+'/support_u_books.json','r'), object_hook=jsonKeys2int)
query_u_books = json.load(open(output_dir+state+'/query_u_books.json','r'), object_hook=jsonKeys2int)
support_u_books_y = json.load(open(output_dir+state+'/support_u_books_y.json','r'), object_hook=jsonKeys2int)
query_u_books_y = json.load(open(output_dir+state+'/query_u_books_y.json','r'), object_hook=jsonKeys2int)
if support_u_books.keys() == query_u_books.keys():
    u_id_list = support_u_books.keys()
print(len(u_id_list))

cur_train_u_books =  train_u_books.copy()

if support_u_books.keys() == query_u_books.keys():
    u_id_list = support_u_books.keys()
print(len(u_id_list))
for idx, u_id in tqdm(enumerate(u_id_list)):
    if u_id not in cur_train_u_books:
        cur_train_u_books[u_id] = []
    cur_train_u_books[u_id] += support_u_books[u_id]

print(len(cur_train_u_books),  len(train_u_books))
print(len(set(train_u_id_list) & set(u_id_list)))

(len(u_id_list) +  len(train_u_books) - len(set(train_u_id_list) & set(u_id_list))) == len(set(cur_train_u_books))

# %%
# get mp data 
print(state)

u_b_u_books = {}
u_b_a_books= {}

cur_train_b_users = reverse_dict(cur_train_u_books)

for u in tqdm(u_id_list):
    u_b_u_books[u] = {}
    u_b_a_books[u] = {}
    for b in support_u_books[u]:
        u_b_a_books[u][b] = set([b])
        for _a in b_authors[b]:
            cur_bs = a_books[_a]
            u_b_a_books[u][b].update(cur_bs)
        
        u_b_u_books[u][b] = set([b])
        u_b_u_books[u][b].update(cur_train_u_books[u].copy())  # add itself to avoid empty tensor when build the support set
        if b in support_b_users:
            for _u in cur_train_b_users[b]:  #  only include user in training set !!!!
                cur_bs = cur_train_u_books[_u]  # list
                u_b_u_books[u][b].update(cur_bs)
    
    for b in query_u_books[u]:
        if b in u_b_a_books[u] or b in u_b_u_books[u]:
            print('error!!!')
            break
        u_b_a_books[u][b] = set([b])
        for _a in b_authors[b]:
            cur_bs = a_books[_a]
            u_b_a_books[u][b].update(cur_bs)
        
        u_b_u_books[u][b] = set([b])
        u_b_u_books[u][b].update(cur_train_u_books[u].copy())  # add itself to avoid empty tensor when build the support set
        if b in support_b_users:
            for _u in cur_train_b_users[b]:  #  only include user in training set !!!!
                cur_bs = cur_train_u_books[_u]  # list
                u_b_u_books[u][b].update(cur_bs)
        
print(len(u_b_u_books), len(u_b_a_books))
print(len(cur_train_u_books), len(train_u_books))

# %%
import pickle
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_ub_app = []
    support_ubub_app = []
    support_ubab_app = []
        
    for b_id in support_u_books[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
        except:
            support_x_app = tmp_x_converted

        # meta-paths
        # UB
        support_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], cur_train_u_books[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UBUB
        support_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_books[u_id][b_id])), dim=0))
        # UBAB
        support_ubab_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_a_books[u_id][b_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_books_y[u_id])

    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ub_app, open("{}/{}/support_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubub_app, open("{}/{}/support_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubab_app, open("{}/{}/support_ubab_{}.pkl".format(output_dir, state, idx), "wb"))
    
    query_x_app = None
    query_ub_app = []
    query_ubub_app = []
    query_ubab_app = []
        
    for b_id in query_u_books[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
        except:
            query_x_app = tmp_x_converted

        # meta-paths
        # UM
        query_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], cur_train_u_books[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        query_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_books[u_id][b_id])), dim=0))
        # UMAM
        query_ubab_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_a_books[u_id][b_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_books_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ub_app, open("{}/{}/query_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubub_app,open("{}/{}/query_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubab_app,open("{}/{}/query_ubab_{}.pkl".format(output_dir, state, idx), "wb"))
    
print(idx)

# %%
