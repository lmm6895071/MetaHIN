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
from collections import defaultdict

# %%
input_dir = 'original/'
output_dir = './'
melu_output_dir = '../MeLU/movielens/'
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]

if not os.path.exists("{}/meta_training/".format(output_dir)):
    if not os.path.exists("{}/log/".format(output_dir)):
        os.mkdir("{}/log/".format(output_dir))
    for state in states:
        _path = "{}/{}/".format(output_dir, state)
        if not os.path.exists(_path):
            os.mkdir(_path)
        _path = "{}/{}/".format(melu_output_dir, state)
        if not os.path.exists(_path):
            os.mkdir(_path)
        if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
            os.mkdir("{}/{}/{}".format(output_dir, "log", state))
            
print('====================')


# %%
ui_data = pd.read_csv(input_dir+'rating.dat', names=['user', 'item', 'rating', 'timestamp'],sep="\t", engine='python')
len(ui_data)  # 1302630

# %%
user_list = set(ui_data.user)
item_list = set(ui_data.item)
len(user_list), len(item_list)

# %%
new_user = pd.read_csv(input_dir+'new_user.dat', names=['user'], engine='python').user.tolist()
new_item = pd.read_csv(input_dir+'new_item.dat', names=['item'], engine='python').item.tolist()
len(new_user), len(new_item)

# %%
exist_user = list(user_list - set(new_user))
exist_item = list(item_list- set(new_item))
len(exist_user), len(exist_item)

# %%
"""
# meta-training and meta-testing data
"""

# %%
meta_training_data = ui_data[(ui_data['user'].isin(exist_user)) & (ui_data['item'].isin(exist_item))]
len(meta_training_data)  # 967291 

# %%
warm_data = meta_training_data.sample(int(0.1*len(meta_training_data)))
len(warm_data)  # 96729 

# %%
# warm data
warm_x = {k: g["item"].tolist() for k,g in warm_data.groupby("user")}
warm_y = {k: g["rating"].tolist() for k,g in warm_data.groupby("user")}
json.dump(warm_x, open("{}/warm_up.json".format(output_dir), 'w'))
json.dump(warm_y, open("{}/warm_up_y.json".format(output_dir), 'w'))
len(warm_x), len(warm_y)

# %%
count_user_interaction = dict(warm_data.user.value_counts())
less_100_user = [k for k, v in count_user_interaction.items() if 13<=v<=100]
len(less_100_user)

# %%
training_data = meta_training_data.loc[meta_training_data.index.difference(warm_data.index)]
len(training_data)  # 870562 

# %%
# training data
training_x = {k: g["item"].tolist() for k,g in training_data.groupby("user")}
training_y = {k: g["rating"].tolist() for k,g in training_data.groupby("user")}
json.dump(training_x, open("{}/meta_training.json".format(output_dir), 'w'))
json.dump(training_y, open("{}/meta_training_y.json".format(output_dir), 'w'))
len(training_x), len(training_y)

# %%
count_user_interaction = dict(training_data.user.value_counts())
less_100_user = [k for k, v in count_user_interaction.items() if 13<=v<=100]
len(less_100_user)

# %%
# meta-testing
# user_cold_testing
user_cold_data =ui_data[(ui_data['user'].isin(new_user)) & (ui_data['item'].isin(exist_item))]
len(user_cold_data)  # 164136 

# %%
user_cold_x = {k: g["item"].tolist() for k,g in user_cold_data.groupby("user")}
user_cold_y = {k: g["rating"].tolist() for k,g in user_cold_data.groupby("user")}
json.dump(user_cold_x, open("{}/user_cold_testing.json".format(output_dir), 'w'))
json.dump(user_cold_y, open("{}/user_cold_testing_y.json".format(output_dir), 'w'))
len(user_cold_x), len(user_cold_y)

# %%
count_user_interaction = dict(user_cold_data.user.value_counts())
less_100_user = [k for k, v in count_user_interaction.items() if 13<=v<=100]
len(less_100_user)

# %%
# item_cold_testing
item_cold_data =ui_data[(ui_data['user'].isin(exist_user)) & (ui_data['item'].isin(new_item))]
len(item_cold_data)  # 118467 

# %%
item_cold_x = {k: g["item"].tolist() for k,g in item_cold_data.groupby("user")}
item_cold_y = {k: g["rating"].tolist() for k,g in item_cold_data.groupby("user")}
json.dump(item_cold_x, open("{}/item_cold_testing.json".format(output_dir), 'w'))
json.dump(item_cold_y, open("{}/item_cold_testing_y.json".format(output_dir), 'w'))
len(item_cold_x), len(item_cold_y)

# %%
count_user_interaction = dict(item_cold_data.user.value_counts())
less_100_user = [k for k, v in count_user_interaction.items() if 13<=v<=100]
len(less_100_user)

# %%
# user_and_item_cold_testing
user_item_cold_data =ui_data[(ui_data['user'].isin(new_user)) & (ui_data['item'].isin(new_item))]
len(user_item_cold_data)  # 52736 

# %%
user_item_cold_x = {k: g["item"].tolist() for k,g in user_item_cold_data.groupby("user")}
user_item_cold_y = {k: g["rating"].tolist() for k,g in user_item_cold_data.groupby("user")}
json.dump(user_item_cold_x, open("{}/user_and_item_cold_testing.json".format(output_dir), 'w'))
json.dump(user_item_cold_y, open("{}/user_and_item_cold_testing_y.json".format(output_dir), 'w'))
len(user_item_cold_x), len(user_item_cold_y)

# %%
count_user_interaction = dict(user_item_cold_data.user.value_counts())
less_10_user = [k for k, v in count_user_interaction.items() if 13<=v<=100]
len(less_10_user)

# %%
len(training_data)+len(warm_data)+len(user_cold_data)+len(item_cold_data)+len(user_item_cold_data)

# %%
"""
# support set and query set
"""

# %%
"""
### 1. user and item feature
"""

# %%
user_fans = pd.read_csv(input_dir+'user_fans.dat', names=['user','fans'], sep='\t', engine='python')
# user_fans
len(user_fans)

# %%
user_avgrating = pd.read_csv(input_dir+'user_avgrating.dat', names=['user','avgrating'], sep='\t', engine='python')
len(user_avgrating)

# %%
user_friends = pd.read_csv(input_dir+'user_friends.dat', names=['user','friends'], sep='\t', engine='python')
len(user_friends)

# %%
item_stars =  pd.read_csv(input_dir+'item_stars.dat', names=['item','stars'], sep='\t', engine='python')
len(item_stars)

# %%
item_postalcode =  pd.read_csv(input_dir+'item_postalcode.dat', names=['item','postalcode'], sep='\t', engine='python')
len(item_postalcode)

# %%
item_city =  pd.read_csv(input_dir+'item_city.dat', names=['item','city'], sep='\t', engine='python')
len(item_city)

# %%
item_city.city.value_counts()

# %%
item_category =  pd.read_csv(input_dir+'item_category.dat', names=['item','category'], sep='\t', engine='python')
len(item_category)

import collections
from collections import defaultdict
def reverse_dict(d):
    # {1:[a,b,c], 2:[a,f,g],...}
    re_d = collections.defaultdict(list)
    for k, v_list in d.items():
        for v in v_list:
            re_d[v].append(k)
    return dict(re_d)
    
# %%
b_types = defaultdict(set)
for index, row in item_category.iterrows():
    types = list(map(int, row['category'].strip().split(' ')))
    b_types[row['item']].update(types)
b_types = dict(b_types)
t_businesses = reverse_dict(b_types)

# %%
sorted([len(v) for k, v in t_businesses.items()], reverse=True)

# %%
item_state =  pd.read_csv(input_dir+'item_state.dat', names=['item','state'], sep='\t', engine='python')
len(item_state)

# %%
item_reviewcount =  pd.read_csv(input_dir+'item_reviewcount.dat', names=['item','reviewcount'], sep='\t', engine='python')
len(item_reviewcount)

# %%
item_neighbor =  pd.read_csv(input_dir+'item_neighbor.dat', names=['item','neighbor'], sep='\t', engine='python')
len(item_neighbor)

# %%
list(item_category[item_category['item']==1].category)[0].strip().split(' ')

# %%
len(set(item_category.category))

# %%
user_fea = {}
for i in tqdm(user_list):
    fans_idx = list(user_fans[user_fans['user']==i].fans)[0]
    fans = torch.tensor([[fans_idx]]).long()
    avgrating_idx = list(user_avgrating[user_avgrating['user']==i].avgrating)[0]
    avgrating = torch.tensor([[avgrating_idx]]).long()
    
    user_fea[i] = torch.cat((fans, avgrating),1)
len(user_fea)

# %%
np.save(output_dir+'user_feature.npy',user_fea)

# %%
item_fea_homo = {}
item_fea_hete = {}
for i in tqdm(item_list):
    stars_idx = list(item_stars[item_stars['item']==i].stars)[0]
    stars = torch.tensor([[stars_idx]]).long()
    postalcode_idx = list(item_postalcode[item_postalcode['item']==i].postalcode)[0]
    postalcode = torch.tensor([[postalcode_idx]]).long()
    reviewcount_idx = list(item_reviewcount[item_reviewcount['item']==i].reviewcount)[0]
    reviewcount = torch.tensor([[reviewcount_idx]]).long()
    
    city_idx = list(item_city[item_city['item']==i].city)[0]
    city = torch.tensor([[city_idx]]).long()
    state_idx = list(item_state[item_state['item']==i].state)[0]
    state = torch.tensor([[state_idx]]).long()
    
#     category = torch.zeros(1, 542).long()
#     categories = list(item_category[item_category['item']==i].category)[0].strip().split(' ')
#     for c in categories:
#         category[0, int(c)] = 1
#     item_fea_hete[i] = torch.cat((stars, postalcode,reviewcount, category),1)
#     item_fea_homo[i] = torch.cat((stars, postalcode, reviewcount, city, state, category), 1)
    
    item_fea_hete[i] = torch.cat((stars, postalcode,reviewcount),1)
    item_fea_homo[i] = torch.cat((stars, postalcode, reviewcount, city, state), 1)
len(item_fea_hete), len(item_fea_homo)

# %%
np.save(output_dir+'item_feature_hete.npy',item_fea_hete)
np.save(output_dir+'item_feature_homo.npy',item_fea_homo)

# %%
"""
### 2. mp data
"""

# %%
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]

# %%


# %%
tqdm._instances.clear()
del tqdm
from tqdm import tqdm

# %%
b_cities = {k: g["city"].tolist() for k,g in item_city.groupby("item")}
c_businesses = reverse_dict(b_cities)
b_states = {k: g["city"].tolist() for k,g in item_city.groupby("item")}
s_businesses = reverse_dict(b_states)

b_types = defaultdict(set)
for index, row in item_category.iterrows():
    types = list(map(int, row['category'].strip().split(' ')))
    b_types[row['item']].update(types)
b_types = dict(b_types)
t_businesses = reverse_dict(b_types)

# %%
sum([len(v) for k, v in b_cities.items()]), sum([len(v) for k, v in b_states.items()])

# %%
# get UM in support set and query set 
state = "meta_training"
print(state)
u_businesses = training_x
u_businesses_y = training_y

support_u_businesses = {}
support_u_businesses_y = {}
query_u_businesses = {}
query_u_businesses_y = {}
train_u_businesses = {}
train_u_businesses_y = {}

for u_id in tqdm(u_businesses):  # each task contains support set and query set
    seen_movie_len = len(u_businesses[u_id])
    indices = list(range(seen_movie_len))
    if seen_movie_len < 13 or seen_movie_len > 100:
        continue
    
    support_u_businesses[u_id] = []
    support_u_businesses_y[u_id] = []
    query_u_businesses[u_id] = []
    query_u_businesses_y[u_id] = []
    
    train_u_businesses[u_id]  = []
    train_u_businesses_y[u_id] = []
    
    random.shuffle(indices)
    tmp_movies = np.array(u_businesses[u_id])
    tmp_y = np.array(u_businesses_y[u_id])
    
    support_u_businesses[u_id] += list(map(int, tmp_movies[indices[:-10]]))
    support_u_businesses_y[u_id] += list(map(int, tmp_y[indices[:-10]]))
    query_u_businesses[u_id] += list(map(int, tmp_movies[indices[-10:]]))
    query_u_businesses_y[u_id] += list(map(int, tmp_y[indices[-10:]]))
    
    train_u_businesses[u_id] += u_businesses[u_id]
    train_u_businesses_y[u_id] += u_businesses_y[u_id]
    

json.dump(support_u_businesses, open(output_dir+state+'/support_u_businesses.json','w'))
json.dump(support_u_businesses_y, open(output_dir+state+'/support_u_businesses_y.json','w'))
json.dump(query_u_businesses, open(output_dir+state+'/query_u_businesses.json','w'))
json.dump(query_u_businesses_y, open(output_dir+state+'/query_u_businesses_y.json','w'))
len(support_u_businesses), len(support_u_businesses_y), len(query_u_businesses), len(query_u_businesses_y), len(train_u_businesses), len(train_u_businesses_y)

# %%
# get mp data 
print(state)

# u_b_u_businesses = {}
u_b_c_businesses = {}
u_b_s_businesses = {}
# u_b_t_businesses = {}

support_b_users = reverse_dict(support_u_businesses)

for u, bs in tqdm(train_u_businesses.items()):
#     u_b_u_businesses[u] = []
    u_b_c_businesses[u] = []
    u_b_s_businesses[u] = []
#     u_b_t_businesses[u] = []
    for b in bs:    
#         cur_bs = set([b])
#         if b in support_b_users:  # for meta_training, only support set can be seen!!!
#             for _u in support_b_users[b]:  #  only include user in training set !!!!
#                 cur_bs.update(support_u_businesses[_u])  # list        
#         u_b_u_businesses[u].append(list(cur_bs))
        
        cur_bs = set()
        for _c in b_cities[b]:
            cur_bs.update(c_businesses[_c])
        u_b_c_businesses[u].append(list(cur_bs))
        
        cur_bs = set()
        for _s in b_states[b]:
            cur_bs.update(s_businesses[_s])
        u_b_s_businesses[u].append(list(cur_bs))
        
#         cur_bs = set()
#         for _t in b_types[b]:
#             cur_bs.update(t_businesses[_t])
#         u_b_t_businesses[u].append(list(cur_bs))

# print(len(u_b_u_businesses))
print(len(u_b_c_businesses))
print(len(u_b_s_businesses))
# print(len(u_b_t_businesses))

# json.dump(u_b_u_businesses, open(output_dir+state+'/u_b_u_businesses.json','w'))
json.dump(u_b_c_businesses, open(output_dir+state+'/u_b_c_businesses.json','w')) 
json.dump(u_b_s_businesses, open(output_dir+state+'/u_b_s_businesses.json','w')) 
# json.dump(u_b_t_businesses, open(output_dir+state+'/u_b_t_businesses.json','w'))
print('write done!')

# %%
len(train_u_businesses[4]) == len(u_b_u_businesses[4]), len(train_u_businesses[4]) == len(u_b_c_businesses[4]), len(train_u_businesses[4]) == len(u_b_s_businesses[4])


# %%
tqdm._instances.clear()
del tqdm
from tqdm import tqdm 

# %%
if support_u_businesses.keys() == query_u_businesses.keys():
    u_id_list = support_u_businesses.keys()
print(len(u_id_list))
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_x_app_melu = None
    support_ub_app = []
    support_ubub_app = []
    support_ubcb_app = []
    support_ubsb_app = []
        
    for index1, b_id in enumerate(support_u_businesses[u_id]):
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            support_x_app_melu = torch.cat((support_x_app_melu, tmp_x_converted_melu), 0)
        except:
            support_x_app = tmp_x_converted
            support_x_app_melu = tmp_x_converted_melu

        # meta-paths
        # UB
        support_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], support_u_businesses[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UBUB
        support_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][index1])), dim=0))
        # UBCB
        support_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][index1])), dim=0))
        # UBSB
        support_ubsb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_s_businesses[u_id][index1])), dim=0))
#         # UBTB
#         support_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][b_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_businesses_y[u_id])
    
    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ub_app, open("{}/{}/support_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubub_app, open("{}/{}/support_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubcb_app, open("{}/{}/support_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubsb_app, open("{}/{}/support_ubsb_{}.pkl".format(output_dir, state, idx), "wb"))
    # data for MeLU
    pickle.dump(support_x_app_melu, open("{}/{}/support_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    
    query_x_app = None
    query_x_app_melu = None
    query_ub_app = []
    query_ubub_app = []
    query_ubcb_app = []
    query_ubsb_app = []
        
    for index2, b_id in enumerate(query_u_businesses[u_id]):
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            query_x_app_melu = torch.cat((query_x_app_melu, tmp_x_converted_melu), 0)
        except:
            query_x_app = tmp_x_converted
            query_x_app_melu = tmp_x_converted_melu

        # meta-paths
        # UB
        query_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], support_u_businesses[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UBUB
        query_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][index2])), dim=0))
        # UBCB
        query_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][index2])), dim=0))
        # UBSB
        query_ubsb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_s_businesses[u_id][index2])), dim=0))
#         # UBTB
#         query_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][m_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_businesses_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ub_app, open("{}/{}/query_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubub_app,open("{}/{}/query_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubcb_app,open("{}/{}/query_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubsb_app,open("{}/{}/query_ubsb_{}.pkl".format(output_dir, state, idx), "wb"))
    # data for MeLU
    pickle.dump(query_x_app_melu, open("{}/{}/query_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))

    with open("{}/log/{}/supp_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
        for i, b_id in enumerate(support_u_businesses[u_id]):
            f.write("{}\t{}\t{}\n".format(u_id, b_id, support_u_businesses_y[u_id][i]))
                
    with open("{}/log/{}/query_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
        for i, b_id in enumerate(query_u_businesses[u_id]):
            f.write("{}\t{}\t{}\n".format(u_id, b_id,  query_u_businesses_y[u_id][i]))
            
print(idx)


# %%
# get UM in support set and query set 
# state = "warm_up"
# print(state)
# u_businesses = warm_x
# u_businesses_y = warm_y

# state = "user_cold_testing"
# print(state)
# u_businesses = user_cold_x
# u_businesses_y = user_cold_y

# state = "item_cold_testing"
# print(state)
# u_businesses = item_cold_x
# u_businesses_y = item_cold_y

state = "user_and_item_cold_testing"
print(state)
u_businesses = user_item_cold_x
u_businesses_y = user_item_cold_y

support_u_businesses = {}
support_u_businesses_y = {}
query_u_businesses = {}
query_u_businesses_y = {}

cur_training_u_businesses = train_u_businesses

for u_id in tqdm(u_businesses):  # each task contains support set and query set
    seen_movie_len = len(u_businesses[u_id])
    indices = list(range(seen_movie_len))
    if seen_movie_len < 13 or seen_movie_len > 100:
        continue
    
    support_u_businesses[u_id] = []
    support_u_businesses_y[u_id] = []
    query_u_businesses[u_id] = []
    query_u_businesses_y[u_id] = []
    
    random.shuffle(indices)
    tmp_movies = np.array(u_businesses[u_id])
    tmp_y = np.array(u_businesses_y[u_id])
    
    support_u_businesses[u_id] += list(map(int, tmp_movies[indices[:-10]]))
    support_u_businesses_y[u_id] += list(map(int, tmp_y[indices[:-10]]))
    query_u_businesses[u_id] += list(map(int, tmp_movies[indices[-10:]]))
    query_u_businesses_y[u_id] += list(map(int, tmp_y[indices[-10:]]))
    
    if u_id in cur_training_u_businesses:
        cur_training_u_businesses[u_id] += support_u_businesses[u_id]  # based on meat-traing, add the current support set data
    else:
        cur_training_u_businesses[u_id] = support_u_businesses[u_id]
    
json.dump(support_u_businesses, open(output_dir+state+'/support_u_businesses.json','w'))
json.dump(support_u_businesses_y, open(output_dir+state+'/support_u_businesses_y.json','w'))
json.dump(query_u_businesses, open(output_dir+state+'/query_u_businesses.json','w'))
json.dump(query_u_businesses_y, open(output_dir+state+'/query_u_businesses_y.json','w'))

len(support_u_businesses), len(support_u_businesses_y), len(query_u_businesses), len(query_u_businesses_y), len(cur_training_u_businesses)


# %%
# get mp data 
print(state)

u_m_u_movies = {}
u_m_a_movies = {}
u_m_d_movies = {}

cur_training_m_users = reverse_dict(cur_training_u_movies)

if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))

for u in tqdm(u_id_list):
    u_m_u_movies[u] = {}
    u_m_a_movies[u] = {}
    u_m_d_movies[u] = {}
    for m in support_u_movies[u]:
        u_m_u_movies[u][m] = [m]   # add itself to avoid empty tensor when build the support set
        u_m_a_movies[u][m] = []   
        u_m_d_movies[u][m] = []  
        
        if m in cur_training_m_users:  # include users in meta-training  and users  in current support set
            for _u in cur_training_m_users[m]:  
                cur_ms = cur_training_u_movies[_u]  # list
                u_m_u_movies[u][m].extend(cur_ms)
        u_m_u_movies[u][m] = list(set(u_m_u_movies[u][m]))
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].extend(cur_ms)
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].extend(cur_ms)
    
    for m in query_u_movies[u]:
        u_m_u_movies[u][m] = [m]   # add itself to avoid empty tensor when build the support set
        u_m_a_movies[u][m] = []   
        u_m_d_movies[u][m] = []  
        
        if m in cur_training_m_users:  # include users in meta-training  and users  in current support set
            for _u in cur_training_m_users[m]:  
                cur_ms = cur_training_u_movies[_u]  # list
                u_m_u_movies[u][m].extend(cur_ms)
        u_m_u_movies[u][m] = list(set(u_m_u_movies[u][m]))       
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].extend(cur_ms)
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].extend(cur_ms)
             
print(len(u_m_u_movies), len(u_m_a_movies), len(u_m_d_movies))

json.dump(u_m_u_movies, open(output_dir+state+'/u_m_u_movies.json','w'))
json.dump(u_m_a_movies, open(output_dir+state+'/u_m_a_movies.json','w'))
json.dump(u_m_d_movies, open(output_dir+state+'/u_m_d_movies.json','w')) 
print('write done!')

# %%
if support_u_businesses.keys() == query_u_businesses.keys():
    u_id_list = support_u_businesses.keys()
print(len(u_id_list))
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_x_app_melu = None
    support_ub_app = []
    support_ubub_app = []
    support_ubcb_app = []
    support_ubtb_app = []
        
    for b_id in support_u_businesses[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            support_x_app_melu = torch.cat((support_x_app_melu, tmp_x_converted_melu), 0)
        except:
            support_x_app = tmp_x_converted
            support_x_app_melu = tmp_x_converted_melu

        # meta-paths
        # UM
        support_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], cur_training_u_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        support_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][b_id])), dim=0))
        # UMAM
        support_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][b_id])), dim=0))
        # UMDM
        support_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][b_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_businesses_y[u_id])
    
    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ub_app, open("{}/{}/support_ub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubub_app, open("{}/{}/support_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubcb_app, open("{}/{}/support_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_ubtb_app, open("{}/{}/support_ubtb_{}.pkl".format(output_dir, state, idx), "wb"))
    # data for MeLU
    pickle.dump(support_x_app_melu, open("{}/{}/support_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    
    query_x_app = None
    query_x_app_melu = None
    query_ub_app = []
    query_ubub_app = []
    query_ubcb_app = []
    query_ubtb_app = []
        
    for b_id in query_u_businesses[u_id]:
        tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
        tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            query_x_app_melu = torch.cat((query_x_app_melu, tmp_x_converted_melu), 0)
        except:
            query_x_app = tmp_x_converted
            query_x_app_melu = tmp_x_converted_melu

        # meta-paths
        # UM
        query_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], cur_training_u_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        query_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][m_id])), dim=0))
        # UMAM
        query_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][m_id])), dim=0))
        # UMDM
        query_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][m_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_businesses_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ub_app, open("{}/{}/query_um_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubub_app,open("{}/{}/query_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubcb_app,open("{}/{}/query_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_ubtb_app,open("{}/{}/query_ubtb_{}.pkl".format(output_dir, state, idx), "wb"))
    # data for MeLU
    pickle.dump(query_x_app_melu, open("{}/{}/query_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))

    with open("{}/log/{}/supp_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
        for i, b_id in enumerate(support_u_businesses[u_id]):
            f.write("{}\t{}\t{}\n".format(u_id, b_id, support_u_businesses_y[u_id][i]))
                
    with open("{}/log/{}/query_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
        for i, b_id in enumerate(query_u_businesses[u_id]):
            f.write("{}\t{}\t{}\n".format(u_id, b_id,  query_u_businesses_y[u_id][i]))
            
print(idx)


# %%


# %%


# %%


# %%


# %%
# # state = "user_and_item_cold_testing"
# # u_businesses = user_item_cold_x
# # u_businesses_y = user_item_cold_y

# # state = "user_cold_testing"
# # u_businesses = user_cold_x
# # u_businesses_y = user_cold_y

# # state = "item_cold_testing"
# # u_businesses = item_cold_x
# # u_businesses_y = item_cold_y

# # state = "warm_up"
# # u_businesses = warm_up_x
# # u_businesses_y = warm_up_y

# state = 'meta_training'
# u_businesses = training_x
# u_businesses_y = training_y

# u_b_u_businesses = {}
# u_b_c_businesses = {}
# u_b_t_businesses = {}

# for u, bs in tqdm(u_businesses.items()):
#     u_b_u_businesses[u] = []
#     u_b_c_businesses[u] = []
#     u_b_t_businesses[u] = []
#     for b in bs:
#         if b in train_b_users:
#             for _u in train_b_users[b]:  #  include user in training set !!!!
#                 u_b_u_businesses[u].append(training_x[_u])
#         else:
#             u_b_u_businesses[u].append([b])  # add itself to avoid empty tensor when build the support set
#         for _c in b_cities[b]:
#             u_b_c_businesses[u].append(c_businesses[_c])
#         for _t in b_types[b]:
#             u_b_t_businesses[u].append(t_businesses[_t])
        
# print(len(u_b_u_businesses), len(u_b_c_businesses), len(u_b_t_businesses))
    
# np.save(output_dir+state+'/u_b_u_businesses.npy',u_b_u_businesses)
# np.save(output_dir+state+'/u_b_c_businesses.npy',u_b_c_businesses)
# np.save(output_dir+state+'/u_b_t_businesses.npy',u_b_t_businesses)
# # json.dump(u_b_u_businesses, open(output_dir+state+'/u_b_u_businesses.json', 'w'))
# # json.dump(u_b_c_businesses, open(output_dir+state+'/u_b_c_businesses.json', 'w'))
# # json.dump(u_b_t_businesses, open(output_dir+state+'/u_b_t_businesses.json', 'w'))

# %%
#     if not os.path.exists("{}/log/".format(output_dir)):
#         os.mkdir("{}/log/".format(output_dir))
#     if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
#         os.mkdir("{}/{}/{}".format(output_dir, "log", state))
    
#     print(state)
#     print(len(u_businesses), len(u_b_u_businesses), len(u_b_c_businesses), len(u_b_t_businesses))
#     idx = 0
#     for _, u_id in tqdm(enumerate(u_businesses.keys())):  # each task contains support set and query set
#         seen_business_len = len(u_businesses[u_id])
#         indices = list(range(seen_business_len))
        
#         if seen_business_len < 13 or seen_business_len > 100:
#             continue
            
#         random.shuffle(indices)
#         tmp_businesses = np.array(u_businesses[u_id])
#         tmp_y = np.array(u_businesses_y[u_id])

#         support_x_app = None
#         support_x_app_melu = None
#         support_ub_app = []
#         support_ubub_app = []
#         support_ubcb_app = []
#         support_ubtb_app = []
#         for index1, b_id in enumerate(tmp_businesses[indices[:-10]]):
#             u_id = int(u_id)
#             b_id = int(b_id)
#             tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
#             tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
#             try:
#                 support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
#                 support_x_app_melu = torch.cat((support_x_app_melu, tmp_x_converted_melu), 0)
#             except:
#                 support_x_app = tmp_x_converted
#                 support_x_app_melu = tmp_x_converted_melu

#             # meta-paths
#             # UM
#             support_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_businesses[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
#             # UMUM
#             support_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][index1])), dim=0))
#             # UMAM
#             support_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][index1])), dim=0))
#             # UMDM
#             support_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][index1])), dim=0))
        
#         support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])

#         query_x_app = None
#         query_x_app_melu = None
#         query_ub_app = []
#         query_ubub_app = []
#         query_ubcb_app = []
#         query_ubtb_app = []
#         for index2, b_id in enumerate(tmp_businesses[indices[-10:]]):
#             u_id = int(u_id)
#             b_id = int(b_id)
#             tmp_x_converted = torch.cat((item_fea_hete[b_id], user_fea[u_id]), 1)
#             tmp_x_converted_melu = torch.cat((item_fea_homo[b_id], user_fea[u_id]), 1)
#             try:
#                 query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
#                 query_x_app_melu = torch.cat((query_x_app_melu, tmp_x_converted_melu), 0)
#             except:
#                 query_x_app = tmp_x_converted
#                 query_x_app_melu = tmp_x_converted_melu

#             # meta-paths
#             # UM
#             query_ub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_businesses[u_id])), dim=0))
#             # UMUM
#             query_ubub_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_u_businesses[u_id][index2])), dim=0))
#             # UMAM
#             query_ubcb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_c_businesses[u_id][index2])), dim=0))
#             # UMDM
#             query_ubtb_app.append(torch.cat(list(map(lambda x: item_fea_hete[x], u_b_t_businesses[u_id][index2])), dim=0))

#         query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

#         pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(support_ub_app, open("{}/{}/support_ub_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(support_ubub_app, open("{}/{}/support_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(support_ubcb_app, open("{}/{}/support_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(support_ubtb_app, open("{}/{}/support_ubtb_{}.pkl".format(output_dir, state, idx), "wb"))
        
#         pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(query_ub_app, open("{}/{}/query_um_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(query_ubub_app,open("{}/{}/query_ubub_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(query_ubcb_app,open("{}/{}/query_ubcb_{}.pkl".format(output_dir, state, idx), "wb"))
#         pickle.dump(query_ubtb_app,open("{}/{}/query_ubtb_{}.pkl".format(output_dir, state, idx), "wb"))
        
#         # data for MeLU
#         pickle.dump(support_x_app_melu, open("{}/{}/support_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
#         pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))
#         pickle.dump(query_x_app_melu, open("{}/{}/query_x_{}.pkl".format(melu_output_dir, state, idx), "wb"))
#         pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(melu_output_dir, state, idx), "wb"))

#         with open("{}/log/{}/supp_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
#             for i, b_id in enumerate(tmp_businesses[indices[:-10]]):
#                 f.write("{}\t{}\t{}\n".format(u_id, b_id, tmp_y[indices[:-10]][i]))
#         with open("{}/log/{}/query_x_{}_u_i_ids.txt".format(output_dir, state, idx), "w") as f:
#             for i, b_id in enumerate(tmp_businesses[indices[-10:]]):
#                 f.write("{}\t{}\t{}\n".format(u_id, b_id,  tmp_y[indices[-10:]][i]))
#         idx += 1
        
#     print(idx)  

# %%
