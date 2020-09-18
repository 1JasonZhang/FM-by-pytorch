import torch.nn as nn
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#FM model
class FM_model(nn.Module):
    def __init__(self,p,k):
        super(FM_model,self).__init__()
        self.p = p
        self.k = k
        self.linear = nn.Linear(self.p,1,bias=True)
        self.v = nn.Parameter(torch.randn(self.k,self.p))
    def fm_layer(self,x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x,self.v.t())
        inter_part2 = torch.mm(torch.pow(x,2),torch.pow(self.v,2).t())
        output = linear_part + 0.5*torch.sum(torch.pow(inter_part1,2) - inter_part2)
        return output
    def forward(self,x):
        output = self.fm_layer(x)
        return output

def vectorize_dic(dic,ix=None,p=None,n=0,g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """
    if ix==None:
        ix = dict()

    nz = n * g

    col_ix = np.empty(nz,dtype = int)

    i = 0
    for k,lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k),0) + 1
            col_ix[i+t*g] = ix[str(lis[t]) + str(k)]
        i += 1

    row_ix = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

#data proprecess
cols = ['user','item','rating','timestamp']

train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values,
                            'items':train['item'].values},n=len(train.index),g=2)


x_test,ix = vectorize_dic({'users':test['user'].values,
                           'items':test['item'].values},ix,x_train.shape[1],n=len(test.index),g=2)


y_train = train['rating'].values
y_test = test['rating'].values

x_train = x_train.todense()
x_test = x_test.todense()


print(x_train.shape)
print(x_test.shape)

#train
n,p = x_train.shape
k = 10
batch_size=64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FM_model(p,k).cuda()
loss_fn =nn.MSELoss()
optimer = torch.optim.SGD(model.parameters(),lr=0.0001,weight_decay=0.001)
epochs = 100
for epoch in range(epochs):
    loss_epoch = 0.0
    loss_all = 0.0
    perm = np.random.permutation(x_train.shape[0])
    model.train()
    for x,y in tqdm(batcher(x_train[perm], y_train[perm], batch_size)):
        model.zero_grad()
        x = torch.as_tensor(np.array(x.tolist()), dtype=torch.float,device=device)
        y = torch.as_tensor(np.array(y.tolist()), dtype=torch.float,device=device)
        x = x.view(-1, p)
        y = y.view(-1, 1)
        preds = model(x)
        loss = loss_fn(preds,y)
        loss_all += loss.item()
        loss.backward()
        optimer.step()
    loss_epoch = loss_all/len(x)
    print(f"Epoch [{epoch}/{10}], "
              f"Loss: {loss_epoch:.8f} ")
    
