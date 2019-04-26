
import pickle

import numpy as np
import pickle
import torch
import os
from tqdm import tqdm

# change the dim and class_num to generate you PEDCC-weights

latent_variable_dim = 512
class_num=100  # The number of class centroids
PEDCC_root=r"./"
PEDCC_ui=os.path.join(PEDCC_root,str(class_num)+"_"+str(latent_variable_dim)+".pkl")

if not os.path.isdir(PEDCC_root):
    os.makedirs(PEDCC_root)

mu=np.zeros(latent_variable_dim)
sigma = np.eye(latent_variable_dim)

u_init = np.random.multivariate_normal(mu,sigma,class_num)
v = np.zeros(u_init.shape)
u=[]
for i in u_init:
    i=i/np.linalg.norm(i)
    u.append(i)
u=np.array(u)
G=1e-2

def countnext(u,v,G):
    num = u.shape[0]
    dd = np.zeros((num, num))
    for m in range(num):
        for n in range(num):
            dd[m,n] = np.linalg.norm(u[m,:] - u[n,:])
            dd[n,m] = dd[m,n]

    dd[dd<1e-2] = 1e-2
    F = np.zeros((latent_variable_dim,num))
    for m in range(num):
        for n in range(num):
            F[:,m] += (u[m,:]-u[n,:])/((dd[m][n])**3)
    F=F.T
    tmp_F=[]
    for i in range(F.shape[0]):
        tmp_F.append(np.dot(F[i],u[i]))
    d = np.array(tmp_F).T.reshape(len(tmp_F), 1)
    Fr = u*np.repeat(d, latent_variable_dim, 1)
    Ft = F-Fr
    u = u+v
    ll = np.sum(u**2,1)**0.5
    u=u/np.repeat(ll.reshape(ll.shape[0],1),latent_variable_dim,1)
    v = v+G*Ft
    return u,v

def generate_center(u,v,G):
    for i in tqdm(range(200)):
        un,vn=countnext(u,v,G)
        u=un
        v=vn
    print(" ")
    print("Generate Done!")
    # return u*(latent_variable_dim)**0.5
    return u

r1=generate_center(u,v,G)

f=open('./tmp.pkl', 'wb')
pickle.dump(r1,f)
f.close()

ff=open("./tmp.pkl", 'rb')
b=pickle.load(ff)
ff.close()

os.remove("./tmp.pkl")

## distance matrix
# result=np.zeros((len(b),len(b)))
# for a in range(len(b)):
#     for aa in range(a,len(b)):
#         result[a][aa]=np.linalg.norm((b[a]-b[aa]))
#         result[aa][a] = result[a][aa]

fff=open(PEDCC_ui,'wb')
map={}
for i in range(len(b)):
    map[i]=torch.from_numpy(np.array([b[i]]))
pickle.dump(map,fff)
fff.close()


