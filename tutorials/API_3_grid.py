#!/usr/bin/env python
# coding: utf-8

# # Demo 3: Grid

# One important feature of KANs is that they embed splines to neural networks. However, splines are only valid for approximating functions in known bounded regions, while the range of activations in neural networks may be changing over training. So we have to update grids properly according to that. Let's first take a look at how we parametrize splines. 

# In[1]:


from kan.spline import B_batch
import torch
import matplotlib.pyplot as plt
import numpy as np

# consider a 1D example.
# Suppose we have grid in [-1,1] with G intervals, spline order k
G = 5
k = 3
grid = torch.linspace(-1,1,steps=G+1)[None,:]

# and we have sample range in [-1,1]
x = torch.linspace(-1,1,steps=1001)[None,:]

basis = B_batch(x, grid, k=k)

for i in range(G+k):
    plt.plot(x[0].detach().numpy(), basis[0,i,:].detach().numpy())
    
plt.legend(['B_{}(x)'.format(i) for i in np.arange(G+k)])
plt.xlabel('x')
plt.ylabel('B_i(x)')


# There are $G+k$ B-spline basis. The function is a linear combination of these bases $${\rm spline}(x)=\sum_{i=0}^{G+k-1} c_i B_i(x).$$ We don't need worry about the implementation since it's already built in KAN. But let's check if KAN is indeed implementing this. We initialize a [1,1] KAN, which is simply a 1D spline.

# In[2]:


from kan import KAN

model = KAN(width=[1,1], grid=G, k=k)
# obtain coefficients c_i
model.act_fun[0].coef
assert(model.act_fun[0].coef[0].shape[0] == G+k)

# the model forward
model_output = model(x[0][:,None])

# spline output
spline_output = torch.einsum('i,ij->j',model.act_fun[0].coef[0], basis[0])[:,None]

torch.mean((model_output - spline_output)**2)


# They are not the same, what's happening? We want to remind that we model the activation function to have two additive parts, a residual function $b$(x) plus the spline function, i.e., $$\phi(x)={\rm scale\_base}*b(x)+{\rm scale\_sp}*{\rm spline}(x),$$ and by default $b(x)={\rm silu}(x)=x/(1+e^{-x})$.

# In[3]:


# residual output
residual_output = torch.nn.SiLU()(x[0][:,None])
scale_base = model.act_fun[0].scale_base
scale_sp = model.act_fun[0].scale_sp
torch.mean((model_output - (scale_base * residual_output + scale_sp * spline_output))**2)


# What if my grid does not match my data? For example, my grid is in [-1,1], but my data is in [10,10] or [-0.5,0.5]. Use update_grid_from_sample to adjust grids to samples. This grid update applies to all splines in all layers.

# In[4]:


model = KAN(width=[1,1], grid=G, k=k)
print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
x = torch.linspace(-10,10,steps = 1001)[:,None]
model.update_grid_from_samples(x)
print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


# In[5]:


model = KAN(width=[1,1], grid=G, k=k)
print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
x = torch.linspace(-0.5,0.5,steps = 1001)[:,None]
model.update_grid_from_samples(x)
print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


# Uniform grid or non-uniform? We consider two options: (1) uniform grid; (2) adaptive grid (based on sample distribution) such that there are (rougly) same number of samples in each interval. We provide a parameter grid_eps to interpolate between these two regimes. grid_eps = 1 gives (1), and grid_eps = 0 gives (0). By default we set grid_eps = 1 (uniform grid). There could be other options but it is out of our scope here.

# In[6]:


# uniform grid
model = KAN(width=[1,1], grid=G, k=k)
print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
x = torch.normal(0,1,size=(1000,1))
model.update_grid_from_samples(x)
print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


# In[7]:


# adaptive grid based on sample distribution
model = KAN(width=[1,1], grid=G, k=k, grid_eps = 0.)
print(model.act_fun[0].grid) # by default, the grid is in [-1,1]
x = torch.normal(0,1,size=(1000,1))
model.update_grid_from_samples(x)
print(model.act_fun[0].grid) # now the grid becomes in [-10,10]. We add a 0.01 margin in case x have zero variance


# In[ ]:




