#!/usr/bin/env python
# coding: utf-8

# # Demo 6: Training Hyperparamters
# 
# Regularization helps interpretability by making KANs sparser. This may require some hyperparamter tuning. Let's see how hyperparameters can affect training

# Load KAN and create_dataset

# In[1]:


from kan import KAN, create_dataset
import torch

f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
dataset['train_input'].shape, dataset['train_label'].shape


# Default setup

# In[2]:


# train the model
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.1);
model.plot()
model.prune()
model.plot(mask=True)


# ### Parameter 1: $\lambda$, overall penalty strength. 
# 
# Previously $\lambda=0.1$, now we try different $\lambda$.

# $\lambda=0$

# In[3]:


# train the model
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.00);
model.plot()
model.prune()
model.plot(mask=True)


# $\lambda=10^{-2}$

# In[4]:


# train the model
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=10.);
model.plot()
model.prune()
model.plot(mask=True)


# $\lambda=1$

# In[5]:


# train the model
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=1);
model.plot()
model.prune()
model.plot(mask=True)


# ### Parameter 2: (relative) penalty strength of entropy $\lambda_{\rm ent}$.
# 
# The absolute magnitude is $\lambda\lambda_{\rm ent}$. Previously we set $\lambda=0.1$ and $\lambda_{\rm ent}=10.0$. Below we fix $\lambda=0.1$ and vary $\lambda_{\rm ent}$.

# $\lambda_{\rm ent}=0.0$

# In[6]:


# train the model
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=0.0);
model.plot()
model.prune()
model.plot(mask=True)


# $\lambda_{\rm ent}=10.$

# In[7]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=10.0);
model.plot()
model.prune()
model.plot(mask=True)


# $\lambda_{\rm ent}=100.$

# In[8]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.1, lamb_entropy=100.0);
model.plot()
model.prune()
model.plot(mask=True)


# ### Parameter 3: Grid size $G$. 
# 
# Previously we set $G=5$, we vary $G$ below.

# $G=1$

# In[9]:


model = KAN(width=[2,5,1], grid=1, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
model.plot()
model.prune()
model.plot(mask=True)


# $G=3$

# In[10]:


model = KAN(width=[2,5,1], grid=3, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
model.plot()
model.prune()
model.plot(mask=True)


# $G=5$

# In[11]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
model.plot()
model.prune()
model.plot(mask=True)


# $G=10$

# In[12]:


model = KAN(width=[2,5,1], grid=10, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
model.plot()
model.prune()
model.plot(mask=True)


# $G=20$

# In[13]:


model = KAN(width=[2,5,1], grid=20, k=3, seed=0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.);
model.plot()
model.prune()
model.plot(mask=True)


# ### Parameter 4: seed. 
# 
# Previously we use seed = 0. Below we vary seed.

# ${\rm seed} = 1$

# In[14]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=1, noise_scale_base=0.0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
model.plot()
model.prune()
model.plot(mask=True)


# ${\rm seed} = 42$

# In[15]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=42, noise_scale_base=0.0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
model.plot()
model.prune()
model.plot(mask=True)


# ${\rm seed} = 2024$

# In[16]:


model = KAN(width=[2,5,1], grid=5, k=3, seed=2024, noise_scale_base=0.0)
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
model.plot()
model.prune()
model.plot(mask=True)


# In[ ]:




