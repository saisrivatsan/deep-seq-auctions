import itertools
import numpy as np
import torch

""" Envs and Distrs"""
class VDistBase:
    def __init__(self, num_items, demand, device = "cuda"):
        
        self.num_items = num_items
        self.device = device
               
        """ All possible subsets """
        self.allocs = np.array(list(itertools.product([0, 1], repeat=num_items)))
        
        """ Filter if k-demand """
        if demand is not None:
            self.allocs = self.allocs[self.allocs.sum(-1) <= demand]
                       
        """ Set number of menus """
        self.num_menus = self.allocs.shape[0]        
        
        """ Create Torch Tensors"""
        self.allocs_tensor = torch.Tensor(self.allocs).to(device = self.device)   
     
    def sample(self):
        """ Function to sample valuations. Re-implement as required """
        V = np.random.rand(self.num_items)
        return self.allocs @ V
    
    def sample_tensor(self, num_samples):
        V = torch.rand(num_samples, self.num_items, device = self.device)
        return V @ self.allocs_tensor.T
    
    def set_action_scale(self, action_scale = [1.0]):
        """ Function to set scale and offset. Re-implement as required """
        self.action_scale = action_scale
        self.action_scale_tensor = torch.Tensor(action_scale).to(device = self.device)
        
        
class UNIF(VDistBase):
    def __init__(self, num_items, demand, device = "cuda"):
        super().__init__(num_items, demand, device = "cuda")
        self.set_action_scale(self.allocs.sum(-1))
                              
class ASYM(VDistBase):
    def __init__(self, num_items, demand, device = "cuda"):
        super().__init__(num_items, demand, device = "cuda")
        self.value_scalers = np.arange(1, self.num_items + 1)/(self.num_items)
        self.value_scalers_tensor = torch.Tensor(self.value_scalers).to(device = self.device)
        ub = self.allocs @ self.value_scalers
        self.set_action_scale(ub)

    def sample(self):
        V = np.random.rand(self.num_items) * self.value_scalers
        return self.allocs @ V

    def sample_tensor(self, num_samples):
        V = torch.rand(num_samples, self.num_items, device = self.device) * self.value_scalers_tensor
        return V @ self.allocs_tensor.T
                              
class COMB1(VDistBase):
    def __init__(self, num_items, demand, device = "cuda"):
        super().__init__(num_items, demand, device = "cuda")
        self.value_scalers = np.sqrt(self.allocs.sum(-1))
        self.value_scalers_tensor = torch.Tensor(self.value_scalers).to(device = self.device)
        ub = self.value_scalers
        self.set_action_scale(ub)

    def sample(self):
        V = np.random.rand(self.num_menus) * self.value_scalers
        return V

    def sample_tensor(self, num_samples):
        V = torch.rand(num_samples, self.num_menus, device = self.device) * self.value_scalers_tensor
        return V
                              
                              
class COMB2(VDistBase):
    def __init__(self, num_items, demand, device = "cuda"):
        super().__init__(num_items, demand, device = "cuda")
        self.value_scalers = self.allocs.sum(-1)
        self.value_scalers_tensor = torch.Tensor(self.value_scalers).to(device = self.device)
        ub = self.value_scalers * 3
        self.set_action_scale(ub)


    def sample(self):
        v = np.random.rand(self.num_items) + 1
        c = (2 * np.random.rand(self.num_menus) - 1) * self.value_scalers
        V = self.allocs @ v + c
        return V

    def sample_tensor(self, num_samples):
        v_samples = torch.rand(num_samples, self.num_items, device = self.device) + 1.0
        c = torch.rand(num_samples, self.num_menus, device = self.device) * self.value_scalers_tensor 
        return v_samples @ self.allocs_tensor.T + c
        
        
class UNIFScale:
    def __init__(self, num_items, demand, device = "cuda"):
        
        self.num_items = num_items
        self.device = device
        self.set_action_scale([1.0])
               
    def sample(self):
        """ Function to sample valuations. Re-implement as required """
        V = np.random.rand(self.num_items)
        return V
    
    def sample_tensor(self, num_samples):
        V = torch.rand(num_samples, self.num_items, device = self.device)
        return V
    
    def set_action_scale(self, action_scale = [1.0]):
        """ Function to set scale and offset. Re-implement as required """
        self.action_scale = action_scale
        self.action_scale_tensor = torch.Tensor(action_scale).to(device = self.device)
        
        