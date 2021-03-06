

from operator import imod


import torch

class Material():
    
    def __init__(self) -> None:
        self.__roughness = torch.tensor([[0.8304324]]).cuda().float()
        self.__specular_reflectance = torch.tensor([[0.7752, 0.7752, 0.7752]]).cuda().float()
    
    def get_diffuse_albedo(self, x_reflect_):
        x_reflect_[:,:] = torch.tensor([0.1,0.1,0.1]).cuda().float()
        return x_reflect_
    
    def get_specular_reflectance(self):
        return self.__specular_reflectance
    
    def get_roughness(self):
        return self.__roughness
        

class Shape():
    def __init__(self) -> None:
        pass
    
    def get_normal(self, x_):
        return x_ / torch.norm(x_, dim=1).unsqueeze(1).expand(x_.shape)
    
    def get_sdf(self, x_):
        return (torch.norm(x_, dim=1) - 1).unsqueeze(0)