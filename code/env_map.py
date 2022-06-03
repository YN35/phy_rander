
import torch

class Env_map():
    
    def __init__(self) -> None:
        self.__SGLobes = torch.tensor([[0,-1,-1]]).cuda().float()
        self.__SGLambdas = torch.tensor([[0.9]]).cuda().float()
        self.__SGMus = torch.tensor([[1,1,1]]).cuda().float()
        
    def get_envSGs(self):#sgの数 x SGのそそれぞれのパラメータ(:3 Lobes  3:4 Lambda  -3:Mu)計7つ
        envSGs = torch.empty(1,7).cuda().float()
        envSGs[..., :3] = self.__SGLobes
        envSGs[..., 3:4] = self.__SGLambdas
        envSGs[..., -3:] = self.__SGMus
        return envSGs