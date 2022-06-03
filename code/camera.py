import math

import numpy as np
import torch

class Camera():
    
    def __init__(self,cam_pos,cam_dir,view_up,pix_width,pix_hight,fov) -> None:
        self.__cam_pos_ = torch.tensor(cam_pos).cuda().float()
        self.__cam_dir_ = torch.tensor(cam_dir).cuda().float()
        self.__cam_dir_ = (self.__cam_dir_ / torch.norm(self.__cam_dir_))
        self.__view_up_ = torch.tensor(view_up).cuda().float()
        self.__view_up_ = (self.__view_up_ / torch.norm(self.__view_up_))
        self.__fov = fov
        
        self.__pix_width = pix_width
        self.__pix_hight = pix_hight
        
        self.__image = torch.empty(3, self.__pix_hight, self.__pix_width)
        
        
    def get_pix_raydir(self):
        """_summary_

        Returns:
            tensor(2d): _description_
        """
        
        w = math.tan(math.radians(self.__fov/2))
        h = w * (self.__pix_hight / self.__pix_width)
        Z_ =  -self.__cam_dir_
        X_ = torch.cross(self.__view_up_,Z_)
        Y_ = torch.cross(Z_,X_)
        u_ = 2 * w * X_
        v_ = 2 * h * Y_
        w_ = - w*X_ - h*Y_ - Z_
        
        #横がx　縦がy　左上が0
        _pix_raydir = torch.empty(3, self.__pix_hight, self.__pix_width).cuda().float()
        _x_np = np.zeros((self.__pix_hight, self.__pix_width))
        _y_np = np.zeros((self.__pix_hight, self.__pix_width))
        _x_np[:,:] = np.arange(self.__pix_width)
        _y_np[:,:] = np.array([np.arange(self.__pix_hight)]).T
        
        x_ = torch.from_numpy(_x_np).cuda().float()
        y_ = torch.from_numpy(_y_np).cuda().float()
        
        _pix_raydir[0,:,:] = u_[0] * (x_ / self.__pix_width) + v_[0] * (y_ / self.__pix_hight) + w_[0]
        _pix_raydir[1,:,:] = u_[1] * (x_ / self.__pix_width) + v_[1] * (y_ / self.__pix_hight) + w_[1]
        _pix_raydir[2,:,:] = u_[2] * (x_ / self.__pix_width) + v_[2] * (y_ / self.__pix_hight) + w_[2]
        
        # u_x_ = (x_ / self.__pix_width)
        # v_y_ = (y_ / self.__pix_hight)
        # _pix_raydir[0,:,:] = u_ * torch.tensor([u_x_,u_x_,u_x_]).cuda().float() + v_ * torch.tensor([v_y_,v_y_,v_y_]).cuda().float() + w_
        
        #正規化
        _pix_raydir = _pix_raydir / torch.norm(_pix_raydir, dim=0)
        
        _pix_raydir = self.conv_3d_to_2d(_pix_raydir)
        
        return _pix_raydir
    
    def conv_3d_to_2d(self, tensor):
        return tensor.permute(1, 2, 0).view(self.__pix_width * self.__pix_hight, 3)
        
    def conv_2d_to_3d(self, tensor):
        return tensor.permute(1, 0).view(3, self.__pix_width * self.__pix_hight)
        
    def get_image(self):
        return self.__image
    
    def update_image(self, image):
        self.__image = image
        
    def get_cam_pos(self):
        return self.__cam_pos_
    
    def get_pix_width(self):
        return self.__pix_width
    
    def get_pix_hight(self):
        return self.__pix_hight
    