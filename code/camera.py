import torch

class Camera():
    
    def __init__(self,cam_pos,view_up,pix_width,pix_hight,fov) -> None:
        self.__cam_pos = cam_pos
        self.__view_up = view_up
        self.__fov = fov
        
        self.__pix_width = pix_width
        self.__pix_hight = pix_hight
        
        self.__image = torch.empty(3, self.__pix_hight, self.__pix_width)
        
        
    def get_pix_raydir(self):
            
        
        return pix_raydir
        
    def get_image(self):
        return self.__image
    
    def update_image(self, image):
        self.__image = image
        
    def get_cam_pos(self):
        return self.__cam_pos
    