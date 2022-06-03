import os

import torch

import rendering
import camera

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    cam = camera.Camera([5,0,0], [-1,0,0], [0,0,1], 300, 150, 90)
        
    cam_pos_ = cam.get_cam_pos()
    pix_raydir_ = cam.get_pix_raydir()
    # print(pix_raydir_[:,-1,-1])
    # print(pix_raydir_[:,0,0])
    rend = rendering.Render(cam_pos_,pix_raydir_)
    
    image_masked = rend.get_pix_color()
    image_2d = torch.empty(cam.get_pix_hight()*cam.get_pix_width(), 3).cuda().float()
    image_2d[rend.get_surface_mask()] = image_masked
    image_3d = cam.conv_2d_to_3d(image_2d)
    
    cam.update_image(image_3d)
    
    
    out = cam.get_image()
    
    