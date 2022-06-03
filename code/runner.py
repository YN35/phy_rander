import os

import torch

import rendering
import camera

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    pix_hight = 1000
    pix_width = 2000
    
    
    ######################################################
    cam = camera.Camera([5,0,0], [-1,0,0], [0,0,1], pix_width, pix_hight, 90)
     
    cam_pos_ = cam.get_cam_pos()
    pix_raydir_ = cam.get_pix_raydir()
    rend = rendering.Render(cam_pos_,pix_raydir_,pix_width,pix_hight)
    
    image_masked = rend.get_pix_color()
    image_2d = torch.zeros(cam.get_pix_hight()*cam.get_pix_width(), 3).cuda().float()
    image_2d[rend.get_surface_mask()[:,0],:] = image_masked['sg_rgb']
    ######################################################
    
    image_3d = cam.conv_2d_to_3d(image_2d)
    
    cam.update_image(image_3d)
    
    cam.save_image()
    
    
    out = cam.get_image()
    
    