import os

import rendering
import camera

if __name__ == 'main':
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
    
    cam = camera.Camera()
    rend = rendering.Render()
        
    cam_pos = cam.get_cam_pos()
    pix_raydir = cam.get_pix_raydir()
    image = rend.get_pix_color(cam_pos,pix_raydir)
    cam.update_image(image)
    
    
    out = cam.get_image()
    
    