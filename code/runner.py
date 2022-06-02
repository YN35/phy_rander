import os

import rendering
import camera

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    cam = camera.Camera([5,0,0], [-1,0,0], [0,0,1], 300, 150, 90)
    rend = rendering.Render()
        
    cam_pos = cam.get_cam_pos()
    pix_raydir = cam.get_pix_raydir()
    print(pix_raydir[:,-1,-1])
    print(pix_raydir[:,0,0])
    image = rend.get_pix_color(cam_pos,pix_raydir)
    cam.update_image(image)
    
    
    out = cam.get_image()
    
    