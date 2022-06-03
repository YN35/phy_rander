


import torch
import numpy as np

import camera
import object
import env_map

class Render():
    
    def __init__(self,cam_pos_,pix_raydir_,pix_width,pix_hight,max_distance=100,allowable_error=0.005,blending_weights=None, diffuse_rgb=None) -> None:
        self.TINY_NUMBER = 1e-6
        self.blending_weights = blending_weights
        self.diffuse_rgb = diffuse_rgb
        
        self.__cam_pos_ = cam_pos_
        self.__pix_raydir_ = pix_raydir_
        self.__pix_hight = pix_hight
        self.__pix_width = pix_width
        # self.__omega_0_ = self.__cam_pos_ - self.__x_reflect_
        # self.__omega_0_ = self.__omega_0_ / torch.norm(self.__omega_0_, dim=0)

        self.shape = object.Shape()
        self.mate = object.Material()
        self.envmp = env_map.Env_map()
        
        self.__x_reflect_, self.__surface_mask = self.ray_marching(max_distance,allowable_error)
        
    
    def get_pix_color(self):
        envSGs_ = self.envmp.get_envSGs()
        normal_ = self.shape.get_normal(self.__x_reflect_)
        pix_raydir_ = self.__pix_raydir_[self.__surface_mask[:,0],:]
        roughness = self.mate.get_roughness()
        specular_reflectance = self.mate.get_specular_reflectance()
        diffuse_albedo = self.mate.get_diffuse_albedo(self.__x_reflect_)
        
        M = envSGs_.shape[0]
        K = self.mate.get_specular_reflectance().shape[0]
        assert (K == roughness.shape[0])
        dots_shape = list(normal_.shape[:-1])
        
        normal_ = normal_.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])
        pix_raydir_ = pix_raydir_.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])
        
        envSGs_ = self.prepend_dims(envSGs_, dots_shape)
        envSGs_ = envSGs_.unsqueeze(-2).expand(dots_shape + [M, K, 7])
        lgtSGLobes = envSGs_[..., :3] / (torch.norm(envSGs_[..., :3], dim=-1, keepdim=True) + self.TINY_NUMBER)
        lgtSGLambdas = torch.abs(envSGs_[..., 3:4])
        lgtSGMus = torch.abs(envSGs_[..., -3:])
        
        brdfSGLobes = normal_
        inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)
        brdfSGLambdas = self.prepend_dims(2. * inv_roughness_pow4, dots_shape + [M, ])
        mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])
        brdfSGMus = self.prepend_dims(mu_val, dots_shape + [M, ])
        
        v_dot_lobe = torch.sum(brdfSGLobes * pix_raydir_, dim=-1, keepdim=True)
        
        v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
        warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - pix_raydir_
        warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + self.TINY_NUMBER)
        
        warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + self.TINY_NUMBER)
        warpBrdfSGMus = brdfSGMus
        
        new_half = warpBrdfSGLobes + pix_raydir_
        new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + self.TINY_NUMBER)
        v_dot_h = torch.sum(pix_raydir_ * new_half, dim=-1, keepdim=True)
        
        v_dot_h = torch.clamp(v_dot_h, min=0.)
        specular_reflectance = self.prepend_dims(specular_reflectance, dots_shape + [M, ])  # [..., M, K, 3]
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)
        
        dot1 = torch.sum(warpBrdfSGLobes * normal_, dim=-1, keepdim=True)  # equals <o, n>
        ### note: for numeric stability
        dot1 = torch.clamp(dot1, min=0.)
        dot2 = torch.sum(pix_raydir_ * normal_, dim=-1, keepdim=True)  # equals <o, n>
        ### note: for numeric stability
        dot2 = torch.clamp(dot2, min=0.)
        k = (roughness + 1.) * (roughness + 1.) / 8.
        G1 = dot1 / (dot1 * (1 - k) + k + self.TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + self.TINY_NUMBER)
        G = G1 * G2

        Moi = F * G / (4 * dot1 * dot2 + self.TINY_NUMBER)
        warpBrdfSGMus = warpBrdfSGMus * Moi

        # multiply with light sg
        final_lobes, final_lambdas, final_mus = self.lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                            warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

        # now multiply with clamped cosine, and perform hemisphere integral
        mu_cos = 32.7080
        lambda_cos = 0.0315
        alpha_cos = 31.7003
        lobe_prime, lambda_prime, mu_prime = self.lambda_trick(normal_, lambda_cos, mu_cos,
                                                        final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal_, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal_, dim=-1, keepdim=True)
        # [..., M, K, 3]
        specular_rgb = mu_prime * self.hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * self.hemisphere_int(final_lambdas, dot2)
        
        if self.blending_weights is None:     
            specular_rgb = specular_rgb.sum(dim=-2).sum(dim=-2)
        else:
            specular_rgb = (specular_rgb.sum(dim=-3) * self.blending_weights.unsqueeze(-1)).sum(dim=-2)
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        # ### debug
        # if torch.sum(torch.isnan(specular_rgb)) + torch.sum(torch.isinf(specular_rgb)) > 0:
        #     print('stopping here')
        #     import pdb
        #     pdb.set_trace()

        ########################################
        # per-point hemisphere integral of envmap
        ########################################
        if self.diffuse_rgb is None:
            diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, 1, 3])

            # multiply with light sg
            final_lobes = lgtSGLobes.narrow(dim=-2, start=0, length=1)  # [..., M, K, 3] --> [..., M, 1, 3]
            final_mus = lgtSGMus.narrow(dim=-2, start=0, length=1) * diffuse
            final_lambdas = lgtSGLambdas.narrow(dim=-2, start=0, length=1)

            # now multiply with clamped cosine, and perform hemisphere integral
            lobe_prime, lambda_prime, mu_prime = self.lambda_trick(normal_, lambda_cos, mu_cos,
                                                            final_lobes, final_lambdas, final_mus)

            dot1 = torch.sum(lobe_prime * normal_, dim=-1, keepdim=True)
            dot2 = torch.sum(final_lobes * normal_, dim=-1, keepdim=True)
            self.diffuse_rgb = mu_prime * self.hemisphere_int(lambda_prime, dot1) - \
                        final_mus * alpha_cos * self.hemisphere_int(final_lambdas, dot2)
            self.diffuse_rgb = self.diffuse_rgb.sum(dim=-2).sum(dim=-2)
            self.diffuse_rgb = torch.clamp(self.diffuse_rgb, min=0.)

        # combine diffue and specular rgb, then return
        rgb = specular_rgb + self.diffuse_rgb
        ret = {'sg_rgb': rgb,
            'sg_specular_rgb': specular_rgb,
            'sg_diffuse_rgb': self.diffuse_rgb,
            'sg_diffuse_albedo': diffuse_albedo}

        return ret
        
    def ray_marching(self, max_distance,allowable_error):
        """_summary_

        Args:
            max_distance (float): どのくらいの距離光線が離れたら衝突しなかっとみなすか
            allowable_error (float): 交差点算出の許容誤差

        Returns:
            Tensor(2d): それぞれのピクセルから出た光線の交差する場所
            Tensor(2d): それぞれのピクセルにオブジェクトの像が存在するか
        """
        
        ray_edge_pos_ = torch.empty(self.__pix_hight*self.__pix_width, 3).cuda().float()
        ray_edge_pos_[:,:] = self.__cam_pos_
        sdf_ = self.shape.get_sdf(ray_edge_pos_)#sdf_.shape[0]
        while torch.sum(sdf_ >= max_distance) + torch.sum(sdf_ <= allowable_error) != torch.numel(sdf_) :
            ray_edge_pos_ = ray_edge_pos_ + (self.__pix_raydir_ * torch.t(sdf_))
            sdf_ = self.shape.get_sdf(ray_edge_pos_)
            
        surface_mask = torch.t(sdf_ <= max_distance)
        x_reflect_ = ray_edge_pos_[surface_mask[:,0],:]
            
        return x_reflect_, surface_mask
    
    def prepend_dims(self, tensor, shape):
        '''
        :param tensor: tensor of shape [a1, a2, ..., an]
        :param shape: shape to prepend, e.g., [b1, b2, ..., bm]
        :return: tensor of shape [b1, b2, ..., bm, a1, a2, ..., an]
        '''
        orig_shape = list(tensor.shape)
        tensor = tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))
        return tensor
    
    def lambda_trick(self, lobe1, lambda1, mu1, lobe2, lambda2, mu2):
        # assume lambda1 << lambda2
        ratio = lambda1 / lambda2

        dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
        tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
        tmp = torch.min(tmp, ratio + 1.)

        lambda3 = lambda2 * tmp
        lambda1_over_lambda3 = ratio / tmp
        lambda2_over_lambda3 = 1. / tmp
        diff = lambda2 * (tmp - ratio - 1.)

        final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
        final_lambdas = lambda3
        final_mus = mu1 * mu2 * torch.exp(diff)

        return final_lobes, final_lambdas, final_mus
    
    def hemisphere_int(self, lambda_val, cos_beta):
        lambda_val = lambda_val + self.TINY_NUMBER
        # orig impl; might be numerically unstable
        # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)

        inv_lambda_val = 1. / lambda_val
        t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                    1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

        # orig impl; might be numerically unstable
        # a = torch.exp(t)
        # b = torch.exp(t * cos_beta)
        # s = (a * b - 1.) / ((a - 1.) * (b + 1.))

        ### note: for numeric stability
        inv_a = torch.exp(-t)
        mask = (cos_beta >= 0).float()
        inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
        s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
        b = torch.exp(t * torch.clamp(cos_beta, max=0.))
        s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
        s = mask * s1 + (1. - mask) * s2

        A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
        A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

        return A_b * (1. - s) + A_u * s
    
    def get_surface_mask(self):
        return self.__surface_mask