#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

from torch import nn


def calculate_selection_score(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            scores = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = (scores >= score_threshold).float()
            else:
                scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
        return scores

def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            mask = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            
            scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
            mask = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda())
            
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                mask = torch.bitwise_or((scores >= score_threshold), mask).float()
        
        return mask


def render_edit(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, text_feature : torch.Tensor, edit_dict : dict,
                scaling_modifier = 1.0, override_color = None): 
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    semantic_feature = pc.get_semantic_feature

    


    
    positive_ids = edit_dict["positive_ids"]
    score_threshold = edit_dict["score_threshold"]
    op_dict = edit_dict["operations"]

    # edtiing
    if "deletion" in op_dict:
        scores = calculate_selection_score_delete(semantic_feature[:, 0, :], text_feature, 
                                       score_threshold=score_threshold, positive_ids=positive_ids) # # torch.Size([331617])
        opacity.masked_fill_(scores[:, None] >= 0.5, 0)
        # print(scores) # tensor(1., device='cuda:0') tensor(0., device='cuda:0')
    if "extraction" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], text_feature, 
                                       score_threshold=score_threshold, positive_ids=positive_ids)
        opacity.masked_fill_(scores[:, None] <= 0.5, 0)
    if "color_func" in op_dict:
        scores = calculate_selection_score(semantic_feature[:, 0, :], text_feature, 
                                       score_threshold=score_threshold, positive_ids=positive_ids)
        shs[:, 0, :] = shs[:, 0, :] * (1 - scores[:, None]) + op_dict["color_func"](shs[:, 0, :]) * scores[:, None]
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, feature_map, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map}


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature
    var_loss = torch.zeros(1,viewpoint_camera.image_height,viewpoint_camera.image_width) ###d

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, feature_map, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        semantic_feature = semantic_feature, 
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'feature_map': feature_map,
            "depth": depth} ###d

from gsplat.project_gaussians import project_gaussians
from gsplat.sh import spherical_harmonics
from gsplat.rasterize import rasterize_gaussians
def gsplat_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    img_height = int(viewpoint_camera.image_height)
    img_width = int(viewpoint_camera.image_width)

    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means3d=pc.get_xyz,
        scales=pc.get_scaling,
        glob_scale=scaling_modifier,
        quats=pc.get_rotation,
        viewmat=viewpoint_camera.world_view_transform.T,
        # projmat=viewpoint_camera.full_projection.T,
        fx=focal_length_x,
        fy=focal_length_y,
        cx=img_width / 2.,
        cy=img_height / 2.,
        img_height=img_height,
        img_width=img_width,
        block_width=16,
    )

    try:
        xys.retain_grad()
    except:
        pass

    viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
    # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
    rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

    # opacities = pc.get_opacity
    # if self.anti_aliased is True:
    #     opacities = opacities * comp[:, None]

    def rasterize_features(input_features, bg, distilling: bool = False):
        opacities = pc.get_opacity
        if distilling is True:
            opacities = opacities.detach()
        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            input_features,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=16,
            background=bg,
            return_alpha=False,
        ).permute(2, 0, 1)

    rgb = rasterize_features(rgbs, bg_color)
    depth = rasterize_features(depths.unsqueeze(-1).repeat(1, 3), torch.zeros((3,), dtype=torch.float, device=bg_color.device))

    semantic_features = pc.get_semantic_feature.squeeze(1)
    output_semantic_feature_map_list = []
    chunk_size = 32
    bg_color = torch.zeros((chunk_size,), dtype=torch.float, device=bg_color.device)
    for i in range(semantic_features.shape[-1] // chunk_size):
        start = i * chunk_size
        output_semantic_feature_map_list.append(rasterize_features(
            semantic_features[..., start:start + chunk_size],
            bg_color,
            distilling=True,
        ))
    feature_map = torch.concat(output_semantic_feature_map_list, dim=0)

    return {
        "render": rgb,
        "depth": depth[:1],
        'feature_map': feature_map,
        "viewspace_points": xys,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
