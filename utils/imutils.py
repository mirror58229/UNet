from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

import cv2
import random
# from utils.projection import surface_projection
from copy import deepcopy
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)



def vis_img(im):
    ratiox = 300/int(im.shape[0])
    ratioy = 300/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow("img",0)
    cv2.resizeWindow("img",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    cv2.moveWindow("img",0,0)
    if im.max() > 1:
        im = im/255.
    cv2.imshow('img',im)
    cv2.waitKey()


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def calc_aabb(ptSets):
    lt = np.array([ptSets[0][0], ptSets[0][1]])
    rb = lt.copy()
    for pt in ptSets:
        if pt[0] == 0 and pt[1] == 0:
            continue
        lt[0] = min(lt[0], pt[0])
        lt[1] = min(lt[1], pt[1])
        rb[0] = max(rb[0], pt[0])
        rb[1] = max(rb[1], pt[1])

    return lt, rb

def drawkp(image, pts):
    src_image = deepcopy(image)
    bones = [
        [0, 1, 255, 0, 0],
        [1, 2, 255, 0, 0],
        [2, 12, 255, 0, 0],
        [3, 12, 0, 0, 255],
        [3, 4, 0, 0, 255],
        [4, 5, 0, 0, 255],
        [12, 9, 0, 0, 255],
        [9, 10, 0, 0, 255],
        [10, 11, 0, 0, 255],
        [12, 8, 255, 0, 0],
        [8, 7, 255, 0, 0],
        [7, 6, 255, 0, 0],
        [12, 13, 0, 255, 0]
    ]

    for pt in pts:
        # if pt[2] > 0.2:
        cv2.circle(src_image, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        # if pa[2] > 0.2 and pb[2] > 0.2:
        cv2.line(src_image, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 2)
    cv2.imshow('kp2d',src_image)
    cv2.waitKey(0)


def synthesize_occlusion(img, occlusion, mask, lt_s, rb_s, out_mask):
    # occlusion size
    lt = lt_s.copy()
    rb = rb_s.copy()
    lt = np.clip(lt, 0, 255)
    rb = np.clip(rb, 0, 255)

    human_size = (rb-lt).max()
    oc_size = np.array(occlusion.shape[:2]).max()
    ratio = (human_size * random.uniform(0.1, 0.9)) / oc_size
    sizex = int(occlusion.shape[1] * ratio)
    sizey = int(occlusion.shape[0] * ratio)
    if sizex < 5 or sizey < 5:
        return img, out_mask
    occlusion = cv2.resize(occlusion, (int(occlusion.shape[1] * ratio), int(occlusion.shape[0] * ratio)))
    mask = cv2.resize(mask, (int(mask.shape[1] * ratio), int(mask.shape[0] * ratio)))
    mask[np.where(mask<127)] = 0
    
    # occlusion position
    temp = np.zeros((img.shape[0]*3, img.shape[1]*3, 3))
    temp_mask = np.zeros((img.shape[0]*3, img.shape[1]*3))
    
    randx = img.shape[0] + random.randint(int(lt[1] - occlusion.shape[1]), int(rb[1]))
    randy = img.shape[1] + random.randint(int(lt[0] - occlusion.shape[0]), int(rb[0]))

    temp[randx:randx+occlusion.shape[0],randy:randy+occlusion.shape[1],:] = occlusion
    temp_mask[randx:randx+occlusion.shape[0],randy:randy+occlusion.shape[1]] = mask

    occlusion = temp[img.shape[0]:img.shape[0]*2,img.shape[1]:img.shape[1]*2,:]
    mask = temp_mask[img.shape[0]:img.shape[0]*2,img.shape[1]:img.shape[1]*2]

    img[np.where(mask>0)] = occlusion[np.where(mask>0)]
    out_mask[np.where(mask>0)] = 0.

    return img, out_mask

def get_transform(center, h, w, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix

    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def get_transform_new(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t



def trans(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform_new(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def croppad(image, mask, label, lt, rb):
    w, h, c = image.shape
    center = (rb + lt) / 2
    f = 255
    img_size = 256
    content_size = rb - lt
    pd = content_size.max()*0.0
    offset = np.array([random.uniform(-lt[0]-pd,f-rb[0]+pd), random.uniform(-lt[1]-pd,f-rb[1]+pd)])
    offlt = lt + offset
    offrb = rb + offset
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    image = cv2.warpAffine(image, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (img_size, img_size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0))
    label[:, 0] = label[:, 0] + offset[0]
    label[:, 1] = label[:, 1] + offset[1]
    return image, mask, label, offlt, offrb

def color_gamma_contrast(image):
    alpha = 1.0 + random.uniform(-0.5, 0.5)
    beta = (1.0 - alpha)*0.5
    gamma = random.uniform(0.2, 2.0) #<1, brighter; >1, darker 
    image[:] = (pow(image[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    return image

def color_gamma_contrast_patch(image, patch):
    alpha = 1.0 + random.uniform(-0.5, 0.5)
    beta = (1.0 - alpha)*0.5
    gamma = random.uniform(0.2, 1.5) #<1, brighter; >1, darker 
    image[:] = (pow(image[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    patch[:] = (pow(patch[:]/255.0, gamma)*alpha + beta).clip(0,1) * 255.0
    return image, patch

def scale(image, img_mask, label, lt, rb):
    s1 = random.uniform(0.3, 0.9) # scale range
    w, h, c = image.shape
    content_size = rb - lt
    s = min(255 * s1 / content_size[0], 255 * s1 / content_size[1])
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    img_mask = cv2.resize(img_mask, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    label[:, :2] = label[:, :2] * s
    lt[0] = int(lt[0] * s)
    lt[1] = int(lt[1] * s)
    rb[0] = int(rb[0] * s)
    rb[1] = int(rb[1] * s)
    return image, img_mask, label, lt, rb, s

def random_mask(image, lt, rb):
    num_rect = random.randint(1, 4)
    mask = np.ones(image.shape[:2])
    u1 = lt[0]
    u2 = rb[0]
    v1 = lt[1]
    v2 = rb[1]

    for i in range(num_rect):
        x = np.random.randint(u1, u2, size=2)
        y = np.random.randint(v1, v2, size=2)
        # 小于5像素作为噪声
        if abs(x[0] - x[1]) < 5 and abs(x[0] - x[1]) < 5:
            continue
        else:
            mask = cv2.rectangle(mask, (x[0], y[0]), (x[1], y[1]), (0), -1)

    return mask

def resize(image, label, cropsize):
        w, h, c = image.shape
        dst_image = cv2.resize(image, (cropsize, cropsize), interpolation=cv2.INTER_CUBIC)
        ratio = cropsize / w
        label[:, :2] = label[:, :2] * ratio
        return dst_image, label

def estimate_translation(S, joints_2d, focal_length=5000, img_size=256):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
   # weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
   # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def rot_mesh(mesh, J3d, gt3d):
    G3d = gt3d.copy()
    J = J3d.copy()
    cent_J = np.mean(J, axis=0, keepdims=True)
    J -= cent_J
    cent_G = np.mean(G3d, axis=0, keepdims=True)
    G3d -= cent_G
    M = np.dot(J.T, G3d)
    U, D, V = np.linalg.svd(M) 
    R = np.dot(V.T, U.T)
    out_mesh = np.dot(mesh, R)
    out_joint = np.dot(J3d, R)
    return out_mesh, out_joint

def surface_project(vertices, exter, intri):
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_v)
    mesh_3d = out_point.transpose(1,0)[:,:3]
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    mesh_2d = (out_point.astype(np.int32)).transpose(1,0)
    return mesh_3d, mesh_2d

def wp_project(mesh, J3d, J2d, face, image, focal=5000.):
    f = focal
    cy = image.shape[0] / 2.
    cx = image.shape[1] / 2.
    wp_cam_intri  = np.array([[f,0,cx], [0,f,cy], [0,0,1]])
    init_extri = np.eye(4)
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_np(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy)
    init_extri = np.eye(4)
    init_extri[:3,3] = gt_cam_t
    mesh_proj = surface_project(mesh, init_extri, wp_cam_intri)
    return mesh_proj

def img_reshape(image):
    w, h = image.shape[:2]
    f = 256
    if w > h: 
        M1 = np.float32([[1, 0, (w-h)/2.], [0, 1, 0]])
    else:
        M1 = np.float32([[1, 0, 0], [0, 1, (h-w)/2.]])
    image = cv2.warpAffine(image, M1, (f, f), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0))
    return image
