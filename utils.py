import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


def hello_apna_nerf():
	print('Hello Apna NeRF!')
	return


'''Vectorized implementation for positional encoding using L frequencies'''
def positional_encoding(inp, L):
  
	'''
	Vectorized implementation for positional encoding using L frequencies

	Arguments:
		inp: data of shape Nx3 on which positional encoding has to be applied
		L: integer value denoting number of frequencies to use while encoding

	Return:
	out: positionally encoded data of shape Nx(3*2*L)
	'''

	freq = (2**torch.arange(L))[None,:] #omega = (2^i)*pi i=0,...,L-1
	freq = freq.to(inp.device)
	inp_dim = inp.shape[1]
	out_dim = inp_dim+(inp_dim*2*L)
	out = torch.ones(inp.shape[0], out_dim, dtype=inp.dtype, device=inp.device)
	out[:,:3] = inp.clone()
	arg = torch.matmul(inp.clone()[...,None],freq.to(inp)).transpose(1,2) #Argument to the sinusoidal function (omega*x)
	sin_enc = torch.sin(arg) #[batch_size,L,3]
	cos_enc = torch.cos(arg) #[batch_size,L,3]
	pos_enc_mat = torch.concat([sin_enc, cos_enc],-1) #[batch_size,L,3*2]
	out[:,3:] = pos_enc_mat.reshape(inp.shape[0],-1) #[batch_size,3*2*L]

	return out


def posenc(x,L):
  '''Tiny NeRF positional encoding to check our vectorized implementation'''
  rets = [x]
  for i in range(L):
    for fn in [torch.sin, torch.cos]:
      rets.append(fn(torch.pi*(2**i)*x))
  return torch.concat(rets,-1)


def project_rays(H, W, f, camera_mat):
	'''
	Our implementaion (replaced dot product in the original implementation with matrix multiplication)

	Arguments:
		H: height of the image(integer)
		W: width of the image(integer)
		f: focal length of the camera(float or double)
		camera_mat: camera intrinsic matrix of shape (4x4)

	Return:
		dir_rays: shape of HxWx3, contains direction of ray projected through each pixel
		origin_rays: shape of HxWx3, contains origins of ray projected through each pixel
	'''

	j,i = torch.meshgrid(torch.arange(H, dtype=camera_mat.dtype, device=camera_mat.device),torch.arange(W, dtype=camera_mat.dtype, device=camera_mat.device))
	direction = torch.stack([(i-0.5*W)/f, -(j-0.5*H)/f, -torch.ones_like(i)], -1)
	dir_rays = torch.matmul(direction,camera_mat[:3,:3].T)
	origin_rays = torch.broadcast_to(camera_mat[:3,-1], dir_rays.shape)
	return dir_rays, origin_rays


def get_rays(H, W, f, camera_mat):
	'''PyTorch version of official NeRF code'''
	j,i = torch.meshgrid(torch.arange(H, dtype=camera_mat.dtype, device=camera_mat.device),torch.arange(W, dtype=camera_mat.dtype, device=camera_mat.device))
	dirs = torch.stack([(i-0.5*W)/f, -(j-0.5*H)/f, -torch.ones_like(i)], -1)
	rays_d = torch.sum(dirs[...,None,:]*camera_mat[:3,:3],-1)
	rays_o = camera_mat[:3,-1].expand(rays_d.shape)
	return rays_d, rays_o


def translation_mat(t, dtype, device):
	'''
	Arguments:
		t: translation

	Return:
		mat: translation matrix of shape 4x4 that translates 3D homogenous points by t
	'''
	mat = torch.eye(4, dtype=dtype, device=device)
	mat[2,:-1] = t
	return mat


def rotation_mat(ang, dtype, device):
	'''
	Arguments:
		ang: rotation angle in radians, anti-clockwise direction

	Return:
		mat: 2D rotation matrix of shape 2x2 that rotates by 'ang'
	'''
	c = torch.cos(ang)
	s = torch.sin(ang)
	mat = torch.tensor([[c,-s],[s,c]], dtype=dtype, device=device)
	return mat


def get_mat_from_angle(theta, phi, t, dtype, device = 'cpu'):
	'''
	Generates camera matrix to translate by 't', rotate by 'theta' and 'phi'

	Arguments:
		theta: angle in degrees to rotate about z-axis
		phi: angle in degree to rotate about y-axis
		t: denotes the amount to translate

	Return:
		camera_mat: camera matrix of shape 4x4
	'''
	trans_mat = translation_mat(t)
	rot_mat_phi = torch.eye(4, dtype=dtype, device=device)
	rot_mat_phi[1:3,1:3] = rotation_mat(phi*(np.pi/180), dtype, device)
	rot_mat_theta = torch.eye(4, dtype=dtype, device=device)
	rot_mat_theta[::2,::2] = rotation_mat(theta*(np.pi/180), dtype, device)
	camera_mat = rot_mat_theta @ rot_mat_phi @ trans_mat
	camera_mat[0,:] *=-1
	return camera_mat


def convert_to_batch(model,pts, rays_d, batch_size):
  '''
  Call model in smaller batches
  Input:-
    model : pytorch model instance describing MLP model.
    pts : `N_sample` samples between [tn,tf] along the rays `rays_d`for each pixels in the image
    rays_d : direction rays for each pixel in the image.
    batch_size : Maximum number of rays to process simultaneously. Use it to control memory usage
  
  Output:-
    model output predicting `rgb` and `sigma` value for each pixels in the image.
  '''
  return torch.cat([model.forward(pts[i:i+batch_size],rays_d[i:i+batch_size]) for i in range(0,pts.shape[0],batch_size)],0)



def render_rays(model,rays_o, rays_d, near, far, N_samples, rand=False, batch_size=1024,L=6):
  '''
  Rendering function

  Input:-
    model : Pytorch model instance describing MLP model.
    rays_o : origin ray for each pixels in the image.
    rays_d : direction ray for each pixel in the image.
    near : Near plane.
    far : Far plane.
    N_samples : Number of sample to take along each ray.
    batch_size : Maximum number of rays to process simultaneously. Use it to control memory usage
  
  Output:-
    rgb : Estimated RGB color of a ray.
    depth : Depth map.
    acc : accumulated opacity along each ray. 
  '''
  z_vals = torch.linspace(near,far,N_samples,dtype=rays_o.dtype,device=rays_o.device)
  if(rand):
    sz = list(rays_o.shape[:-1]) + [N_samples]
    temp = torch.rand(sz)*(far-near)/N_samples
    z_vals = z_vals.unsqueeze(0).unsqueeze(0) + temp.to(rays_o)
  pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2)*z_vals.unsqueeze(-1)

  # Run model
  H, W, P, PP = pts.shape
  pts = pts.reshape(-1,3)
  pts = positional_encoding(pts,L)
  
  rays_d = rays_d.unsqueeze(-2)
  rays_d = rays_d.repeat(1, 1, P, 1).reshape(-1,3)
  rays_d = positional_encoding(rays_d, L)
  
  obj = convert_to_batch(model,pts,rays_d,batch_size)
  out_rgb = torch.sigmoid(obj[...,:3])
  out_sigma = torch.relu(obj[...,-1])
  out_rgb = out_rgb.reshape(H, W, P, 3)
  out_sigma = out_sigma.reshape(H, W, P)

  # Do Volume rendering
  diff = z_vals[...,1:]-z_vals[...,:-1]

  sz = list(diff.shape[:-1])+[1]
  distance = torch.concat((diff,torch.ones(sz,dtype=diff.dtype,device=diff.device)*1e10),axis=-1)
  alpha = 1 - torch.exp(-out_sigma*distance)
  
  sz = list(alpha.shape[:-1]) + [1]
  apx = torch.cat([torch.ones(sz,dtype=z_vals.dtype,device=z_vals.device),1.0 -alpha + 1e-10],-1)
  weights = alpha * torch.cumprod(apx,-1)[...,:-1]

  rgb = torch.sum(weights.unsqueeze(-1)*out_rgb,-2)
  depth = torch.sum(weights*z_vals,-1)
  acc = torch.sum(weights,-1)
  
  return rgb,depth,acc


def get_psnr(target, pred):
	'''
	Computes the Peak Signal-to-Noise Ratio

	Arguments:
		target: np.ndarray image which is the target image
		pred: np.ndarray image which is the predicted image
	'''
	mse = np.mean(np.square(pred - target))
	return -10*np.log(mse)/np.log(10.0)


def get_color_ssim(target, pred):
	'''
	Computes the Structure Similarity Index Measure. Note that
	this implementation computes the SSIM value for each channel
	and then combines them by using a weighted average with the
	same weight of 0.333 for each channel

	Arguments:
		target: np.ndarray image which is the target image
		pred: np.ndarray image which is the predicted image
	'''
	target = np.transpose(target, (2,0,1))
	pred = np.transpose(pred, (2,0,1))
	weight = 1/3
	ssim_val = 0.0

	data_range = np.amax(pred[0]) - np.amin(pred[0])
	ssim_val += weight*ssim(target[0], pred[0], data_range=data_range)
	data_range = np.amax(pred[1]) - np.amin(pred[1])
	ssim_val += weight*ssim(target[1], pred[1], data_range=data_range)
	data_range = np.amax(pred[2]) - np.amin(pred[2])
	ssim_val += weight*ssim(target[2], pred[2], data_range=data_range)

	return ssim_val


def get_image_from_pose(H, W, focal, pose, nerf_model, device):
	'''
	Computes the image from an input pose by taking in the parameters
	defined below and returns the predicted RGB image

	Arguments:
		H: An integer which is the image height
		W: An integer which is the image width
		focal: An integer or 1x1 np.ndarray which is the camera focal length
		pose: An np.ndarray pose of shape 4x4
		nerf_model: The trained nerf model
	'''
	pose = torch.tensor(pose).to(device)
	focal = torch.tensor(focal).to(device)
	dir_rays, origin_rays = project_rays(H, W, focal, pose)
	pred_rgb, _, _ = render_rays(nerf_model, origin_rays, dir_rays, near=2., far=6., N_samples=64, rand=True, L=6, batch_size=1024*64)
	return pred_rgb

class NeRF_MLP_original(nn.Module):
  def __init__(self, W: int, inp_dim: int):
    super().__init__()
    self.enc = nn.Sequential(
                nn.Linear(inp_dim,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU()
                )
    self.dec = nn.Sequential(
                nn.Linear(inp_dim+W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU()
                )
    self.feat_to_sigma_rgb = nn.Sequential(
                        nn.Linear(W+inp_dim,4)
                        )
    
  def forward(self, x, d):
    h = self.enc(x)
    feat = self.dec(torch.concat([x,h],dim=-1))
    obj = self.feat_to_sigma_rgb(torch.concat([feat,d], dim=-1))
    return obj

class NeRF_MLP_reduced(nn.Module):
  def __init__(self, W: int, inp_dim: int):
    super().__init__()
    self.enc = nn.Sequential(
                nn.Linear(inp_dim,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU()
                )
    self.dec = nn.Sequential(
                nn.Linear(inp_dim+W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU(),
                nn.Linear(W,W),
                nn.ReLU()
                )
    self.feat_to_sigma_rgb = nn.Sequential(
                        nn.Linear(W+inp_dim,4)
                        )
    self.feat_to_sigma = nn.Sequential(
                        nn.Linear(W,1),
                        nn.ReLU())
    
  def forward(self, x, d):
    h = self.enc(x)
    feat = self.dec(torch.concat([x,h],dim=-1))
    obj = self.feat_to_sigma_rgb(torch.concat([feat,d], dim=-1))
    return obj

class NeRF_MLP_autoencoder(nn.Module):
  def __init__(self, W: int, inp_dim: int):
    super().__init__()
    self.enc = nn.Sequential(
                nn.Linear(inp_dim,512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,100),
                nn.ReLU()
                )
    self.dec = nn.Sequential(
                nn.Linear(inp_dim+100,128),
                nn.ReLU(),
                nn.Linear(128,256),
                nn.ReLU(),
                nn.Linear(256,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU()
                )
    self.feat_to_sigma_rgb = nn.Sequential(
                        nn.Linear(W+inp_dim,4)
                        )
    self.feat_to_sigma = nn.Sequential(
                        nn.Linear(W,1),
                        nn.ReLU())
    
  def forward(self, x, d):
    h = self.enc(x)
    feat = self.dec(torch.concat([x,h],dim=-1))
    obj = self.feat_to_sigma_rgb(torch.concat([feat,d], dim=-1))
    return obj
