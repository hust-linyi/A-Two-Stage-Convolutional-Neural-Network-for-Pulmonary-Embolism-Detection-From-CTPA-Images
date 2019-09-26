import numpy as np
class AffineTransformation_3d(object):
	"""
	For this class, x,w represent a img's width when looking at.
					y,h represent a img's height when looking at.
	
	All img's intensity's range is [0,255]

	"""
	def __init__(self,whole_img_3d,center,to_size=(15,15,15)):
		"""
		whole_img_3d : A 3d numpy, from which one can do crop
		crop_x_y : A 1-D list with 6 elements, which represent unnormalized
				   x_min, x_max, y_min, y_max, z_min and z_max respectively. 
		           Can be with a different shape from to_size.
		to_size : A tuple of target image size , default = (15,15,15)
		crop : A sub_img of proposal, created by crop_x_y with to_size
		"""
		self.center = center 
		self.img = whole_img_3d
		self.to_size = to_size
		self.crop = None
		self.empty = False

	def crop_img_and_do_pca(self,threshold=0.30257220216*255):
		"""
		Compute the feature vectors from the cropped img(proposal).
		threshold : To classify whether a pixel belongs to vessel, defualt 127

		Note that the returned centers may not be a good value.
		"""
		x_center,y_center,z_center = self.center
		x_size,y_size,z_size = self.to_size
		x_min, x_max = x_center-x_size//2,x_center+x_size//2+1
		y_min, y_max = y_center-y_size//2,y_center+y_size//2+1
		z_min, z_max = z_center-z_size//2,z_center+z_size//2+1
		self.crop = self.img[0,x_min:x_max,y_min:y_max,z_min:z_max]
		# print(self.img.shape)
		# print(self.crop.shape)
		
		# prepare x,y,z points for PCA
		mask = np.where(self.crop>threshold)
		if mask[0].shape[0] == 0:
			self.empty = True
			return None
		else:
			x = mask[0]
			y = mask[1]
			z = mask[2]
			# Normalize the points
			X = np.vstack((x,y,z)).transpose()
			mean = np.mean(X,axis=0)
			X = X - mean
			# Covariance matrix and PCA
			A = X.T.dot(X) / float(X.shape[0])
			u, s, vh = np.linalg.svd(A)

		return u

	def compute_matrix(self,center,u):
		"""
		Compute the affine transformation matrix with first scale, second rotation
		finally translation.

		input :
		x_center,y_center,z_center,u: All are outputs from the function crop_img_and_do_pca

		output :
		M : Transformation Matrix of shape (3,4)
		"""
		if self.empty:
			return None
		x_center,y_center,z_center = center
		_,Lx,Ly,Lz = self.img.shape
		tx,ty,tz = self.to_size
		offset_x = 2.0 * x_center / Lx - 1.0
		offset_y = 2.0 * y_center / Ly - 1.0
		offset_z = 2.0 * z_center / Lz - 1.0		
		x_ratio = 1.0 * tx / Lx		
		y_ratio = 1.0 * ty / Ly
		z_ratio = 1.0 * tz / Lz

		if u[2,0] < 0:
			u = -u
		R = np.hstack((u[:,2].reshape(3,1),u[:,1].reshape(3,1),u[:,0].reshape(3,1)))
		M = R.dot(np.array([[x_ratio,0,0],
							[0,y_ratio,0],
							[0,0,z_ratio]]))

		M = np.hstack((M,np.array([offset_x,offset_y,offset_z]).reshape(3,1))) 
		return M

	def affine_transform(self,M):
		"""
		Compute the output matrix cropped inside original whole 3-D image 

		input :
			An affine transformation matrix of shape (3,4)

		output :
		    A caculated 3-d image from original whole 3D image.
		"""
		if self.empty:
			return self.crop

		_,Lx,Ly,Lz = self.img.shape
		tx,ty,tz = self.to_size

		x_t_grid = np.linspace(-1, 1, tx)
		y_t_grid = np.linspace(-1, 1, ty)
		z_t_grid = np.linspace(-1, 1, tz)

		x_t, y_t, z_t = np.meshgrid(x_t_grid, y_t_grid, z_t_grid)
		ones = np.ones(np.prod(x_t.shape))
		sampling_grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])
		
		# Source_grid has shape (3, Lx*Ly*Lz)
		source_grids = np.matmul(M, sampling_grid)
		# reshape to (3, Lx, Ly, Lz)
		source_grids = source_grids.reshape(3, tx, ty, tz)

		x_s = source_grids[0]
		y_s = source_grids[1]
		z_s = source_grids[2]

		# rescale x and y to [0, Lx/Ly/Lz]
		x = ((x_s + 1.) * Lx) * 0.5
		y = ((y_s + 1.) * Ly) * 0.5
		z = ((z_s + 1.) * Lz) * 0.5

		# grab 8 nearest corner points for each (x_i, y_i, z_i)
		x0 = np.floor(x).astype(np.int64)
		x1 = x0 + 1
		y0 = np.floor(y).astype(np.int64)
		y1 = y0 + 1
		z0 = np.floor(z).astype(np.int64)
		z1 = z0 + 1

		# make sure it's inside img range [0, Lx-1] [0, Ly-1] [0, Lz-1] 
		x0 = np.clip(x0, 0, Lx-1)
		x1 = np.clip(x1, 0, Lx-1)
		y0 = np.clip(y0, 0, Ly-1)
		y1 = np.clip(y1, 0, Ly-1)
		z0 = np.clip(z0, 0, Lz-1)
		z1 = np.clip(z1, 0, Lz-1)

		# look up pixel values at corner coords
		Ia = self.img[0,x0.flatten(), y0.flatten(), z0.flatten()].reshape(tx,ty,tz)
		Ib = self.img[0,x0.flatten(), y1.flatten(), z0.flatten()].reshape(tx,ty,tz)
		Ic = self.img[0,x1.flatten(), y0.flatten(), z0.flatten()].reshape(tx,ty,tz)
		Id = self.img[0,x1.flatten(), y1.flatten(), z0.flatten()].reshape(tx,ty,tz)
		Ie = self.img[0,x0.flatten(), y0.flatten(), z1.flatten()].reshape(tx,ty,tz)
		If = self.img[0,x0.flatten(), y1.flatten(), z1.flatten()].reshape(tx,ty,tz)
		Ig = self.img[0,x1.flatten(), y0.flatten(), z1.flatten()].reshape(tx,ty,tz)
		Ih = self.img[0,x1.flatten(), y1.flatten(), z1.flatten()].reshape(tx,ty,tz)

		# calculate deltas
		wa = (x1-x) * (y1-y) * (z1 - z)
		wb = (x1-x) * (y-y0) * (z1 - z)
		wc = (x-x0) * (y1-y) * (z1 - z)
		wd = (x-x0) * (y-y0) * (z1 - z)
		we = (x1-x) * (y1-y) * (z - z0)
		wf = (x1-x) * (y-y0) * (z - z0)
		wg = (x-x0) * (y1-y) * (z - z0)
		wh = (x-x0) * (y-y0) * (z - z0)

		# compute output
		out = wa*Ia + wb*Ib + wc*Ic + wd*Id + we*Ie + wf*If + wg*Ig + wh*Ih
		out = np.clip(out, 0, 255)
		out = out.astype('uint8')
		return out






