from os.path import join

# ------------- Create Folders -------------------

def create_folders(working_dir, output_dir, overwrite_df):
  import pandas as pd

  mat_foldname = 'matrices'
  mat_foldpath = join(output_dir,mat_foldname)
  df_filename = 'dataframe.csv'
  df_filepath = join(output_dir,df_filename)
  for f in [working_dir, output_dir]:
    if not os.path.isdir(f):
      os.mkdir(f)
      print("Creating... %s" % f)
    else:
      print("Already exists: %s" % f)
  if not os.path.isdir(mat_foldpath):
    os.mkdir(mat_foldpath)
  d = df_filepath
  c = ['reg_type','subject','f_pathname','px','py','pz','rot','length','comp_length','vox_dim','mse','rmse','similarity']
  if overwrite_df:
    pd.DataFrame(columns=c).to_csv(d)
    print("Overwriting... %s" % d.split('/')[-1]) 
  if not os.path.isfile(d):
    pd.DataFrame(columns=c).to_csv(d)
    print("Initialising... %s" % d.split('/')[-1])
  else:
    print("Already exists: %s" % d)

# ------------- Generate Transformation Matrices -------------------

def trig(angle):
  from math import cos, sin, radians
  r = radians(angle)
  return cos(r), sin(r)

def point(v=(0,0,0)):
  p = [0,0,0]
  point = (0,0,0) # (1,1,1)
  for r in range(3):
    p[r] += v[r][3]
    for c in range(3):
      p[r] += point[c] * v[r][c]
  return p

def matrix(rotation=(0,0,0), translation=(0,0,0)):
  Cx, Sx = trig(rotation[0])
  Cy, Sy = trig(rotation[1])
  Cz, Sz = trig(rotation[2])
  Tx = translation[0]
  Ty = translation[1]
  Tz = translation[2]
  T = np.array([[1, 0, 0, Tx],
                  [0, 1, 0, Ty],
                  [0, 0, 1, Tz],
                  [0, 0, 0, 1]])
  Rx = np.array([[1, 0, 0, 0],
                  [0, Cx, Sx, 0],
                  [0, -Sx, Cx, 0],
                  [0, 0, 0, 1]])
  Ry = np.array([[Cy, 0, -Sy, 0],
                  [0, 1, 0, 0],
                  [Sy, 0, Cy, 0],
                  [0, 0, 0, 1]])
  Rz = np.array([[Cz, Sz, 0, 0],
                  [-Sz, Cz, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
  return np.dot(Rz, np.dot(Ry, np.dot(Rx,T)))

def randomise_transformations(t_min, t_max, t_step, r_min, r_max, r_step): # removed z
  # Randomised vectors (i.e. x, y, z) with fixed lengths in range [r_min, r_max].
  from math import pow, sqrt
  from random import uniform 
  t_samps = (t_min-t_max)/t_step
  r_samps = (r_max-r_min)/r_step
  p = [[0,0,0,0]] # removed z
  l = [x for x in np.linspace(t_min,t_max,n_samps+1) if x > 0]
  r = [x for x in np.linspace(r_min,r_max,n_samps_r+1)]
  for i in l:
    z = uniform(0,i)
    y = uniform(0,i)
    x = sqrt(abs(pow(i,2)-pow(y,2)-pow(z,2)))
    p.append([x,y,z,i])
  return p, r

def matrix_lst(Rx, Ry, Rz, Tx, Ty, Tz):
  from itertools import product
  rot_lst = (list(product(Rx, Ry, Rz, (0,), (0,), (0,))))
  trl_lst = (list(product((0,), (0,), (0,), Tx, Ty, Tz)))
  com_lst = (list(product(Rx, Ry, Rz, Tx, Ty, Tz)))
  return rot_lst, trl_lst, com_lst

def generate_matfiles(outpath, t_min, t_max, t_step, r_min, r_max, r_step, prefix='matrix',  overwrite=None):
  from itertools import product
  trans, rot = randomise_transformations(t_min=t_min, t_max=t_max, t_step=t_step, r_min=r_min, r_max=r_max, r_step=r_step)
  for c in product(trans, rot):
    f_name = prefix + '_' + ''.join(str(float(c[0][3]))) + '_' + str(float(c[1])) + '.mat' # why ''.join??
    f_name.replace('.','')
    f = join(outpath, f_name)
    m = matrix((0,c[1],0), c[0]) # rotation only in y
    if not os.path.isfile(f):
      np.savetxt(f, m, fmt='%f')
    else:
      if overwrite is not None:
        np.savetxt(f, m, fmt='%f')
      else:
        print("Checking \'%s\'..." % f_name)

def list_of_matfiles(outpath):
  l, n = [], []
  if os.path.isdir(outpath):
    for f in os.listdir(outpath):
        if f.endswith('.mat'):
            l.append(str(join(outpath,f)))
            n.append(str(f[:-4]))
  return l

# ------------- Generate Iterable Lists -------------------

def vox_lst(min_dim=0.5,max_dim=2.0,step=0.1):
  vox_lst = []
  n_samps = (max_dim-min_dim)/step
  for x in list(np.linspace(min_dim,max_dim,n_samps+1)):
    vox_lst.append(tuple([x,x,x]))
  return vox_lst

def var_lst(min_var=0.001, max_var=0.01, samples=10):
  import numpy as np
  return list(np.linspace(min_var, max_var, samples))

# ------------- Compute Output Values -------------------

def compute_pixel_sim(img1, img2):
  import numpy as np
  err = np.sum(np.absolute(img1 - img2))
  err /= float(img2.shape[0] * img2.shape[1] * img2.shape[2])
  return err

def vec(in_mat, out_mat):
  from math import pow, sqrt
  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat) # Variable 'b' not used here.
  p, point = [0,0,0], (0,0,0)
  for r in range(3):
    p[r] += a[r][3]
    for c in range(3):
      p[r] += point[c] * a[r][c]
  px, py, pz = p[0], p[1], p[2]
  l = sqrt(pow(p[0],2) + pow(p[1],2) + pow(p[2],2))
  return px, py, pz, l

def compute_len(in_mat, out_mat):
  import numpy as np
  from math import pow, sqrt
  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat)
  p0, point = [0,0,0], (0,0,0) # (1,1,1) # from utils import vector
  for r in range(3):
    p0[r] += a[r][3]
    for c in range(3):
      p0[r] += point[c] * a[r][c]
  p = p0
  l = round(sqrt(pow(p[0],2)+pow(p[1],2)+pow(p[2],2)),1)
  px, py, pz = round(p[0],2), round(p[1],2), round(p[2],2)
  return px, py, pz, l

def compute_mse(img1, img2):
  import numpy as np
  err = np.sum((img1 - img2) ** 2)
  err /= float(img2.shape[0] * img2.shape[1] * img2.shape[2])
  return err

def compute_rmse(in_mat, out_mat):
  import numpy as np
  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat)
  rmse = np.sqrt(((np.matmul(a,b)-np.identity(4))**2).mean())
  return str(round(rmse, 6))

# ------------- Compute Output Values -------------------

def big_panda(base_dir, output_dir, overwrite_df, list_of_folders=[]):
  # Expected that dataframe.csv is located in general output directory. 
  import numpy as np
  import pandas as pd
  from os import listdir
  from math import pow, sqrt
  import nibabel as nib

  list_of_folders = ['masked/coreg_fsl']
  list_of_subfolders = ['resolution']
  out_path = join(base_dir,output_dir)
  df_path = join(out_path,'dataframe_masked.csv')
  cols = ['reg_type','subject','f_pathname','px','py','pz','rot','length','comp_length','vox_dim','mse','rmse','similarity']

  df_out = pd.DataFrame(columns=cols)

  for folder in list_of_folders:
    folder_path = join(out_path,folder)
    for subfold in list_of_subfolders:
      subfold_path = join(folder_path,subfold)
      for subj in listdir(subfold_path):
        subj_path = join(subfold_path,subj)
        raw_subj_path = join(base_dir,'data',subj,'anat.nii')
        for mat in listdir(subj_path):
          mat_path = join(subj_path,mat)
          mat_len = mat.split('_')[1]
          mat_rot = mat.split('_')[2]
          for vox in listdir(mat_path):
            vox_path = join(mat_path,vox)
            if os.path.isdir(vox_path):
              vox_size = ((vox.split('_')[-1]))[:3]
              in_mat, out_mat = None, None
              for ff in listdir(vox_path):
                print(ff)
                in_mat = join(out_path,'matrices'+'/'+mat+'.mat')
                print(in_mat)
                if ff.endswith('.nii'):
                  output_im = join(vox_path,ff)
                if ff.endswith('.mat'):
                  out_mat = join(vox_path,ff)
                  print(out_mat)
              if os.path.isfile(in_mat) and os.path.isfile(out_mat):
                data_raw = nib.load(raw_subj_path).get_data().astype('float')
                data_out = nib.load(output_im).get_data().astype('float')
                mse = compute_mse(data_raw, data_out)
                sim = compute_pixel_sim(data_raw, data_out)
                rmse_val = compute_rmse(in_mat,out_mat)
                px,py,pz,l = compute_len(in_mat,out_mat)
                df_add = pd.DataFrame([[folder,subj,output_im,px,py,pz,mat_rot,mat_len,l,vox_size,mse,rmse_val,sim]], columns=cols)
                print (df_add)
                df_out = df_out.append(df_add, ignore_index=True)
                del out_mat, output_im
  df_out.to_csv(df_path)