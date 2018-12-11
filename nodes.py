def snr_img(base_dir, output_dir, in_file, var):
  import os
  import numpy as np
  import nibabel as nib
  from os.path import join

  copy_folderpath = join(base_dir, output_dir, 'noise_copy')
  if not os.path.isdir(copy_folderpath):
    os.makedirs(copy_folderpath)
  mean = 0
  img = nib.load(in_file)
  dat_orig = img.get_data()
  # filename = img.get_filename()
  # mx_name = in_file.split('/')[-3][-19:-4]
  dat = dat_orig.copy()
  noise = np.random.normal(mean, var**0.5, dat.shape)
  dat_noise = dat + np.max(dat) * noise
  dat_clipped = np.clip(dat_noise, a_min=np.min(dat), a_max=np.max(dat))
  new_img = nib.Nifti1Image(dat_clipped, img.affine, img.header)
  new_filename = join(copy_folderpath, 'anat_flirt_'+'_'+str(var)+'_noised.nii')
  nib.save(new_img, new_filename)

  return new_filename

def compute_rmse(in_mat, out_mat):
  import os
  import numpy as np

  filename = 'rmse.txt'
  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat)
  with open(filename, 'w') as f:
    f.write(str(np.sqrt(((np.matmul(a,b)-np.identity(4))**2).mean())))

  return os.path.abspath(filename)

def compute_vals(in_mat, out_mat):
  import numpy as np
  from utils import point

  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat)
  rmse_val = str(np.sqrt(((np.matmul(a,b)-np.identity(4))**2).mean()))

  return point(a), point(b), rmse_val

def compute(outpath, subject_id, in_mat, out_mat): # overwrite_dat=OVERWRITE_DF

  import os
  import numpy as np
  import pandas as pd
  from math import pow, sqrt

  f = os.path.join(outpath,'dataframe.csv')
  cols = ['subject','transform_fname','px','py','pz','length','rmse']
  df_out = pd.read_csv(f, index_col=0)
  matrix_id = in_mat.split("/")[-1]

  a, b = np.loadtxt(in_mat), np.loadtxt(out_mat)
  rmse_val = str(round(np.sqrt(((np.matmul(a,b)-np.identity(4))**2).mean()),6))
  p0, point = [0,0,0], (0,0,0) # (1,1,1) # from utils import point ** small problem.s
  for r in range(3):
    p0[r] += a[r][3]
    for c in range(3):
      p0[r] += point[c] * a[r][c]
  p = p0
  l = sqrt(pow(p[0],2)+pow(p[1],2)+pow(p[2],2))
  px, py, pz = p[0], p[1], p[2]
  
  df_add = pd.DataFrame([[subject_id,matrix_id,px,py,pz,l,rmse_val]], columns=cols)
  df_out = df_out.append(df_add, ignore_index=True)
  df_out.to_csv(f)
  print (df_out)

def dataframe(df_path, subject_id, subject_template, in_mat, out_mat): # overwrite_dat=OVERWRITE_DF
  import os
  import pandas as pd
  from utils import compute_rmse, vec

  if not os.path.isfile(df_path):
    e = 'File %s does not exist. Please re-run initialisation script.' % df_path
    return print (e)
  else:
    df = pd.read_csv(df_path, index_col=0)
  cols_h = ['subject','template','matfile_name','px','py','pz','translation','rotation','total_length','resolution','noise','rmse']
  cols_r = list(df)
  if not cols_h != cols_r:
    e = 'Column names do not match. Please check.'
    return print (e) 
  mat_id = in_mat.split('/')[-1] # px, py, pz = v[0], v[1], v[2], # rx, ry, rz = xx, # l = v[3]
  t, r = mat_id.split('_')[0], mat_id.split('_')[1]
  vox_dim = '1' # find params of functions, to get vox_dim ## HERE
  rmse = compute_rmse
  df_tmp = pd.DataFrame([[subject_id,mat_id,vec[0],vec[1],vec[2],vec[3],t,r,vox_dim,rmse]], columns=cols_r)
  df = df.append(df_tmp, ignore_index=True)
  df.to_csv(df_path)