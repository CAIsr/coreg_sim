import os
from os.path import join

from nipype import config, logging
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.ants import Registration, RegistrationSynQuick
from nipype.interfaces.spm import Smooth, Coregister, ApplyTransform
from nipype.interfaces.fsl import (BET, ExtractROI, FAST, FLIRT, ImageMaths,
                                   MCFLIRT, SliceTimer, Threshold, ApplyXFM, FNIRT)      
from nipype.interfaces.fsl.maths import Threshold                                                     
from nipype.interfaces.afni import Resample
from nipype.interfaces.spm.utils import ResliceToReference
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms.misc import Gunzip
from nipype.pipeline.engine import Workflow, Node, MapNode

from utils import *
from nodes import *

# --------------- INIT --------------------
OVERWRITE = True
OVERWRITE_DF = False

# Directories
base_dir = '/data/fastertemp/uqdeden1/Nextcloud/COREG2018-Q0585/project'
data_dir = join(base_dir,'data')
working_dir = join(base_dir,'workingdir')
output_dir = join(base_dir,'outputdir')
# log_dir = join(base_dir,'logdir')
outpath_mat = join(base_dir,output_dir,'matrices','')
outpath_df = join(base_dir,output_dir)

# Subject List
subjects =  ['subject_1', 'subject_2' , 'subject_3', 'subject_4']
templates = {'anat': data_dir+'/{subject_id}/anat.nii', # {'label': path_to_data/image.nii'}
             'mask': data_dir+'/{subject_id}/mask.nii',
             'slab': data_dir+'/{subject_id}/slab.nii'}

# Sampling Values
t_min, t_max, t_step = 0, 50, 10
r_min, r_max, r_step = 0, 15, 15
min_dim, max_dim, step_dim = 0.5, 4.0, 0.5
min_var, max_var, samp_var = 0.001, 0.01, 10

# Node Variables
smoothing_size = 8

# ------------ WORKFLOW ------------------
# Config # config['execution']['crashdump_dir'] = base_dir
# config.update_config({'logging': {'log_directory': log_dir,
#                                   'log_to_file': False}}) 
# logging.update_logging(config)

# Infosource
infosource = Node(IdentityInterface(fields=['subject_id']),
                    name="infosource")
infosource.iterables = [('subject_id', subjects)]

# SelectFiles
selectfiles = Node(SelectFiles(templates,
                                base_directory=base_dir),
                    name="selectfiles")

# DataSink
datasink = Node(DataSink(),
                  name="datasink") # To differentiate output files from temporary files.
datasink.inputs.base_directory = base_dir
datasink.inputs.container = output_dir

# DataSink Substitutions
substitutions = [('_subject_id', ''), ('_subject','subject'), # Substitutions have an order!!!
                  ('_in_matrix_file_', ''), # Turning a node into an iterable defaults to naming the output file to input variable name
                  ('%s' % outpath_mat.replace('/','..'), ''), # When creating own nodes, output files are (by default) named according to the absolute path, however Nipype replaces '/' with '..' to avoid creating directories. It's ugly. This is my 'make-do' workaround.
                  ('.mat', ''),
                  ('_voxel_size_','vox_dims_'), 
                  ('anat_flirt','anat_tform'), # Output files are (by default) named according to the input function or variable name. Each additional node will add '_functionname'. Be careful when creating own nodes. Nipype gets confused. Overwritten or misrecognised in 'processing' folder.
                  ('anat_resample','anat_rsmpl'),
                  ('_var_','var_')
                  ]
datasink.inputs.substitutions = substitutions

# Smooth
smooth = Node(ImageMaths(op_string='-fmean -s 2'), 
                name="smooth")
smooth_spm = Node(Smooth(fwhm=smoothing_size),
                    name="smooth")

# Resample
resample = Node(Resample(outputtype='NIFTI',
                          resample_mode='Li'), 
                  name="resample")
resample.iterables = ('voxel_size', vox_lst(min_dim,max_dim,step_dim)) # Example of turning a node (function) into an iterable, e.g. nodename.iterables = ('parameter',list_of_iterables). Depending on parameter, list may need to be tuple, etc.

# Noise
noise = Node(interface=Function(input_names=['base_dir','output_dir','in_file','var'], # Self-created node. Needs improvement.
                                output_names=['out_file'], 
                                function=snr_img), 
              name="noise")
noise.iterables = ('var', var_lst(min_var,max_var,samp_var)) # List of values (iterables).
noise.inputs.base_dir = base_dir
noise.inputs.output_dir = output_dir

# ApplyXFM
transform = Node(ApplyXFM(output_type='NIFTI', 
                          apply_xfm=True), 
                    name="transform")                          
transform.iterables = ('in_matrix_file', list_of_matfiles(outpath_mat)) # Example of turning a node (function) into an iterable.

# TransformMask
transform_mask = Node(ApplyXFM(output_type='NIFTI', 
                                apply_xfm=True), 
                        name="transform_mask") 

# Co-Registration
coreg_spm = Node(Coregister(jobtype='estimate'), 
                    name="coreg_spm")

coreg_fsl = Node(FLIRT(cost_func='bbr',
                        output_type='NIFTI', 
                        uses_qform=True), 
                    name="coreg_fsl") # iterfield=['in_file']) # Example of turning a variable into an iterfield.

FNIRT_anat2func = Node(FNIRT(cost_func='bbr',
                            output_type='NIFTI', 
                            uses_qform=True), 
                        name="FLIRT_anat2func")

FNIRT_func2anat = Node(FLIRT(cost_func='bbr', 
                            output_type='NIFTI', 
                            uses_qform=True), 
                        name="FNIRT_func2anat")

coreg_ANTs_SyNQuick = Node(RegistrationSynQuick(num_threads=2, 
                                                transform_type='a'), 
                            name="coreg_ANTs_SyNQuick") # Spelling error in documentation.

# Gunzip Files
gunzip = MapNode(Gunzip(), 
                name="gzip", 
                iterfield=['in_file'])

# ComputeRMSE
rmse = Node(interface=Function(input_names=['in_mat','out_mat'], 
                                output_names=['out_file'],
                                function=compute_rmse), 
              name="rmse")

# ComputeVals
vals = Node(interface=Function(input_names=['in_mat','out_mat'], 
                                output_names=['vector_in','vector_out','rmse_val'],
                                function=compute_vals), 
              name="vals")

# ComputeVals&DataFrame
compute = Node(interface=Function(input_names=['outpath','subject_id','in_mat','out_mat'], 
                                    output_names=[],
                                    function=compute), 
                name="compute")
compute.inputs.outpath = outpath_df
 
# Threshold
threshold = Node(ImageMaths(output_type='NIFTI', 
                            op_string='-nan -thr 0.5 -bin'), 
                name="threshold")  

# Masked Workflow
masked = Workflow(name="masked")
masked.base_dir = working_dir
masked.connect([
                # Select Files
                (infosource, selectfiles, [('subject_id','subject_id')]),

                # Transform
                (selectfiles, transform, [('mask','in_file'), 
                                          ('mask','reference')]),
                (transform, datasink, [('out_file','mask.1_transformed')]),

                # Resample
                (transform, resample, [('out_file','in_file')]), 
                (resample, datasink, [('out_file','mask.2_resampled')]),

                # Co-registration w/ FSL (anat to func)
                (resample, FLIRT_anat2func, [('out_file','reference')]),
                (selectfiles, FNIRT_anat2func, [('anat','in_file')]),
                (FNIRT_anat2func, datasink, [('out_file','mask.3_coreg_anat2func')]),

                # Coregistration w/ FSL (func to anat)
                (resample, coreg_FNIRT_func2anat, [('out_file','in_file')]),
                (selectfiles, coreg_FNIRT_func2anat, [('anat','reference')]),
                (coreg_FNIRT_func2anat, datasink, [('out_file','mask.3_coreg_func2anat.FNIRT')]),

                # Coregistration w/ ANTs_SyNQuick
                (resample, coreg_ANTs_SyNQuick, [('out_file','moving_image')]),
                (selectfiles, coreg_ANTs_SyNQuick, [('anat','fixed_image')]),
                (coreg_ANTs_SyNQuick, datasink, [('warped_image','mask.3_coreg_func2anat.SyNQuick'),
                                                  ('out_matrix', 'mask.3_coreg_func2anat.SyNQuick.@mat')])             
                ])

if __name__ == "__main__":
  # Create Folders
  create_folders(working_dir, output_dir, overwrite_df=OVERWRITE_DF)

  # Generate Transformation Matrices, output as .mat
  generate_matfiles(outpath_mat, t_min, t_max, t_step, r_min, r_max, r_step, overwrite=None)

  # Run Workflow
  masked.run('MultiProc', plugin_args={'n_procs':50})

  # Pandas
  # big_panda(base_dir, output_dir, overwrite_df=OVERWRITE_DF, list_of_folders=['coreg_fsl'])