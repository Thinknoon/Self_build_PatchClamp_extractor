from fp_extractor_visuialization import plot_w_style
from fp_extractor_visuialization import build_fp_data
import glob
import os
from pynwb import NWBHDF5IO
import pandas as pd

all_files = glob.glob('LIP_HAKA/*/*.nwb')
if os.path.exists('LIP_HAKA/cell_metadata.csv'):
    pass_check_cell_meta = pd.read_csv('LIP_HAKA/cell_metadata.csv',index_col=0)
else:
    pass_check_cell_meta = pd.DataFrame(index=[os.path.split(i)[1].split('.')[0] for i in all_files],columns=['file_path','pass_check'])
counter=0
for i in all_files:
    file_name = os.path.split(i)[1].split('.')[0]
    if os.path.exists(f'check_fig/HAKA/{file_name}.png'):
        print(f'sample {file_name} already exists, moving to next.....')
        continue
    pass_check_cell_meta.loc[file_name,'file_path'] = i
    io_ = NWBHDF5IO(i, 'r', load_namespaces=True)
    nwb = io_.read()
    try:
        check_info = build_fp_data(nwb)
    except:
        pass_check_cell_meta.loc[file_name,'pass_check']='build_fp_ERROR'
        continue
    if isinstance(check_info, int):
        print('goto next cell')
        pass_check_cell_meta.loc[file_name,'pass_check']='Not FP'
    else:
        try:
            axs,fig = plot_w_style(nwb)
            fig.savefig(f'check_fig/HAKA/{file_name}.png')
        except:
            pass_check_cell_meta.loc[file_name,'pass_check']='N'
        else:
            pass_check_cell_meta.loc[file_name,'pass_check']='Y'
    counter+=1
    if counter>=20:
        pass_check_cell_meta.to_csv('LIP_HAKA/cell_metadata.csv')
        counter=0
pass_check_cell_meta.to_csv('LIP_HAKA/cell_metadata.csv')