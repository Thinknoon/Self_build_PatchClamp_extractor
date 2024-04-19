from fp_extractor_visuialization import plot_w_style
from fp_extractor_visuialization import build_fp_data
import glob
import os
from pynwb import NWBHDF5IO
import pandas as pd
from multiprocessing.pool import Pool
from multiprocessing import Process

def check_cell_info(nwb_path):
    file_name = os.path.split(nwb_path)[1].split('.')[0]
    if os.path.exists(f'check_fig/{subset}/{file_name}.png'):
        print('sample {file_name} already checked, moving to next...')
        return 1
    pass_check_cell_meta.loc[file_name,'file_path'] = i
    io_ = NWBHDF5IO(nwb_path, 'r', load_namespaces=True)
    nwb = io_.read()
    try:
        check_info = build_fp_data(nwb)
    except:
        pass_check_cell_meta.loc[file_name,'pass_check']='build_fp_ERROR'
        return 1
    if isinstance(check_info, int):
        print('goto next cell')
        pass_check_cell_meta.loc[file_name,'pass_check']='Not FP'
    else:
        try:
            axs,fig = plot_w_style(nwb)
            fig.savefig(f'check_fig/{subset}/{file_name}.png')
        except:
            pass_check_cell_meta.loc[file_name,'pass_check']='N'
        else:
            pass_check_cell_meta.loc[file_name,'pass_check']='Y'

if __name__ == '__main__':
    p = Pool(10)
    subset='HAKA'
    all_files = glob.glob(f'LIP_{subset}/*/*.nwb')
    if os.path.exists(f'LIP_{subset}/cell_metadata.csv'):
        pass_check_cell_meta = pd.read_csv(f'LIP_{subset}/cell_metadata.csv')
    else:
        pass_check_cell_meta = pd.DataFrame(index=[os.path.split(i)[1].split('.')[0] for i in all_files],columns=['file_path','pass_check'])
    for i in all_files:
        p.apply_async(check_cell_info,args=(i,))
    p.close()
    p.join()
    pass_check_cell_meta.to_csv(f'LIP_{subset}/cell_metadata.csv')