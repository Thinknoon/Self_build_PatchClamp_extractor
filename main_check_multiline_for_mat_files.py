from fp_extractor_visuialization import plot_w_style
import glob
import os
import pandas as pd
from multiprocessing.pool import Pool
import scipy.io
import numpy as np

def check_cell_info(mat_filepath):
    file_name = os.path.split(mat_filepath)[1].split('.')[0]
    if os.path.exists(f'check_fig/{file_name}.png'):
        print('sample {file_name} already checked, moving to next...')
        return 1
    pass_check_cell_meta = pd.DataFrame(index=[os.path.split(mat_filepath)[1].split('.')[0]],columns=['file_path','pass_check'])
    temp_mat_load = scipy.io.loadmat(mat_filepath)
    voltage = temp_mat_load['m_FP']['alldt'][0,0][0,0]
    current = temp_mat_load['m_FP']['StiStep'][0,0][0]
    if len(current)!=voltage.shape[1]:
        print('Attention!!! current length is not equal to voltage')
        current = current[:voltage.shape[1]]
    current_step =  temp_mat_load['m_FP']['stepCurr'][0,0][0,0]
    time = np.arange(25000)/25000
    curr_index_0 = temp_mat_load['m_FP']['curr_index_0'][0,0][0,0]
    pass_check_cell_meta.loc[file_name,'file_path']=mat_filepath
    try:
        axs,fig = plot_w_style(time=time,voltage=voltage,current=current,curr_index_0=curr_index_0,current_step=current_step)
        fig.savefig(f'check_fig/{file_name}.png')
        pass_check_cell_meta.loc[file_name,'pass_check']='Y'
    except:
        print('***************')
        print(f'sample {file_name} plot_error')
        print('***************')
        pass_check_cell_meta.loc[file_name,'pass_check']='N'
    pass_check_cell_meta.to_csv(f'./temp/{file_name}.csv')

if __name__ == '__main__':
    p = Pool(15)
    all_files = glob.glob(f'../LIP/ephys/matlab_ephys/*.mat')
    for i in all_files:
        p.apply_async(check_cell_info,args=(i,))
    p.close()
    p.join()  
    subset_features_files = glob.glob('./temp/*.csv')
    features = pd.DataFrame()
    for j in subset_features_files:
        current_features = pd.read_csv(j,index_col=0)
        features = pd.concat([features,current_features])
    features.to_csv(f'all_cellCheck_metadata.csv') 