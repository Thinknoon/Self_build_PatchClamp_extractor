from fp_extractor_visuialization import build_fp_data
import sys
from pynwb import NWBHDF5IO
import pandas as pd
from fp_extractor_visuialization import extract_spike_features
from fp_extractor_visuialization import get_cell_features
import matplotlib.pyplot as plt

axon_passed_check = pd.read_csv('./LIP_AXON/cell_metadata.csv',index_col=0)
axon_passed_check.index = "axon_"+axon_passed_check.index
axon_passed_check = axon_passed_check.loc[axon_passed_check['pass_check']=='Y',]
heka_passed_check = pd.read_csv('./LIP_HAKA/cell_metadata.csv',index_col=0)
heka_passed_check.index = "heka_"+heka_passed_check.index
heka_passed_check = heka_passed_check.loc[heka_passed_check['pass_check']=='Y',]
# 不能有同名的细胞
if sum(axon_passed_check.index.isin(heka_passed_check.index))>0:
    print('some cells have the same name,checking and correct')
    sys.exit()
all_passed_check = pd.concat([axon_passed_check,heka_passed_check])
counter=1
All_Cells_Features = pd.DataFrame()
for i in all_passed_check.index:
    print(f'processing cell {i}...{counter} of total {len(all_passed_check.index)}')
    file_path = all_passed_check.loc[i,'file_path']
    io_ = NWBHDF5IO(file_path, 'r', load_namespaces=True)
    nwb = io_.read()
    time, voltage, current, curr_index_0 = build_fp_data(nwb)
    filter_ = 10
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()
    All_Cells_Features = pd.concat([All_Cells_Features, Cell_Features], sort = True)
    counter+=1
All_Cells_Features.insert(0, 'name sample', all_passed_check.index)
All_Cells_Features.to_csv('all_cells_features.csv')
