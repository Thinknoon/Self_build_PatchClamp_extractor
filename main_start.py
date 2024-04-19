from DatConverter_eachseries import DatConverter_each_series

import glob
import os

if __name__ == '__main__':
    all_axon_files = glob.glob('j:/HuangMY/LIP/ephys/LIP_HAKA/*/*.dat')
    for i in all_axon_files:
        file_name = os.path.split(i)[1].split('.')[0]
        if os.path.exists('LIP_HAKA/' + file_name):
            print(f'{file_name} already exists... moving to next directory')
            continue
        else:
            os.mkdir('LIP_HAKA/' + file_name)
            DatConverter_each_series(i,f'LIP_HAKA/{file_name}/{file_name}.nwb')