import numpy as np
import os
from tqdm import tqdm
import sys

pfr = pd.read_excel("../S-nitrosylation/physiochemical_factors.xlsx")

pfr_num = pfr.shape[1]-1
pfr_dict = {}
for i in range(pfr.shape[0]):
    pfr_dict[pfr.iloc[i,0]] = pfr.iloc[i,1:].values

def pfr_feature(input_path,output_dir):
    
    #打开txt文件并读取内容
    with open(input_path, 'r') as f1:
        lines = f1.readlines()
        stripped_lines = [line.strip() for line in lines]
    
    # 读取fasta文件
#     file = open(input_path, "r")
#     sequences = []
#     for seq_record in SeqIO.parse(file, "fasta"):
#         sequences.append(str(seq_record.seq))
#     # 关闭文件
#     file.close()
    
    rows = len(stripped_lines)
    win_size = len(stripped_lines[0])
    pfr_matrix = np.zeros((rows,win_size*pfr_num))

    for k in range(rows):
        seq = stripped_lines[k]
        list_pfr = []
        for i in range(pfr_num):
            pfr_values = [pfr_dict[s][i] for s in seq]
            list_pfr.extend(pfr_values)
        pfr_matrix[k, :] = list_pfr
        
    if 'train' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'train_pos'+'_PFR'+'.txt',pfr_matrix,fmt='%g',delimiter=',')
        
    elif 'train' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'train_neg'+'_PFR'+'.txt',pfr_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'pos' in input_path:
        
        np.savetxt(output_dir+'test_pos'+'_PFR'+'.txt',pfr_matrix,fmt='%g',delimiter=',')
        
    elif 'test' in input_path and 'neg' in input_path:
        
        np.savetxt(output_dir+'test_neg'+'_PFR'+'.txt',pfr_matrix,fmt='%g',delimiter=',')
        
if __name__=='__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    files = os.listdir(input_dir)

    for f in tqdm(files):
        input_path = input_dir+f
        pfr_feature(input_path,output_dir)