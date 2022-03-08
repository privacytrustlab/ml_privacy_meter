





import numpy as np
import pandas as pd
import csv
import time
import argparse
from collections import defaultdict
from sklearn import metrics

parser = argparse.ArgumentParser(description="Sample extraction")




parser.add_argument("--dir_name_in", type=str, help="dir", default='./batched')
parser.add_argument("--dir_name_out", type=str, help="dir", default='./batched')


parser.add_argument("--dir_name_ref_in", type=str, help="dir", default='./batched')
parser.add_argument("--dir_name_ref_out", type=str, help="dir", default='./batched')



parser.add_argument("--dir_name_train_in", type=str, help="dir")
parser.add_argument("--dir_name_train_out", type=str, help="dir")
parser.add_argument("--dir_name_population_in", type=str, help="dir")
parser.add_argument("--dir_name_population_out", type=str, help="dir")




parser.add_argument("--file_name_in", type=str, help="dir", default='./batched')
parser.add_argument("--file_name_ref", type=str, help="dir", default='./batched')


parser.add_argument("--file_name_train", type=str, help="dir", default='./batched')
parser.add_argument("--file_name_population", type=str, help="dir", default='./batched')

parser.add_argument("--root_path", type=str, help="dir", default='./loss_values/')


args = parser.parse_args()






def load_loss_vals(file, file_name):
    loss_vals = []
    with open(file+'/loss.txt','r') as loss_file, open(file_name,'r') as input_csv_file:
        csv_reader = csv.reader(input_csv_file)
        header = next(csv_reader)
        temp_sum = 0
        temp_cnt = 0
        prev_sub_id = -100
        for line, csv_line in zip(loss_file,csv_reader):
            if csv_line[0] !=  prev_sub_id:
                if temp_cnt != 0:
                    loss_vals.append(temp_sum/temp_cnt)
                else:
                    loss_vals.append(0)
                
                prev_sub_id = csv_line[0]
                temp_cnt = 0
                temp_sum = 0
                
            
            temp_sum += float(line[:-1])
            temp_cnt += 1
            
            
    
        
    return loss_vals[1:]




def get_loss_diff_auc(main_in_list, main_out_list, ref_in_list,ref_out_list):
    
    diff_in = [(s_m-s_ref) for s_m,s_ref in zip(main_in_list,ref_in_list)]
    diff_out = [(s_m-s_ref) for s_m,s_ref in zip(main_out_list,ref_out_list)]
    
    y_test_in = [1]*len(diff_in)
    y_test_out = [0]*len(diff_out)
    y_test = y_test_in + y_test_out

    y_test_np = np.array(y_test)

    diff_list = diff_in + diff_out
    diff_np = np.array(diff_list)

    max_diff = np.max(diff_np)
    min_diff = np.min(diff_np)

    diff_normalized = (max_diff -diff_np)/(max_diff-min_diff)

    auc = metrics.roc_auc_score(y_test_np, diff_normalized)
    
    return auc
    
    


def get_loss_avg_auc(main_in_list, main_out_list):
    

    y_test_in = [1]*len(main_in_list)
    y_test_out = [0]*len(main_out_list)
    y_test = y_test_in + y_test_out

    y_test_np = np.array(y_test)

    diff_list = main_in_list + main_out_list
    diff_np = np.array(diff_list)

    max_diff = np.max(diff_np)
    min_diff = np.min(diff_np)

    diff_normalized = (max_diff -diff_np)/(max_diff-min_diff)

    auc = metrics.roc_auc_score(y_test_np, diff_normalized)
    #print(auc, '    ',len(main_in_list),'   ',len(main_out_list))

    return auc, len(main_in_list),len(main_out_list)


def get_threshold_precision_diff_tail_acc(main_in_list, main_out_list, ref_in_list,ref_out_list,population_out_list=None, population_in_list=None):
    
    
    diff_in = [(s_m-s_ref) for s_m,s_ref in zip(main_in_list,ref_in_list)]
    diff_out = [(s_m-s_ref) for s_m,s_ref in zip(main_out_list,ref_out_list)]
    
    
    
        
    
    if population_in_list is not None and population_out_list is not None:
        diff_population = [(s_m-s_ref) for s_m,s_ref in zip(population_in_list,population_out_list)]
        #diff_population.sort(reverse=True)   
        diff_pop_sorted = sorted(diff_population,reverse=True) 
        threshold = diff_pop_sorted[int(len(diff_pop_sorted)*0.9)+1]#-150
        population_in_list.sort(reverse=True)
        threshold_pop = population_in_list[int(len(population_in_list)*0.9)+1]#-150
    else:
        diff_out_sorted = sorted(diff_out, reverse=True)
        threshold = diff_out_sorted[int(len(diff_out_sorted)*0.9)+1]#-150
        main_out_list_sorted = sorted(main_out_list, reverse=True)
        threshold_pop=  main_out_list_sorted[int(len(main_out_list_sorted)*0.9)+1]#-150
        
    
        
    corr_in = lambda a,th : sum([1 for sample in a if sample<th ])
    

    cnt_corr_in = corr_in(diff_in,threshold)
    cnt_corr_in_pop = corr_in(main_in_list,threshold_pop)
    cnt_wrong_out = corr_in(diff_out, threshold)
    cnt_wrong_out_pop=corr_in(main_out_list,threshold_pop)

    #print(cnt_corr_in,cnt_wrong_out)
    #precision, accuracy, threshold, 
    
    return cnt_corr_in/(cnt_corr_in+cnt_wrong_out), (cnt_corr_in+len(diff_out)-cnt_wrong_out)/(len(diff_out)+len(diff_in)),cnt_corr_in/(len((main_in_list))) ,     cnt_corr_in_pop/(cnt_corr_in_pop+cnt_wrong_out_pop), (cnt_corr_in_pop+len(main_out_list)-cnt_wrong_out_pop)/(len(main_out_list)+len(main_in_list)),cnt_corr_in_pop/(len((main_in_list))) ,threshold, threshold_pop,cnt_corr_in, cnt_wrong_out


def get_threshold_precision_avg_acc(main_in_list, main_out_list, training_in_list=None, training_out_list=None):
    
    
    if training_in_list is not None and training_out_list is not None: 
        threshold = sum(training_in_list)/len(training_in_list)
    else:    
        threshold = sum(main_in_list)/len(main_in_list)
    corr_in = lambda a,th : sum([1 for sample in a if sample<th ])

    cnt_corr_in = corr_in(main_in_list,threshold)
    cnt_wrong_out = corr_in(main_out_list, threshold)

    #print(cnt_corr_in,cnt_wrong_out)
    #precision, accuracy, threshold, 
    
    return cnt_corr_in/(cnt_corr_in+cnt_wrong_out), (cnt_corr_in+len(main_out_list)-cnt_wrong_out)/(len(main_out_list)+len(main_in_list)), cnt_corr_in/(len(main_in_list)),threshold, cnt_corr_in, cnt_wrong_out











root_path = args.root_path
loss_root= root_path


path_main_in=args.dir_name_in
path_main_out = args.dir_name_out

path_ref_in = args.dir_name_ref_in
path_ref_out = args.dir_name_ref_out




if args.dir_name_train_in is not None and args.dir_name_train_out is not None:
    
    loss_train_in = load_loss_vals(loss_root+'/'+args.dir_name_train_in, args.file_name_train)
    loss_train_out = load_loss_vals(loss_root+'/'+args.dir_name_train_out, args.file_name_train)
else:
    loss_train_in = None
    loss_train_out = None
    

if args.dir_name_population_in is not None and args.dir_name_population_out is not None:
    
    loss_population_in = load_loss_vals(loss_root+'/'+args.dir_name_population_in,args.file_name_population)
    loss_population_out = load_loss_vals(loss_root+'/'+args.dir_name_population_out, args.file_name_population)

else: 
    
    loss_population_in = None
    loss_population_out = None

loss_main_in = load_loss_vals(loss_root+'/'+path_main_in, file_name=args.file_name_in)
loss_main_out = load_loss_vals(loss_root+'/'+path_main_out, file_name=args.file_name_ref)

loss_ref_in = load_loss_vals(loss_root+'/'+path_ref_in, file_name=args.file_name_in)
loss_ref_out =  load_loss_vals(loss_root+'/'+path_ref_out, file_name=args.file_name_ref)


avg_auc , len_in, len_out= get_loss_avg_auc(loss_main_in, loss_main_out)
diff_auc = get_loss_diff_auc(loss_main_in, loss_main_out, loss_ref_in,loss_ref_out)


precision,acc,recall,precision_pop,acc_pop,recall_pop,threshold,threshold_pop,cnt_corr_in,cnt_wrong_out = get_threshold_precision_diff_tail_acc(loss_main_in, loss_main_out, loss_ref_in,loss_ref_out, population_in_list= loss_population_in, population_out_list=loss_population_out)
precision_avg,acc_avg,recall_avg,threshold_avg,cnt_corr_in_avg,cnt_wrong_out_avg = get_threshold_precision_avg_acc(loss_main_in, loss_main_out, training_in_list=loss_train_in, training_out_list=loss_train_out)


print(f'{avg_auc},{diff_auc},{precision_avg},{precision_pop},{precision},{recall_avg},{recall_pop},{recall},{acc_avg},{acc_pop},{acc},{threshold_avg},{threshold_pop},{threshold},{len_in},{len_out}')


