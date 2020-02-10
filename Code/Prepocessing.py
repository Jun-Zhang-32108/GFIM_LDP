# -*- coding: utf-8 -*-
# test
# 2017/12/24

import csv
import time
from itertools import islice
import random 
                    
#main函数
if __name__ == '__main__':

    #读取源数据文件

    user_itemset_fixsized = 47

    for filename in ['USCensus']:
        fileroute = 'F:\study_BUPT\Thesis\Code_Lite\Data\USCensus'+'//'+filename+'.txt'
        fileout = 'F:\study_BUPT\Thesis\Code_Lite\Data'+'//'+filename+'_items.csv'
        fileout2 = 'F:\study_BUPT\Thesis\Code_Lite\Data'+'//'+'dataset_descriptions.csv'
        fileout3 = 'F:\study_BUPT\Thesis\Code_Lite\Data'+'//'+filename+'_true_Topk.csv'
        
        print('Source Filerout: %s'%fileroute)

        #读取文件和写入文件
        old_file1 = open(fileroute,'rb')
        new_file1 = open(fileout,'wb')
        new_file2 = open(fileout2,'ab+')
        new_file3 = open(fileout3,'wb')
                
        old_reader1 = csv.reader(old_file1)
        csv_writer1 = csv.writer(new_file1,dialect = 'excel')
        csv_writer2 = csv.writer(new_file2,dialect = 'excel')
        csv_writer3 = csv.writer(new_file3,dialect = 'excel')
  
        max_itemset_length = 0
        counter = 0 
        notset = 0
        index = 0
        dict_distinct ={}
        for row1 in old_reader1:
            index += 1
            row = str(row1).strip('\n').strip(']').strip('[').strip('\'').split(' ')
            # print(row)
            row.pop()
            # print('pop')
            # print(row)
            # row = str(row1).strip('\n').strip(']').strip('[').strip('\'').split(' ')
            # print(row)
            l1 = len(row)

            l2 = len(set(row))
            if l2 < l1:
                notset += 1
                print('My index is %d. I am not set.'%index)
            row = set(row)
            row = list(row)
            if max_itemset_length < len(row):
                max_itemset_length = len(row)
            for i in row:
                if i in dict_distinct:
                    dict_distinct[i] += 1
                else:
                    dict_distinct[i] =  1
            if len(row) < user_itemset_fixsized:
                counter += 1
                Comple_set = [9999]*(user_itemset_fixsized-len(row))
                row.extend(Comple_set)
            elif len(row) > user_itemset_fixsized:
                row = random.sample(row,user_itemset_fixsized)

            row.append(0.5) 
            csv_writer1.writerow(row)

            # csv_writer1.writerow([row1[0],row1[1],row1[3],row1[5],row1[7]])
            #new_file1.writelines([word])
        csv_writer2.writerow(['User_Number','max_itemset_length','user_itemset_fixsized','percentage','number_of_distinct_items'])
        csv_writer2.writerow([index,max_itemset_length,user_itemset_fixsized,counter/float(index),len(dict_distinct)])
        print('max_itemset_length: %d'%max_itemset_length)
        print('%d users owns less than %d items, account for %f.'% (counter,user_itemset_fixsized,counter/float(index)))
        print('%d userset is not set.'%notset)
        print('There are %d disctict items.'%len(dict_distinct))
        print ('finish')
        
        dict_distinct_sorted = sorted(dict_distinct.items(), key = lambda item:item[1], reverse = True)
        for i in dict_distinct_sorted:
            csv_writer3.writerow([i[0],i[1]])

        old_file1.close()
        new_file1.close()
        new_file2.close()
        new_file3.close()


            

