# -*- coding: utf-8 -*-
# 合成测试数据集
# 2018/4/2

import math
import csv
import sys
import datetime
import psutil
import os
import numpy as np
import random
import matplotlib.pyplot as plt

currentTime = datetime.datetime.now()

def foo(exctype, value, tb,):
    import traceback
    global currentTime
    endtime = datetime.datetime.now()
    w=open('Process_Log.txt','a+')
    w.write('Run time: %d'%int((endtime-currentTime).seconds))
    w.write('\r\nStatus:Error')
    w.write('\r\n')
    w.write('\r\n')
    w.write('\r\n')
    w.close()
    a=open('Exception_Log.txt','a+')
    lines = traceback.format_exception(exctype, value, tb)  
    exception_info = ''.join(lines)
    print(exception_info)
    currentTime = datetime.datetime.now()
    Timestamp=currentTime.strftime("%Y-%m-%d %H:%M:%S")
    a.write(str(Timestamp))
    a.write('\r\n')
    a.write(exception_info)
    a.write('\r\n')
    a.write('\r\n')
    a.close()
sys.excepthook=foo

def proc_info():
    a=open('Process_Log.txt','a+')
    argv0_list = sys.argv[0].split('/');
    script_name = argv0_list[len(argv0_list) - 1]
    pid = int(os.getpid()) 
    print ('PID:%d'%pid)
    p = psutil.Process(pid)
    Timestamp=currentTime.strftime("%Y-%m-%d %H:%M:%S")
    a.write(str(Timestamp))
    a.write('\r\n')
    a.write(str(p.name()))
    a.write('   ')
    a.write(str(script_name))
    a.write('   ')
    a.write(str(pid))
    a.write('   ')
    a.write(str(p.status()))
    a.write('\r\n')
    a.write('Bin Route: ')
    a.write(str(p.exe()))
    a.write('\r\n')
    a.write('PID Route: ')
    a.write(str(p.cwd()))
    a.write('\r\n')
    a.write('------------------------------ Process Use -------------------------------------')
    a.write('\r\n')
    a.write(str(p.cpu_times()))
    a.write('\r\n')
    a.write('Memory usage : %s%%' % p.memory_percent())
    a.write('\r\n')
    a.write(str(p.io_counters()))
    a.write('\r\n')
    a.write(str(p.connections()))
    a.write('\r\n')
    a.write('Process number of threads : %s' % p.num_threads())
    a.write('\r\n')
    a.close()
proc_info()

#Synthesis the test data whose frequency exhibits exponential deca

def syn_exp(num): #num = the sample size
	scale = 10 # =1/Lambda
	rand_data0 = np.random.exponential(scale, num)
	rand_data  = []
	for i in rand_data0:
		rand_data.append(int(i))
	#test
	# print(len(rand_data))
	# print(np.var(rand_data))
	# # for i in rand_data:
	# # 	print(i)
	# count, bins, ignored = plt.hist(rand_data, normed=True)
	# plt.show()
	
	return rand_data

def syn_normal(num): #num = the sample size
    standard_devitation = 30*math.sqrt(2)
    mean = 500
    rand_data0 = np.random.normal(mean,standard_devitation, num)
    rand_data  = []
    for i in rand_data0:
        rand_data.append(int(i))
    #test
    # print(len(rand_data))
    # print(np.var(rand_data))
    # # for i in rand_data:
    # #     print(i)
    count, bins, ignored = plt.hist(rand_data, 50, normed=True)
    plt.show()
    
    return rand_data

def syn_laplace(num): #num = the sample size
    lambda1 = 30
    mean = 500
    rand_data0 = np.random.laplace(mean,lambda1, num)
    rand_data  = []
    for i in rand_data0:
        rand_data.append(int(i))
    #test
    # print(len(rand_data))
    # print(np.var(rand_data))
    # # for i in rand_data:
    # #     print(i)
    count, bins, ignored = plt.hist(rand_data, 50, normed=True)
    plt.show()
    
    return rand_data


#main函数
if __name__ == '__main__':
    
	user_itemset_fixsized = 50
	filename = 'Normal_Itemset_different_privacy_003_5e5_true.csv'
	fileout = 'F:\study_BUPT\Thesis\Code_Lite\Data\SynthesisData'+'//'+'new_20180522'+'//'+filename
	new_file1 = open(fileout,'wb')
	csv_writer1 = csv.writer(new_file1,dialect='excel')
	print('Output Route: %s'%fileout)



	num_entry = 500000
	sample_size = 500000
	# exp_data = syn_exp(num_entry)
	# normal_data = syn_normal(num_entry)
	laplace_data = syn_normal(sample_size)
	ratio = num_entry/sample_size
	rand_list = []
	frequency_alphabet = {}
	for i in laplace_data:
		if i in frequency_alphabet:
			frequency_alphabet[i] += 1
		else:
			frequency_alphabet[i] = 1
	frequency_alphabet_sorted = sorted(frequency_alphabet.items(), key = lambda item:item[1], reverse = True)        
	pri_list1 = [0.25, 0.5, 0.75]
	pri_list2 = [0.75, 1.0, 1.25]
	pri       = math.log(3)
	# Privacy_Param =
	list_tem = []
	write_list= [ [] for i in xrange(num_entry)]
	print('Max: %d'%max(laplace_data))
	# for i in xrange(num_entry):
	# 	# pri = random.choice(pri_list1)
	# 	seta = set() 
	# 	while len(seta) < user_itemset_fixsized:         
	# 		seta.add(random.choice(rand_list))
	# 	list_tem = list(seta)
	# 	list_tem.append(0.5)
	# 	csv_writer1.writerow(list_tem)
	lis = [ i for i in xrange(num_entry)]   
	for i in xrange(len(frequency_alphabet_sorted)):
		num_choice = frequency_alphabet_sorted[i][1]*ratio*60
		# print('Number Choice: %d'%num_choice)
		list_choice = random.sample(lis, num_choice)
		for j in list_choice:
			if len(write_list[j]) < (user_itemset_fixsized):
				write_list[j].append(frequency_alphabet_sorted[i][0])
	for i in write_list:
		if len(i) == user_itemset_fixsized:
			random.shuffle(i)
			i.append(3)
			csv_writer1.writerow(i)
		else:
			random.shuffle(i)
			dummy_num = user_itemset_fixsized - len(i)
			i.extend([0]*dummy_num)
			i.append(3)
			csv_writer1.writerow(i)
	new_file1.close()

	read_file1 =open(fileout,'r')
	old_file1 = csv.reader(read_file1)
	count_dict = {}
	for i in old_file1:
		i.pop()
		for term in i:
			if int(term) in count_dict:
				count_dict[int(term)] += 1
			else:
				count_dict[int(term)] = 1
	count_dict_sorted = sorted(count_dict.items(), key = lambda item:item[0], reverse = False)
	x_cord = []
	y_cord = []
	for i in xrange(1,len(count_dict_sorted)):
		x_cord.append(count_dict_sorted[i][0])
		y_cord.append(count_dict_sorted[i][1])
	#for i in count_dict:
		#print('Key:%s  , Values:%d'%(i,count_dict[i]))
	plt.plot(x_cord,y_cord)
	plt.show()
	endtime = datetime.datetime.now()
	w=open('Process_Log.txt','a+')
	w.write('Run time: %d'%int((endtime-currentTime).seconds))
	w.write('\r\nStatus:Done')
	w.write('\r\n')
	w.write('\r\n')
	w.write('\r\n')
	w.close()


