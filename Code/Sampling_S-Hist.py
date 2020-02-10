# -*- coding: utf-8 -*-
# S-Hist，二阶抽样算法，挖掘频繁项，每个用户输入一个集合
# 2018/5/16

import math
import csv
import sys
import datetime
import traceback
import psutil
import os
import numpy as np
# from Local_Randomizer import LR3,LR2
import random

class LR2():
	"""LR2"""
	def __init__(self, list_x, location, privacy, Candidate_LR ):
		self.List_x = list_x
		self.Location = location
		self.Privacy  = privacy
		self.candiate = Candidate_LR
		# self.d = len(list_x) #在区域T内的位置点（location）的个数

	def run(self):
		
		# print(len(x_location))
		c_privacy  = (math.exp(self.Privacy)+1.0)/(math.exp(self.Privacy)-1.0)
		m = 1/(self.List_x[0]**2)
		# print(self.candiate)
		if self.Location in self.candiate:
			isin = 1
			# print('Location: %d'%self.Location)
			# print('Index: %d'%self.candiate.index(self.Location))
			x_location = float(self.List_x[self.candiate.index(self.Location)])
			# print('In! Self.Location: %d. index: %d.'%(self.Location,self.candiate.index(self.Location)))
			seed = random.random()
			p1 = math.exp(self.Privacy)/(math.exp(self.Privacy)+1.0)

			if seed<p1:
				z = c_privacy*m*x_location
			else:
				z = -1*c_privacy*m*x_location
		else:
			# print('Out! Self.Location: %d.'%(self.Location))
			dummy_list = [-1,1]
			z =random.choice(dummy_list)*c_privacy*math.sqrt(m)
			# z = 0
			isin = 0
		return z, isin

class LR3():
	"""LR"""
	def __init__(self, list_x, location, privacy ):
		self.List_x = list_x
		self.Location = location
		self.Privacy  = privacy

	def run(self):
		c_privacy  = (math.exp(self.Privacy)+1.0)/(math.exp(self.Privacy)-1.0)
		m = 1/(self.List_x[0]**2)
		if int(self.Location) == 0:
			# print('0!')
			dummy_list = [-1,1]
			z =random.choice(dummy_list)*c_privacy*math.sqrt(m)
			return z
		else:		
			x_location = float(self.List_x[self.Location-1])
			seed = random.random()
			p1 = math.exp(self.Privacy)/(math.exp(self.Privacy)+1.0)
			if seed<p1:
				z = c_privacy*m*x_location
			else:
				z = -1*c_privacy*m*x_location

			return z

currentTime = datetime.datetime.now()
user_itemset_fixsized = 47

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


def S_Hist_part1_topk(Location_num,user_num,Locations_ID,Privacy_Speic,true_p,true_locations_number_sorted,csv_writer1,k):

	# Beta = 0.01
	# Sigma = math.sqrt((math.log(2*Location_num/Beta))/user_num)  
	# print('Sigma: %f'%Sigma)
	# m0 = math.log(Location_num+1)*math.log(2.0/Beta)/(Sigma**2)
	# m = int(math.ceil(m0))
	# print('m:%d'%m)
	# lis = [ -1/math.sqrt(m), 1/math.sqrt(m)] 
	# mart = [ [ random.choice(lis) for i in xrange(Location_num) ] for i in xrange(m) ]
	# mart = np.array(mart)
	dimension = 2**int(math.ceil(math.log(Location_num,2)))
	print('Dimension: %d. '%dimension)
	S = [[1,-1],[1,1]]
	while len(S) < dimension:
		S_N = []
		for i in S:
			S_N.append(i*2)
			Minus = [x*-1 for x in i]
			S_N.append(i+Minus)
		S = S_N
	S = np.array(S)*(1/math.sqrt(dimension))
	mart = S[0:Location_num].T
	z_vector = np.zeros(dimension)
	f_vector = {}
	for i in xrange(user_num):
		j = random.randint(0,dimension-1)
		x_pri = mart[j]
		LR_Test = LR3( x_pri, Locations_ID[i], Privacy_Speic[i])
		result_z = LR_Test.run()
		z_vector[j] += result_z
	for i in xrange(Location_num):
		e_vector = np.zeros(Location_num)
		e_vector[i] = 1
		f_vector[i+1] = int(np.dot(np.dot(mart,e_vector), z_vector))

	f_vector_sorted = sorted(f_vector.items(), key=lambda item:item[1], reverse = True)	
	dict_estimated = {}
	dict_estimated_frequency ={}
	for i in xrange(Location_num):
		dict_estimated[f_vector_sorted[i][0]] = i
		dict_estimated_frequency[f_vector_sorted[i][0]] = f_vector_sorted[i][1]/float(user_num)*user_itemset_fixsized 


	estimated_sum = sum(f_vector)
	print('The sum of estimated counts is %d. '%(int(estimated_sum*user_itemset_fixsized)))
	re = []
	IREL = Location_num
	candidate_set = {}
	for i in xrange(k):
		estimated_p = (f_vector_sorted[i][1]/float(user_num)*user_itemset_fixsized)
		candidate_set[f_vector_sorted[i][0]] = f_vector_sorted[i][1]
		REL = abs(Location_num-abs(i-dict_estimated[true_locations_number_sorted[i][0]]))
		if true_p[i] == 0:
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)			
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],f_vector_sorted[i][1]*user_itemset_fixsized,true_p[i],dict_estimated_frequency.get(true_locations_number_sorted[i][0]),estimated_p,'none',REL,DCG,IDCG,'single_part',currentTime]) 
			# pass
		elif i<k:
			re.append(abs(true_p[i]-dict_estimated_frequency[true_locations_number_sorted[i][0]])/true_p[i])
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],f_vector_sorted[i][1]*user_itemset_fixsized,true_p[i],dict_estimated_frequency[true_locations_number_sorted[i][0]],estimated_p,re[-1],REL,DCG,IDCG,'single_part',currentTime])



		# print('DCG%d is %f'%(i+1,DCG))
		# print('IDCG%d is %f'%(i+1,IDCG))

	# KL_Divergence = scipy.stats.entropy(true_p, estimated_p)
	# print('The KL-Divergence is %f. '%KL_Divergence)
	# candidate_set.sort()
	RE_Median1 = np.median(re)
	NDCG = DCG/float(IDCG)
	print('The relative error 1 is %f. '%RE_Median1)
	print('The DCG 1 is %f. '%DCG)
	print('The NDCG 1 is %f. '%NDCG)
	return candidate_set, RE_Median1, dict_estimated_frequency,NDCG 
	#write log
	# csv_writer2.writerow([filename,k,RE_Median,DCG,NDCG,str(datetime.datetime.now())])
	# new_file1.close()
	# new_file2.close()


#main函数
if __name__ == '__main__':

	

	filename = 'Random_Itemset_different_privacy_003_5e5_true_6'
	fileroute = 'F:\study_BUPT\Thesis\Code_Lite\Data\SynthesisData'+'//'+'new_20180522'+'//'+filename+'.csv'
	# fileroute = 'F:\study_BUPT\Thesis\Code_Lite\Data\Kosarak'+'//'+filename+'.csv'
	old_file1 = open(fileroute,'r')
	old_reader1 = csv.reader(old_file1)
	print('Source Route1: %s'%fileroute)

	fileout = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+filename+'_estimated_lite_half_topk_1_compared.csv'
	new_file1 = open(fileout,'ab+')
	csv_writer1 = csv.writer(new_file1,dialect='excel')
	print('Output Route: %s'%fileout)

	# 记录运行参数
	fileout2 = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+'algorithm_log_compared.csv'
	new_file2 = open(fileout2,'ab+')
	csv_writer2 = csv.writer(new_file2,dialect='excel')
	print('Output Route2: %s'%fileout2)

	fileout3 = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+'algorithm_log_compared_complete.csv'
	new_file3 = open(fileout3,'ab+')
	csv_writer3 = csv.writer(new_file3,dialect='excel')
	print('Output Route2: %s'%fileout3)

	
	Locations_ID_Uni0 = {}
	count0 = 0
	cache = []
	Pri_Par = 1
	print('Privacy badget: %f'%Pri_Par)
	Privacy_Speic = []
	for row in old_reader1:
		# Privacy_Speic.append(float(row.pop()))
		row.pop()
		Privacy_Speic.append(Pri_Par)
		cache.append(list(row))
		for row_tem in row:
			loc = int(row_tem)
			if loc != 0:
				if loc in Locations_ID_Uni0:
					Locations_ID_Uni0[loc] += 1
				else:
					Locations_ID_Uni0[loc] = 1
			else:
				count0 += 1


	User_num_Total = len(cache)
	Dummy_Perce = float(count0)/User_num_Total
	Location_num_Total = 1000
	print('Location_num_Total:%d'%Location_num_Total)
	print('User_num_Total:%d'%User_num_Total)
	print('There are %d dummy items, account for %f .'%(count0,Dummy_Perce))

	# if Location_num_Total < k or len(Locations_ID_Uni0) < k:
	# 	print('The k value is too large! ')
	# 	sys.exit()

	Locations_ID_Uni = {}
	for i in xrange(1,Location_num_Total+1):
		if i in Locations_ID_Uni0:
			Locations_ID_Uni[i] = Locations_ID_Uni0[i]
		else:
			Locations_ID_Uni[i] = 0
	# print(Locations_ID_Uni.items())
	true_locations_number_sorted0 = sorted(Locations_ID_Uni.items(), key = lambda item:item[1], reverse = True)
	true_frequency = []
	csv_writer1.writerow(['Actual Item ID','Actual Item Number','Estimated Item ID', 'Estimated Item Number','Actual Frequency','Estimated Freqency','Estimated Top k frequency','RE','REL','DCG','IDCG','Type','Start Time'])
	csv_writer2.writerow(['Dataset','User Number','Itemset Size','Dummy Item Proportion','k','Candidate Set Size','Privacy_Parameter','User Itemset Size','RE1','RE2','NDCG1','NDCG2','Start Time','Finishe Time',])
	csv_writer3.writerow(['Dataset','User Number','Itemset Size','Dummy Item Proportion','k','Candidate Set Size','Privacy_Parameter','User Itemset Size','RE1','RE2','NDCG1','NDCG2','Start Time','Finishe Time',])

	# k = 30

	for k in xrange(1,21):
		for inte in xrange(k):		
			true_frequency.append(float(true_locations_number_sorted0[inte][1])/User_num_Total)				
		RE_List = []
		NDCG_List = []
		for counter in xrange(10):
			# Locations_ID， Region_ID, Privacy_Speic
			Locations_ID = []
			
			for row1 in cache:
				# print('Cache:')
				# print(row1)
				loc1 = int(random.choice(row1))
				Locations_ID.append(loc1)
								
			random.shuffle(Locations_ID)
			Candidate = {}
			dict_estimated_frequency_part1_0 = {}
			# csv_writer1.writerow(['Privacy_Parameter:',Pri_Par,'Number:',counter])
			csv_writer1.writerow(['k:',k,'Number:',counter])
			Candidate, RE1, dict_estimated_frequency_part1_0,NDCG1 = S_Hist_part1_topk(Location_num_Total,User_num_Total,Locations_ID,Privacy_Speic,true_frequency,true_locations_number_sorted0,csv_writer1,k)
			RE_List.append(RE1)
			NDCG_List.append(NDCG1)
			csv_writer3.writerow([filename,User_num_Total,Location_num_Total,Dummy_Perce,k,'None',Pri_Par,user_itemset_fixsized,RE1,'None',NDCG1,'None',currentTime,str(datetime.datetime.now())])
		csv_writer2.writerow([filename,User_num_Total,Location_num_Total,Dummy_Perce,k,'None',Pri_Par,user_itemset_fixsized,np.mean(RE_List),'None',np.mean(NDCG_List),'None',currentTime,str(datetime.datetime.now())])
	old_file1.close()
	new_file1.close()
	new_file2.close()
	new_file3.close()
	endtime = datetime.datetime.now()
	w=open('Process_Log.txt','a+')
	w.write('Run time: %d'%int((endtime-currentTime).seconds))
	w.write('\r\nStatus:Done')
	w.write('\r\n')
	w.write('\r\n')
	w.write('\r\n')
	w.close()
			
