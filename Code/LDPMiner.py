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


def S_Hist_part1_topk(Location_num,user_num,Locations_ID,Privacy_Speic,true_p,true_locations_number_sorted,csv_writer1,CandidateSet_Size,k):

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
		LR_Test = LR3( x_pri, Locations_ID[i], Privacy_Speic/2.0)
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
	estimated_topk_set =set()
	true_topk_set =set()
	for i in xrange(CandidateSet_Size):
		estimated_p = (f_vector_sorted[i][1]/float(user_num)*user_itemset_fixsized)
		candidate_set[f_vector_sorted[i][0]] = f_vector_sorted[i][1]
		REL = abs(Location_num-abs(i-dict_estimated[true_locations_number_sorted[i][0]]))
		if true_p[i] == 0:
			estimated_topk_set.add(f_vector_sorted[i][0])
			true_topk_set.add(true_locations_number_sorted[i][0])
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)			
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],f_vector_sorted[i][1]*user_itemset_fixsized,true_p[i],dict_estimated_frequency.get(true_locations_number_sorted[i][0]),estimated_p,'none',REL,DCG,IDCG,'part1',currentTime,candidate_set.keys(),'None']) 
			# pass
		elif i<k:
			estimated_topk_set.add(f_vector_sorted[i][0])
			true_topk_set.add(true_locations_number_sorted[i][0])
			re.append(abs(true_p[i]-dict_estimated_frequency[true_locations_number_sorted[i][0]])/true_p[i])
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],f_vector_sorted[i][1]*user_itemset_fixsized,true_p[i],dict_estimated_frequency[true_locations_number_sorted[i][0]],estimated_p,re[-1],REL,DCG,IDCG,'part1',currentTime,candidate_set.keys(),'None'])


	# KL_Divergence = scipy.stats.entropy(true_p, estimated_p)
	# print('The KL-Divergence is %f. '%KL_Divergence)
	# candidate_set.sort()
	Precision = len(true_topk_set.intersection(estimated_topk_set))/float(k)
	RE_Median1 = np.median(re)
	NDCG = DCG/float(IDCG)
	print('The relative error 1 is %f. '%RE_Median1)
	print('The DCG 1 is %f. '%DCG)
	print('The NDCG 1 is %f. '%NDCG)
	print('The Precision1 is %f'%Precision)
	return candidate_set, RE_Median1, dict_estimated_frequency,NDCG, Precision 
	#write log
	# csv_writer2.writerow([filename,k,RE_Median,DCG,NDCG,str(datetime.datetime.now())])
	# new_file1.close()
	# new_file2.close()

def S_Hist_part2_topk(Location_num,Candidate,user_num_total,Locations_ID,Privacy_Speic,true_p,true_locations_number_sorted,csv_writer1,dict_estimated_frequency_part1,k):

	Candidate_2LR = list(Candidate.keys())
	Candidate_2LR.sort()
	print('The candidate set is ')
	print(Candidate_2LR)
	Location_num_part2 = len(Candidate_2LR)
	print('The length of candidate set is %d'%Location_num_part2)
	if Location_num_part2 < user_itemset_fixsized:
		user_itemset_fixsized_new = Location_num_part2
	else:
		user_itemset_fixsized_new = user_itemset_fixsized
	# Beta = 0.01
	# Sigma = math.sqrt((math.log(2*Location_num_part2/Beta))/user_num)  
	# print('Sigma: %f'%Sigma)
	# m0 = math.log(Location_num_part2+1)*math.log(2.0/Beta)/(Sigma**2)
	# m = int(math.ceil(m0))
	# print('m:%d'%m)
	# lis = [ -1/math.sqrt(m), 1/math.sqrt(m)] 
	# mart = [ [ random.choice(lis) for i in xrange(Location_num_part2) ] for i in xrange(m) ]
	# mart = np.array(mart)
	dimension = 2**int(math.ceil(math.log(Location_num_part2,2)))
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
	mart = S[0:Location_num_part2].T
	# print(mart)
	z_vector = np.zeros(dimension)
	f_vector = {}
	isin_counter = 0
	for i in xrange(user_num_total):
		j = random.randint(0,dimension-1)
		# print('J1:%d'%j)
		x_pri = mart[j]
		LR_Test = LR2( x_pri, Locations_ID[i], Privacy_Speic/2,Candidate_2LR)
		result_z, isin_temp = LR_Test.run()
		# print('J2:%d'%j)
		z_vector[j] += result_z
		isin_counter += isin_temp
	Fall_IN_Candidate_Perce = isin_counter/float(user_num_total) 
	print('There are %d items in candidate_set, accont for %f'%(isin_counter,Fall_IN_Candidate_Perce))

	for i in xrange(Location_num_part2):
		e_vector = np.zeros(Location_num_part2)
		e_vector[i] = 1
		# print('Candidate_2LR: %d'%Candidate_2LR[i])
		f_vector[Candidate_2LR[i]] = int(np.dot(np.dot(mart,e_vector), z_vector))

	# f_vector_sorted = sorted(enumerate(f_vector), key=lambda x:x[1], reverse = True)
	dict_estimated_frequency_final = {}
	dict_estimated_frequency_part2 ={}
	for i in xrange(1,Location_num+1):
		if i in f_vector:
			dict_estimated_frequency_part2[i] = f_vector[i]/float(user_num_total)*user_itemset_fixsized_new
			dict_estimated_frequency_final[i] = (dict_estimated_frequency_part1[i] + dict_estimated_frequency_part2[i]*(user_itemset_fixsized_new-1))/float(user_itemset_fixsized_new)
		else:
			dict_estimated_frequency_part2[i]	= 0
			dict_estimated_frequency_final[i] = dict_estimated_frequency_part1[i]


	f_vector_sorted = sorted(dict_estimated_frequency_final.items(), key = lambda item:item[1], reverse = True)
	dict_estimated = {}
	dict_estimated_frequency_output = {}
	for i in xrange(Location_num):
		dict_estimated[f_vector_sorted[i][0]] = i
		# dict_estimated_frequency_output[f_vector_sorted[i][0]] = f_vector_sorted[i][1]

	estimated_sum = sum(f_vector)
	print('The sum of estimated counts of heavy hitters is %d. '%(int(estimated_sum*user_itemset_fixsized)))
	re = []
	IREL = Location_num
	estimated_topk_set = set()
	true_topk_set =set()
	for i in xrange(k):
		estimated_num = f_vector_sorted[i][1]*user_num_total
		REL = (abs(Location_num-abs(i-dict_estimated[true_locations_number_sorted[i][0]])))
		estimated_topk_set.add(f_vector_sorted[i][0])
		true_topk_set.add(true_locations_number_sorted[i][0])
		if true_p[i] == 0:
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],estimated_num,true_p[i],dict_estimated_frequency_final.get(true_locations_number_sorted[i][0]),f_vector_sorted[i][1],'none',REL,DCG,IDCG,'part2',currentTime,Candidate_2LR,Fall_IN_Candidate_Perce])
		else:
			re.append(abs(true_p[i]-dict_estimated_frequency_final[true_locations_number_sorted[i][0]])/true_p[i])
			if i == 0:
				DCG = REL
				IDCG = IREL
			else:
				DCG += REL/math.log(i+1,2)
				IDCG += IREL/math.log(i+1,2)
			csv_writer1.writerow([true_locations_number_sorted[i][0],true_locations_number_sorted[i][1],f_vector_sorted[i][0],estimated_num,true_p[i],dict_estimated_frequency_final[true_locations_number_sorted[i][0]],f_vector_sorted[i][1],re[-1],REL,DCG,IDCG,'part2',currentTime,Candidate_2LR,Fall_IN_Candidate_Perce])

	# KL_Divergence = scipy.stats.entropy(true_p, estimated_p)
	# print('The KL-Divergence is %f. '%KL_Divergence)
	Precision = len(true_topk_set.intersection(estimated_topk_set))/float(k)
	RE_Median2 = np.median(re)
	NDCG = DCG/float(IDCG)
	print('The relative error 2 is %f. '%RE_Median2)
	print('The DCG 2 is %f. '%DCG)
	print('The NDCG 2 is %f. '%NDCG)
	print('The Precision2 is %f'%Precision)
	return RE_Median2, NDCG, Precision 

	# new_file1.close()



#main函数
if __name__ == '__main__':



	filename = 'USCensus_items_backup'
	fileroute = 'F:\study_BUPT\Thesis\Code_Lite\Data'+'//'+filename+'.csv'
	old_file1 = open(fileroute,'r')
	old_reader1 = csv.reader(old_file1)
	print('Source Route1: %s'%fileroute)

	fileout = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+filename+'_estimated_lite_2phases.csv'
	new_file1 = open(fileout,'ab+')
	csv_writer1 = csv.writer(new_file1,dialect='excel')
	print('Output Route: %s'%fileout)

	# 记录运行参数
	fileout2 = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+'algorithm_log_2phases.csv'
	new_file2 = open(fileout2,'ab+')
	csv_writer2 = csv.writer(new_file2,dialect='excel')
	print('Output Route2: %s'%fileout2)

	fileout3 = 'F:\study_BUPT\Thesis\Code_Lite\Data\EstimatedData'+'//'+'new_20180522'+'//'+'algorithm_log_2phases_complete.csv'
	new_file3 = open(fileout3,'ab+')
	csv_writer3 = csv.writer(new_file3,dialect='excel')
	print('Output Route2: %s'%fileout3)


	# print('Privacy badget: %f'%Pri_Par)
	Locations_ID_Uni0 = {}
	count0 = 0
	cache = []

	for row in old_reader1:
		# Privacy_Speic.append(float(row.pop()))
		row.pop()

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
	Location_num_Total = 400
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
	csv_writer1.writerow(['Actual Item ID','Actual Item Number','Estimated Item ID', 'Estimated Item Number','Actual Frequency','Estimated Freqency','Estimated Top k frequency','RE','REL','DCG','IDCG','Type','Start Time','Candidate Set','Fall In Candidate Percent'])
	csv_writer2.writerow(['Dataset','User Number','Itemset Size','Dummy Item Proportion','k','Candidate Set Size','Privacy_Parameter','User Itemset Size','RE1','RE2','NDCG1','NDCG2','Precision1','Precision2','Start Time','Finishe Time',])
	csv_writer3.writerow(['Dataset','User Number','Itemset Size','Dummy Item Proportion','k','Candidate Set Size','Privacy_Parameter','User Itemset Size','RE1','RE2','NDCG1','NDCG2','Precision1','Precision2','Start Time','Finishe Time',])


	k = 10	
	# Pri_Par = 1
	Pri_Par_list = []
	print('k: %d'%k)
	for i in xrange(1,15):
		Pri_Par_list.append(i/10.0)
	CandidateSet_Size = 2*k
	for inte in xrange(CandidateSet_Size):		
		true_frequency.append(float(true_locations_number_sorted0[inte][1])/User_num_Total)
	for Pri_Par in Pri_Par_list:
		RE_List1 = []
		NDCG_List1 = []
		RE_List2 = []
		NDCG_List2 = []
		Precision1_List =[]
		Precision2_List =[]
		for counter in xrange(20):
			print('Privacy_Parameter: %f'%Pri_Par)
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
			csv_writer1.writerow(['Privacy_Parameter:',Pri_Par,'Number:',counter])
			Candidate, RE1, dict_estimated_frequency_part1_0,NDCG1, Precision1 = S_Hist_part1_topk(Location_num_Total,User_num_Total,Locations_ID,Pri_Par,true_frequency,true_locations_number_sorted0,csv_writer1,CandidateSet_Size,k)
			CandidateSet =  set([str(i) for i in Candidate.keys()])
			Locations_ID = []
			if CandidateSet_Size < user_itemset_fixsized:
				user_itemset_fixsized_2 = CandidateSet_Size
			else:
				user_itemset_fixsized_2 =user_itemset_fixsized
			for row2 in cache:
				new_list = list(set(row2).intersection(CandidateSet))
				if len(new_list) < user_itemset_fixsized_2:
					new_list.extend([0]*(user_itemset_fixsized_2-len(new_list)))
				elif len(new_list) > user_itemset_fixsized_2:
					print('Error!')
					sys.exit()
				loc_Tem = int(random.choice(new_list))
				Locations_ID.append(loc_Tem)
			# Location_num,Location_num_part2,Candidate,user_num,user_num_total,Locations_ID,Privacy_Speic,Locations_ID_Uni,csv_writer1,csv_writer2
			RE2,NDCG2, Precision2 = S_Hist_part2_topk(Location_num_Total,Candidate,User_num_Total,Locations_ID,Pri_Par,true_frequency,true_locations_number_sorted0,csv_writer1,dict_estimated_frequency_part1_0,k)
			RE_List1.append(RE1)
			NDCG_List1.append(NDCG1)
			Precision1_List.append(Precision1)
			RE_List2.append(RE2)
			NDCG_List2.append(NDCG2)
			Precision2_List.append(Precision2)
			csv_writer3.writerow([filename,User_num_Total,Location_num_Total,Dummy_Perce,k,CandidateSet_Size,Pri_Par,user_itemset_fixsized,RE1,RE2,NDCG1,NDCG2,Precision1,Precision2,currentTime,str(datetime.datetime.now())])
		csv_writer2.writerow([filename,User_num_Total,Location_num_Total,Dummy_Perce,k,CandidateSet_Size,Pri_Par,user_itemset_fixsized,np.mean(RE_List1),np.mean(RE_List2),np.mean(NDCG_List1),np.mean(NDCG_List2),np.mean(Precision1_List),np.mean(Precision2_List),currentTime,str(datetime.datetime.now())])
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
			
