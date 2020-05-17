import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import scipy.io
import time
import h5py
import csv

from sklearn.neighbors import NearestNeighbors, KDTree

#base_path= "pointnetvlad_submaps/"

global DATABASE_TREES
DATABASE_TREES = []

global DATABASE_SETS
DATABASE_SETS = []

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set
##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)
    
def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Database Trajectories Loaded.")
		return trajectories
    
def construct_query_and_database_sets(
    base_path, folders, pointcloud_fols, filename, 
    CONSTRUCT = False # True if we want to recreate the database sets and trees, False if we only want to open
):
    
	global DATABASE_TREES
	global DATABASE_SETS
    
	DATABASE_TREES = []
	DATABASE_SETS = []

	for folder in folders:
		print(folder)
		df_database = pd.DataFrame(columns=['file','northing','easting','alting','obj'])
		
		df_locations = pd.read_csv(os.path.join(base_path,folder,filename),sep=',')

		for index, row in df_locations.iterrows():
			df_database = df_database.append(row, ignore_index=True)

		database_tree = KDTree(df_database[['northing','easting','alting']])
		DATABASE_TREES.append(database_tree)

	print("Done getting database trees.")

	test_sets=[]
	test_trees=[]

	for folder in folders:
		database = {}
		test = {} 
		df_locations = pd.read_csv(os.path.join(base_path,folder,filename),sep = ',')
		df_locations['timestamp'] = folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.pickle'
		df_locations = df_locations.rename(columns = {'timestamp':'file'})
		for index,row in df_locations.iterrows():				
			database[len(database.keys())] = {'query':row['file'],'northing':row['northing'],'easting':row['easting'],'alting':row['alting']}
		DATABASE_SETS.append(database)
		#if folder not in train_folders:
		#	test_sets.append(test)
            
	print("Database (Tree) sets:",len(DATABASE_SETS))    

	if CONSTRUCT: 
		output_to_file(DATABASE_SETS, base_path+'3d_evaluation_database.pickle')
    #'partial_spaces/'+partial_name+'_evaluation_database.pickle')


def construct_query_sets(base_path,partial_path, pointcloud_fols, filename):#, partial_name):#, p, output_name):
	test_trees=[]
    
	#for folder in folders:
	#	print(folder)
	df_test= pd.DataFrame(columns=['file','northing','easting','alting','obj'])
        
	df_locations= pd.read_csv(os.path.join(base_path,partial_path,filename),sep=',')
	#df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
	#df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index, row in df_locations.iterrows():
		df_test=df_test.append(row, ignore_index=True)
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#df_test=df_test.append(row, ignore_index=True)
	test_tree = KDTree(df_test[['northing','easting','alting']])
	test_trees.append(test_tree)

	test_sets=[]
	#for folder in folders:
	test={} 
	df_locations['timestamp']=partial_path+pointcloud_fols+df_locations['timestamp'].astype(str)+'.pickle'
	df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index,row in df_locations.iterrows():				
		#entire business district is in the test set
		test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting'],'alting':row['alting']}
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
	test_sets.append(test)
            
	print(" Test (Tree) sets:",len(test_sets))    

	for i in range(len(DATABASE_SETS)):
		tree=DATABASE_TREES[i]
		for j in range(len(test_sets)):
			#if(i==j):
			#	continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"],test_sets[j][key]["alting"]]])
				index = tree.query_radius(coor, r=20) #r=4
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

	#'partial_spaces/'+partial_name+'_evaluation_database.pickle')
	output_to_file(test_sets, base_path+'3d_{}_evaluation_query.pickle'.format(partial_path))#'partial_spaces/'+partial_name+'_evaluation_query.pickle')

def construct_successive_query_sets(base_path,successive_path,partial_path, pointcloud_fols, filename):#, partial_name):#, p, output_name):
	test_trees=[]
    
	#for folder in folders:
	#	print(folder)
	df_test= pd.DataFrame(columns=['file','northing','easting','alting','obj'])
        
	df_locations= pd.read_csv(os.path.join(base_path,successive_path,partial_path,filename),sep=',')
	#df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
	#df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index, row in df_locations.iterrows():
		df_test=df_test.append(row, ignore_index=True)
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#df_test=df_test.append(row, ignore_index=True)
	test_tree = KDTree(df_test[['northing','easting','alting']])
	test_trees.append(test_tree)

	test_sets=[]
	#for folder in folders:
	test={} 
	df_locations['timestamp']=successive_path+'/'+partial_path+pointcloud_fols+df_locations['timestamp'].astype(str)+'.pickle'
	df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index,row in df_locations.iterrows():				
		#entire business district is in the test set
		test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting'],'alting':row['alting']}
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
	test_sets.append(test)
            
	print("Database (Tree) sets:",len(DATABASE_SETS),"; Test (Tree) sets:",len(test_sets))    

	for i in range(len(DATABASE_SETS)):
		tree=DATABASE_TREES[i]
		for j in range(len(test_sets)):
			#if(i==j):
			#	continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"],test_sets[j][key]["alting"]]])
				index = tree.query_radius(coor, r=20) #r=4
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

    #'partial_spaces/'+partial_name+'_evaluation_database.pickle')
	output_to_file(test_sets, base_path+'successive_queries/3d_jittered_{}_evaluation_query.pickle'.format(successive_path+"_"+partial_path))#'partial_spaces/'+partial_name+'_evaluation_query.pickle')
    