import math
import numpy as np
import pandas as pd
import tensorflow as tf
import socket
import importlib
import os
import sys
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)
from pointnetvlad_cls import *
from loading_pointclouds_4096 import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

BATCH_NUM_QUERIES = 3#FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY= 4#FLAGS.positives_per_query
NEGATIVES_PER_QUERY= 12#FLAGS.negatives_per_query
GPU_INDEX = 0#FLAGS.gpu
DECAY_STEP = 200000#FLAGS.decay_step
DECAY_RATE = 0.7#FLAGS.decay_rate

global DATABASE_VECTORS
DATABASE_VECTORS = []

global QUERY_VECTORS
QUERY_VECTORS = []

#global QUERY_DATABASE_NUMPY
#QUERY_DATABASE_NUMPY = []

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

QUERY_PATH = 'pointnetvlad_submaps/'

DATABASE_FILE= 'pointnetvlad_submaps/3d_evaluation_database.pickle' #3d_spaces_evaluation_database
#QUERY_FILE= 'generating_queries/spatial_privacy_evaluation_query.pickle'
        
LOG_DIR = 'model/'
#MODEL_DIR = 'baseline/'
model_file= "model_3d_4m_jittered_4096.ckpt" #"model_baseline.ckpt"#

DATABASE_SETS= get_sets_dict(DATABASE_FILE)
#QUERY_SETS= get_sets_dict(QUERY_FILE)

RESULTS_FOLDER="testing_results/pointnetvlad/"
if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER) 
    
global DATABASE_DFS
DATABASE_DFS = []
    
BASE_PATH = 'pointnetvlad_submaps'
all_folders= ['raw_dataset','ransac_dataset']

print(all_folders)

for folder in ['raw_dataset','ransac_dataset']:
    
    print("Training df for:",folder)
    df_database= pd.DataFrame(columns=['file','northing','easting','alting'])

    df_locations= pd.read_csv(os.path.join(BASE_PATH,folder,"pointcloud_centroids_4m_0.25.csv"),sep=',')

    DATABASE_DFS.append(df_locations)
    

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay     

def evaluate(query_sets, query_db_np, output_file):
    
    global DATABASE_VECTORS
    global QUERY_VECTORS
    #global QUERY_DATABASE_NUMPY
    
    #DATABASE_VECTORS = []
    QUERY_VECTORS=[]
    
    ave_recall = np.zeros(25)
    ave_intra_dist = np.zeros(25)
        
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #print("In Graph")
            query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            eval_queries= placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            #print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, positives, negatives],1)
                #print(vecs)                
                out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
                #print(out_vecs)
                q_vec, pos_vecs, neg_vecs= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
                #print(q_vec)
                #print(pos_vecs)
                #print(neg_vecs)

            saver = tf.train.Saver()
            
        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, os.path.join(LOG_DIR, model_file))#MODEL_DIR, model_file))
        print("Model restored.")

        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'is_training_pl': is_training_pl,
               'eval_queries': eval_queries,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs}
        recall= np.zeros(25)
        count=0
        similarity=[]
        one_percent_recall=[]
        intra_dist_error = np.zeros(25)
        
        test_recall = []
        
        if len(DATABASE_VECTORS) == 0:
            for i in range(len(DATABASE_SETS)):
                print(i,len(DATABASE_SETS))
                DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i]))
        
        for j in range(len(query_sets)):
            print(j,len(query_sets))

            QUERY_VECTORS.append(get_latent_vectors(sess, ops, query_sets[j]))

        for m in range(len(DATABASE_SETS)):
            for n in range(len(query_sets)):
                #if(m==n): # We remove this cause we have disjoint testing sets for partial spaces.
                #    continue
                pair_recall, pair_similarity, pair_opr, mean_intra_dist, top1_obj_cands = get_recall(sess, ops, m, n, query_sets, query_db_np)
                recall+=np.array(pair_recall)
                intra_dist_error += np.array(mean_intra_dist)
                count+=1
                one_percent_recall.append(pair_opr)
                test_recall.append([
                    m,
                    n,
                    top1_obj_cands
                ])
                for x in pair_similarity:
                    similarity.append(x)

        ave_recall=recall/count
        

        print(" Average Inter-space Error:",1-0.01*ave_recall[0])
        
        ave_intra_dist=intra_dist_error/count
        #print("Recall:",recall,count)
        print(" Ave Intra-space Distance Error:",ave_intra_dist[0])

        #print(similarity)
        average_similarity= np.mean(similarity)
        print(" Average similarity:",average_similarity)
        
    return np.asarray(1-0.01*ave_recall), np.asarray(ave_intra_dist), test_recall

def get_latent_vectors(sess, ops, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    #print(len(train_file_idxs))
    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        file_names=[]
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries=load_pc_files(file_names)
        # queries= np.expand_dims(queries,axis=1)
        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)
        #print(q1.shape)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        #print(q2.shape)
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3, ops['is_training_pl']:is_training}
        o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)
        
        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))

        out=np.vstack((o1,o2,o3))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):  
        q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        index=train_file_idxs[q_index]
        queries=load_pc_files([dict_to_process[index]["query"]])
        queries= np.expand_dims(queries,axis=1)
        #print(query.shape)
        #exit()
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
        #print(q.shape)
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        #print(output.shape)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    print(q_output.shape)
    return q_output

def get_recall(sess, ops, m, n, query_sets, query_db_np):
    global DATABASE_VECTORS
    global QUERY_VECTORS
    global DATABASE_DFS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]
    
    print(m,n,"Database output:",database_output.shape,"Queries output:",queries_output.shape)

    #print(len(queries_output))
    database_nbrs = KDTree(database_output)
    
    database_numpy = np.asarray(DATABASE_DFS[m])

    num_neighbors=25
    recall=[0]*num_neighbors
    intra_space_distance = [0]*num_neighbors

    top1_similarity_score=[]
    top1_obj_candidate = []
    one_percent_retrieved=0
    threshold=max(int(round(len(database_output)/100.0)),1)

    num_evaluated=0
    for i in range(len(queries_output)):
        true_neighbors= query_sets[n][i][m]
        if(len(true_neighbors)==0):
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
        
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j==0):
                    similarity= np.dot(queries_output[i],database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j]+=1
                # Only get the intra space for correct labels.
                intra_space_distance[j]+= np.linalg.norm(database_numpy[indices[0][j],1:-1] - query_db_np[i,1:-1])
                break
                
        top1_obj_candidate.append([
            query_db_np[i,1:-1],
            query_db_np[i,-1], # actual obj label
            database_numpy[indices[0][0],-1], # top candidate obj label
        ])
        
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
    recall=(np.cumsum(recall)/float(num_evaluated))*100
    mean_intra_space_distance = np.array(intra_space_distance)/float(num_evaluated)
    
    #print(recall)
    #print(np.mean(top1_similarity_score))
    #print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall, mean_intra_space_distance, np.asarray(top1_obj_candidate)

def get_similarity(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    threshold= len(queries_output)
    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    similarity=[]
    for i in range(len(queries_output)):
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=1)
        for j in range(len(indices[0])):
            q_sim= np.dot(q_output[i], database_output[indices[0][j]])
            similarity.append(q_sim)
    average_similarity=np.mean(similarity)
    print(average_similarity)
    return average_similarity