import numpy as np
#import quaternion
import sys
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import scipy.io
#import cv2
import time

from numpy import linalg as LA
from scipy.spatial import Delaunay

#from pyntcloud import PyntCloud
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import det
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from info3d import *

def fillArCorePlanes(planecloud, p_triangles, point_density = 318):

    plane_pointcloud = []

    for vertices in p_triangles:

        vx0, vx1, vx2 = planecloud[vertices]
        v1, v2 = [vx1[[0,2]], vx2[[0,2]]] - vx0[[0,2]]

        l_point_density = int(np.abs(LA.norm(np.cross(v1,v2)))*point_density)

        a1 = np.random.rand(l_point_density)
        a2 = np.random.rand(l_point_density)

        quad_points =  np.multiply(np.tile(a1,2).reshape(a1.shape[0],2), v1) + np.multiply(np.tile(a2,2).reshape(a2.shape[0],2), v2)

        quad_points2 = [[0,0], v1, v2]

        #print(quad_points)

        for q_point in quad_points:

            #print([q_point,[0,0], v1, v2])

            hull = ConvexHull([
                q_point,
                [0,0], v1, v2
            ])

            if 0 not in np.unique(hull.simplices): #len(hull.simplices) == 3:
                quad_points2.append(q_point)

        quad_points2 = np.insert(np.asarray(quad_points2),1,0,1) + vx0

        if len(plane_pointcloud) == 0:
            plane_pointcloud = quad_points2
        else:
            plane_pointcloud = np.concatenate(
                (plane_pointcloud, quad_points2), 
                axis = 0
            )
                        
    return plane_pointcloud

def subsumeCoplanar(o_planeCloud):

    t_planeCloud = [[pose, planecloud] for pose, planecloud in o_planeCloud if len(planecloud)!= 0]
    
    coplanar_scores = np.zeros((len(t_planeCloud),len(t_planeCloud),2))

    for i1, [pose, planecloud] in enumerate(t_planeCloud):

        #print(i1, pose)
        if len(planecloud) == 0: continue

        cp_planeCloud = [pc for n,pc in enumerate(t_planeCloud) if n!=i1]

        planecloud = planecloud + pose[:3]

        random_pick = np.random.choice(len(planecloud),3,False)

        n0, n1, n2 = planecloud[random_pick]
        normal_v = np.cross(n1-n0,n2-n0)
        plane_normal = normal_v/LA.norm(normal_v,2)

        d = -n0.dot(plane_normal)

        for i2, [pose2, planecloud2] in enumerate(cp_planeCloud):
            
            if len(planecloud) == 0: continue

            co_planar_distance = []

            planecloud2 = planecloud2 + pose2[:3]

            for point in planecloud2:

                co_planar_distance.append(
                    abs((np.dot(plane_normal,point[:3])+d)/LA.norm(plane_normal,ord = 2))
                )

            co_planar_distance = np.asarray(co_planar_distance)

            i2n = i2
            if i1<=i2:
                i2n = i2 + 1

            #print(i1,i2n,np.mean(co_planar_distance), np.std(co_planar_distance))
            coplanar_scores[i1,i2n] = [
                np.mean(co_planar_distance), 
                np.std(co_planar_distance)
            ]

    #plt.imshow(coplanar_scores[:,:,0])
    #np.where(coplanar_scores[:,:,1]<0.01)

    s1, s2 = np.where(coplanar_scores[:,:,0]<0.01)
    unique_pairs = np.stack((s1[np.where(s1 != s2)], s2[np.where(s1 != s2)]),0).T

    unique_planes = np.delete(np.arange(len(t_planeCloud)),np.unique(unique_pairs))

    pair_sets = []

    unique_pairs_remaining = np.copy(unique_pairs)

    #print(np.asarray(len(t_planeCloud)),np.unique(unique_pairs),unique_planes)

    while len(unique_pairs_remaining) != 0:

        n1, n2 = unique_pairs_remaining[0]

        #print(n1, n2)

        current_set = [n1, n2]

        unique_pairs_remaining[np.where(unique_pairs_remaining == n1)] = n2

        unique_pairs_cp = np.unique(unique_pairs_remaining,axis = 0)

        #print("cp",unique_pairs_cp)

        for n12, n22 in unique_pairs_cp[np.where(unique_pairs_cp == n2)[0]]:
            #print(n12,n22,current_set)
            if n12 not in current_set: current_set.append(n12)
            if n22 not in current_set: current_set.append(n22)

        remove_index = []
        for c in current_set:
            remove_index = np.concatenate(
                (np.asarray(remove_index, dtype = np.int),np.where(unique_pairs_remaining == c)[0]),
                axis = 0
            )

        unique_pairs_remaining = np.delete(unique_pairs_remaining,np.unique(remove_index),axis = 0)
        #print("rm",unique_pairs_remaining)

        #print(current_set)
        pair_sets.append(current_set)
        #print(unique_pairs_cp)

    for coplanars in pair_sets:

        area = []

        #print(coplanars)

        for index in coplanars:

            pose, planecloud  = t_planeCloud[index]

            planecloud = planecloud + pose[:3]

            PX = planecloud[:,0]
            PY = planecloud[:,1]
            PZ = planecloud[:,2]

            surfaces_delaunay = Delaunay(np.stack((PX,PZ)).T)
            p_triangles=surfaces_delaunay.simplices

            area.append(getPointCloudArea(planecloud,p_triangles))

        #print(area,coplanars[np.argmax(area)])
        unique_planes = np.append(unique_planes,coplanars[np.argmax(area)])

    #print(unique_planes)
    if len(unique_planes) == 0:
        t_planeCloud2 = t_planeCloud
    else:
        t_planeCloud2 = [[pose, planecloud] for i, [pose, planecloud] in enumerate(t_planeCloud) if i in unique_planes]
    
    return t_planeCloud2