
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import pickle
from mpl_toolkits.mplot3d import Axes3D

from numpy import linalg as LA
from scipy.spatial import Delaunay
from sklearn.preprocessing import normalize

import time

from info3d import *
#from utils_3d import *

#from pyntcloud import PyntCloud


# In[ ]:


def GetRansacPlanes(
    pointCloud,
    triangles,
    planes_to_find = 30, # number of planes to find
    threshold = 0.05,     # the point-plane distance threshold
    inter_point_threshold = 0.25, # the inter-point distance within a RANSAC candidate plane
    normal_threshold_multiplier = 5,
    trials = 100,       # the number of RANSAC trials
    #plane_group = 200     # number of nearby points per plane
):

    planeCollection = []
    test_max = 10

    t0 = time.time()

    planes = []
    plane_properties = []

    depletable_pc = np.copy(pointCloud)
    #print("true points",len(depletable_pc))

    zero_normals = 0
    added_zero_normals = 0
    
    # Getting the planes
    for i_plane in np.arange(planes_to_find):
        #pass
        bestPoints = []
        trial = 0
        t1 = time.time()

        #neighbours, distance = getEuclideanNearestNeighbours(depletable_pc,3)

        #print(object_name,i_plane,len(depletable_pc))
        if len(depletable_pc) < 3:
            continue
        for i_trials in np.arange(trials):

            sample = np.random.randint(len(depletable_pc))

            testPlane = [depletable_pc[sample,:3],depletable_pc[sample,3:]]
                         #np.cross(depletable_pc[near1,:3]-depletable_pc[sample,:3],
                         #        depletable_pc[near2,:3]-depletable_pc[sample,:3]
                         #        )]
            testPoints = []
            d = -testPlane[0].dot(testPlane[1])
            for i, point in enumerate(depletable_pc): #[neighbours[sample]]
                #print(np.asarray(testPoints).shape)
                if len(testPoints) != 0:
                    #print(point.shape,depletable_pc[testPoints].shape)
                    if np.min(LA.norm(point[:3]-depletable_pc[testPoints,:3],axis = 1)) > inter_point_threshold:
                        continue
                if LA.norm(point[3:]) == 0:
                    zero_normals += 1
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        # only add a point with zero_normal if very close to the plane
                        added_zero_normals += 1
                        testPoints.append(i)
                if abs(np.dot(testPlane[1],point[3:])/(LA.norm(testPlane[1])*LA.norm(point[3:]))) > (normal_threshold_multiplier*threshold):
                    #1-20*threshold):
                    # if normals are close accept if near to the candidate plane
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        testPoints.append(i)
            if len(testPoints) < 20:
                continue
            if len(testPoints) > len(bestPoints):#plane_group:
                trial += 1
                bestPlane = testPlane
                bestPoints = testPoints

        #print(object_name,i_plane,len(depletable_pc),bestPlane,depletable_pc[sample])
        #print("Added a ",bestPlane," in",time.time()-t1,"seconds")
        if trial > 1:
            d = -bestPlane[0].dot(bestPlane[1])

            PX = depletable_pc[bestPoints][:,0]
            PY = depletable_pc[bestPoints][:,1]
            PZ = depletable_pc[bestPoints][:,2]        

            # Generalize the bestPlane to one of the surface-axis to get a 3-point Delaunay match
            phi = math.fabs(bestPlane[1][1]* 1./LA.norm(bestPlane[1])) # y/r

            if math.degrees(math.acos(phi)) < 45 : # arc-cos(y/r) = phi < 45 --> horizontal
                # use floor (plane x-z) as origin mesh
                PY = (-bestPlane[1][0] * PX - bestPlane[1][2] * PZ - d) * 1. /bestPlane[1][1]

                surfaces_delaunay = Delaunay(np.stack((PX,PZ)).T)
                p_triangles=surfaces_delaunay.simplices
                #orientation = 'horizontal'
            else:
                #use vertical wall x-y as origin mesh
                PZ = (-bestPlane[1][0] * PX - bestPlane[1][1] * PY - d) * 1. /bestPlane[1][2]

                surfaces_delaunay = Delaunay(np.stack((PX,PY)).T)
                p_triangles=surfaces_delaunay.simplices
                #orientation = 'vertical'

            # Get area and point desity of the planes
            plane_area = getPointCloudArea(np.stack((PX,PY,PZ)).T,p_triangles)
            plane_properties.append([
                plane_area,
                len(bestPoints)/plane_area
            ])

            # Add final candidate plane to list of planes
            planes.append([
                bestPlane,
                np.concatenate((np.stack((PX,PY,PZ)).T,depletable_pc[bestPoints][:,3:6]),axis=1),
                p_triangles,
                depletable_pc[sample]],
            )
            #print("Added a plane.")

            # Remove points of final plane from remaining candidate points
            depletable_pc = np.delete(depletable_pc,bestPoints,0)

    #print(len(pointCloud),"points, Time to extract",len(planes),"planes: ", time.time() - t0)
    #print(len(depletable_pc),"remaining points")
    #print(zero_normals,"with zero normals")
    #print(added_zero_normals,"added zero normals")
    #planeCollection.append([object_name, planes])
    plane_properties = np.asarray(plane_properties)
    return planes, plane_properties


# In[ ]:


def GetLocalizedPlanes(pointCloud, radius, i_point = -1):

    # Extracting a sample point from the point cloud
    if (i_point<0):
        i_point = np.random.choice(len(pointCloud))
    
    point = pointCloud[i_point,:]
    #print("LOCAL",point)
    
    # If normal is zero, pick another point
    while LA.norm(point[3:]) == 0:
        point = pointCloud[np.random.choice(len(pointCloud)),:]
    
    d = -point[:3].dot(point[3:])
    genPoint = np.asarray([point])

    # Get all nearby points from the chosen point based on the radius.
    genNearbyPoints = pointCloud[:,:3]-genPoint[:,:3]
    genPoints = np.delete(pointCloud,np.where(LA.norm(genNearbyPoints,axis=1)>radius)[0],axis=0)
    #print("Gen Points shape",genPoints.shape)
    
    PX = genPoints[:,0]
    PY = genPoints[:,1]
    PZ = genPoints[:,2]        

    # Generalize the bestPlane to one of the surface-axis to get a 3-point Delaunay match
    phi = math.fabs(point[4]* 1./LA.norm(point[3:])) # y/r of the normal vector

    if math.degrees(math.acos(phi)) < 45 : # arc-cos(y/r) = phi < 45 --> horizontal
        # use floor (plane x-z) as origin mesh
        PY = (-point[3] * PX - point[5] * PZ - d) * 1. /point[4]

        surfaces_delaunay = Delaunay(np.stack((PX,PZ)).T)
        p_triangles=surfaces_delaunay.simplices
        #orientation = 'horizontal'
    else:
        #use vertical wall x-y as origin mesh
        PZ = (-point[3] * PX - point[4] * PY - d) * 1. /point[5]

        surfaces_delaunay = Delaunay(np.stack((PX,PY)).T)
        p_triangles=surfaces_delaunay.simplices
        #orientation = 'vertical'

    p_pointCloud = np.concatenate((np.stack((PX,PY,PZ)).T,np.repeat([point[3:]],len(genPoints),axis=0)),
                                  axis=1)
    
    return p_pointCloud, p_triangles, genPoint


