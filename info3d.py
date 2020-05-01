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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


"""
As the name suggests, this returns the indices (as well as the distances)
of the nearest neighbors in 3D euclidean space.
"""
def getEuclideanNearestNeighbours(point_cloud, n=2, thresh_max=20):

    point_cloud_coord = point_cloud[:,:3]

    try:
        # Transform it to get a new matrix that repeats the points to prepare for nearest neighbour matching.
        desired_shape = (point_cloud_coord.shape[0],point_cloud_coord.shape[0],point_cloud_coord.shape[1])
        pointCloudMatrixed = np.repeat(point_cloud_coord,point_cloud_coord.shape[0],0)
        pointCloudMatrixed = np.reshape(pointCloudMatrixed,desired_shape)

        pointCloudMatrixedT = pointCloudMatrixed.transpose(1,0,2)

        difference = LA.norm(pointCloudMatrixedT - pointCloudMatrixed,axis=2)
    except Exception as e1:
        print("Error getting neighbors:", e1)
        return
        
    matches = np.argsort(np.clip(difference,0,thresh_max))
    
    nearestneighbours = matches[:,1:n+1]
    difference.sort(axis=1)
    
    return nearestneighbours, difference[:,1:n+1]

"""
To get the area:
1. Extract vectors of 2 of the three sides of one triangle.
2. Get half the magnitude of their cross product.
"""
def getTriangleAreas(_point_cloud, _triangles):

    v1 = _point_cloud[_triangles[:,0],:3] - _point_cloud[_triangles[:,1],:3]
    v2 = _point_cloud[_triangles[:,2],:3] - _point_cloud[_triangles[:,1],:3]
    area = np.abs(LA.norm(np.cross(v1,v2), axis = 1))*0.5
        
    return area

"""
Sum the areas of the triangles.
"""
def getPointCloudArea(_point_cloud, _triangles):
    
    return np.sum(getTriangleAreas(_point_cloud, _triangles))

def OLDgetQuantizedPointCloudOnly(_point_cloud,scale = 5, verbose = False):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    
    n_pointCloud = []
    
    np_pointCloud[:,:3] = np_pointCloud[:,:3]*scale+2.5*scale

    p_model = np.zeros((5*scale,5*scale,5*scale)) # i, x, y, z space.
    
    if verbose:
        print(" OLD: Model shape", p_model.shape)
        
    for point in np_pointCloud:
        point[:3] = np.clip(point[:3],0,np.min(p_model.shape)-1)
        if p_model[int(point[0]),int(point[1]),int(point[2])] == 0:
            p_model[int(point[0]),int(point[1]),int(point[2])] = int(len(n_pointCloud))
            n_pointCloud.append(point)

    n_pointCloud = np.asarray(n_pointCloud)
    
    if verbose:
        print(" OLD: Successfully created the quantization:",np.asarray(_point_cloud).shape,"to",n_pointCloud.shape)
    
    n_pointCloud[:,:3] = n_pointCloud[:,:3]*1.0/scale-2.5
    
    return n_pointCloud

"""
We sub sample the point cloud given a [sampling-]scale value.
1. We create a 3D matrix that can enclose the given point cloud.
2. The matrix represents bins whose inter-element distance is 1/scale.
"""
def getQuantizedPointCloudOnly(_point_cloud,scale = 5, verbose = False):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    
    n_pointCloud = []
    
    # Compute the bounds and translation of the given point cloud.
    upper_bound = np.ceil(np.max(np_pointCloud[:,:3]))+1
    lower_bound = np.floor(np.min(np_pointCloud[:,:3]))-1
    maximum_span = int(max(upper_bound-lower_bound,5))
    bias = (upper_bound + lower_bound)*0.5
    
    if verbose:
        print(" Upper bound",np.max(np_pointCloud[:,:3]),upper_bound)
        print(" Lower bound",np.min(np_pointCloud[:,:3]),lower_bound)
        print(" Bound space",maximum_span)#,bias)
    
    # Translate the point cloud so center is origin.
    np_pointCloud[:,:3] += 0.5*maximum_span - bias
    np_pointCloud[:,:3] *= scale
    #np_pointCloud[:,:3] += 0.5*maximum_space*scale
    
    if verbose:
        new_upper_bound = np.ceil(np.max(np_pointCloud[:,:3]))+1
        new_lower_bound = np.floor(np.min(np_pointCloud[:,:3]))-1
        print(" New Upper bound",np.max(np_pointCloud[:,:3]),new_upper_bound)
        print(" New Lower bound",np.min(np_pointCloud[:,:3]),new_lower_bound)
        #print(" New Bound space",maximum_span)
        
    # Create an empty 3D array [p_model] that can enclose the given point cloud.
    # The dimensions of the array is dictated by the bounds and the scale.
    # Each element in the 3D array represents a bin with size 1/scale.
    p_model = np.zeros((maximum_span*2*scale,maximum_span*2*scale,maximum_span*2*scale)) # i, x, y, z space.
    if verbose:
        print(" New: Model shape", p_model.shape)
        
    # Iterate the point cloud and fill the empty bins with the first point
    # that fits. The resulting accumulate points represent a down-samnpled point cloud.
    for point in np_pointCloud:
        point[:3] = np.clip(point[:3],0,np.min(p_model.shape)-1)
        if p_model[int(point[0]),int(point[1]),int(point[2])] == 0:
                p_model[int(point[0]),int(point[1]),int(point[2])] = int(len(n_pointCloud))
                n_pointCloud.append(point)

    n_pointCloud = np.asarray(n_pointCloud)

    # Retranslate the down-sampled point cloud.
    n_pointCloud[:,:3] = n_pointCloud[:,:3]*1.0/scale-0.5*maximum_span+bias
    
    if verbose:
        print(" New: Successfully created the quantization:",np.asarray(np_pointCloud).shape,"to",n_pointCloud.shape)

        #p_model = np.zeros((int(maximum_space*scale/2),int(maximum_space*scale/2),int(maximum_space*scale/2))) # i, x, y, z space.
    
    return n_pointCloud

"""
We sub sample the point cloud given a [sampling-]scale value.
1. We create a 3D matrix that can enclose the given point cloud.
2. The matrix represents bins whose inter-element distance is 1/scale.
"""
def getQuantizedPointCloud(_point_cloud, _triangles,scale = 20,verbose = False):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    np_triangles = np.asarray(_triangles)#-vertices_length
    
    n_triangles = np.copy(_triangles)
    n_pointCloud = []
    
    upper_bound = np.ceil(np.max(np_pointCloud[:,:3]))+1
    lower_bound = np.floor(np.min(np_pointCloud[:,:3]))-1
    maximum_span = int(max(upper_bound-lower_bound,5))
    bias = (upper_bound + lower_bound)*0.5
    
    if verbose:
        print("Upper bound",np.max(np_pointCloud[:,:3]),upper_bound)
        print("Lower bound",np.min(np_pointCloud[:,:3]),lower_bound)
        print("Bound space",maximum_span)
    
    np_pointCloud[:,:3] += bias + 0.5*maximum_span
    np_pointCloud[:,:3] *= scale
    
    p_model = np.zeros((maximum_span*2*scale,maximum_span*2*scale,maximum_span*2*scale)) # i, x, y, z space.
    
    if verbose:
        print(" New: Model shape", p_model.shape)
        
    try:
        for i, point in enumerate(np_pointCloud):
            if p_model[int(point[0]),int(point[1]),int(point[2])] == 0:
                p_model[int(point[0]),int(point[1]),int(point[2])] = int(len(n_pointCloud))
                n_pointCloud.append(point)
            np.place(n_triangles,
                     n_triangles==i,
                     p_model[int(point[0]),int(point[1]),int(point[2])])

        #cq_pointCloud = np.transpose(np.nonzero(p_model))*1.0/scale-0.5*maximum_space
        n_pointCloud = np.asarray(n_pointCloud)

        n_triangles = np.delete(n_triangles,
                                np.where(n_triangles[:,1]==n_triangles[:,2]),0)
        n_triangles = np.delete(n_triangles,
                                np.where(n_triangles[:,1]==n_triangles[:,0]),0)

        n_pointCloud[:,:3] = n_pointCloud[:,:3]*1.0/scale-(bias + 0.5*maximum_span)
        
        if verbose:
            print("Successfully created the quantization:",np.asarray(_point_cloud).shape,"to",n_pointCloud.shape)
            
    except Exception as ex:
        print("Failed quantization",ex)
        #n_pointCloud = np_pointCloud[::scale]
        #p_model = np.zeros((int(maximum_space*scale/2),int(maximum_space*scale/2),int(maximum_space*scale/2))) # i, x, y, z 
        
    #print(cq_pointCloud.shape,n_pointCloud.shape)
    return n_pointCloud, n_triangles, p_model


def getQuantizedPointCloudOnlyTEST(_point_cloud,scale = 5, verbose = False):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    
    round_new_pointcloud_only = (1.0/scale)*np.around(scale*_point_cloud[:,:3],decimals=0)
    unq_round_new_pointcloud, indices = np.unique(round_new_pointcloud_only,axis = 0, return_index = True)
    
    #print()
    n_pointCloud = np.hstack((unq_round_new_pointcloud, np_pointCloud[indices,3:]))
        
    return n_pointCloud


def OLDgetQuantizedPointCloud(_point_cloud, _triangles,scale = 20):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    np_triangles = np.asarray(_triangles)#-vertices_length
    
    n_triangles = np.copy(_triangles)
    n_pointCloud = []
    
    np_pointCloud[:,:3] = np_pointCloud[:,:3]*scale+2.5*scale

    p_model = np.zeros((5*scale,5*scale,5*scale)) # i, x, y, z space.
    
    for i, point in enumerate(np_pointCloud):
        if p_model[int(point[0]),int(point[1]),int(point[2])] == 0:
            p_model[int(point[0]),int(point[1]),int(point[2])] = int(len(n_pointCloud))
            n_pointCloud.append(point)
        np.place(n_triangles,
                 n_triangles==i,
                 p_model[int(point[0]),int(point[1]),int(point[2])])

    cq_pointCloud = np.transpose(np.nonzero(p_model))*1.0/scale-2.5
    n_pointCloud = np.asarray(n_pointCloud)
    
    n_triangles = np.delete(n_triangles,
                            np.where(n_triangles[:,1]==n_triangles[:,2]),0)
    n_triangles = np.delete(n_triangles,
                            np.where(n_triangles[:,1]==n_triangles[:,0]),0)
    
    n_pointCloud[:,:3] = n_pointCloud[:,:3]*1.0/scale-2.5
    
    #print(cq_pointCloud.shape,n_pointCloud.shape)
    return n_pointCloud, n_triangles, p_model

#centeredPointCollection # object_name, *centered_vertices, polygons or triangles

def getQuantizedCenteredPointCollection(centeredPointCollection_,resolution = 20):

    quantizedCenteredPointCollection = []

    t0 = time.time()
    for object_name, pointCloud, triangles in centeredPointCollection_:
        np_pointCloud = np.asarray(np.copy(pointCloud))
        np_triangles = np.asarray(triangles)#-vertices_length

        n_pointCloud, n_triangles, p_ = getQuantizedPointCloud(np_pointCloud,np_triangles,scale=resolution)

        quantizedCenteredPointCollection.append([object_name,
                                                 n_pointCloud,
                                                 n_triangles])

    #print("down-sampling factor:",resolution,", Time to down-sample spaces",time.time() - t0)
    
    return quantizedCenteredPointCollection

# Rotations!

def rotatePointCollection(point_collection, theta = (np.pi)/3, axis = 1, verbose = False):# Angle of rotation
    # default theta is 60 degrees
    
    rotatedPointCollection = [] # object_number, *centered_vertices, polygons or triangles
    
    axis_index = np.delete([0,1,2],axis)
    
    t0 = time.time()
    for object_name, pointCloud, triangles in point_collection:
        n_pointCloud = np.copy(pointCloud) # x, y, z
        n_triangles = np.copy(triangles)#-vertices_length

        # Rotate about the vertical axis (y, or axis=1)
        n_pointCloud[:,axis_index[0]] = pointCloud[:,axis_index[0]]*np.cos(theta) - pointCloud[:,axis_index[1]]*np.sin(theta)
        n_pointCloud[:,axis_index[1]] = pointCloud[:,axis_index[0]]*np.sin(theta) + pointCloud[:,axis_index[1]]*np.cos(theta)
        n_pointCloud[:,axis_index[0]+3] = pointCloud[:,axis_index[0]+3]*np.cos(theta) - pointCloud[:,axis_index[1]+3]*np.sin(theta)
        n_pointCloud[:,axis_index[1]+3] = pointCloud[:,axis_index[0]+3]*np.sin(theta) + pointCloud[:,axis_index[1]+3]*np.cos(theta)

        rotatedPointCollection.append([object_name,
                                        n_pointCloud,
                                        n_triangles
                                       ])

    #rotatedPointCollection = np.asarray(rotatedPointCollection)
    if verbose:
        print("Time to rotate",len(rotatedPointCollection),"spaces",time.time() - t0)
    
    return rotatedPointCollection

def getLocalizedPlanes(pointCloud, radius):

    # Extracting a sample point from the point cloud
    point = pointCloud[np.random.choice(len(pointCloud)),:]
    
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
    
    return p_pointCloud, p_triangles


def getPartialPointCloudEbE(
    pointCloud,
    triangles,
    radius = 1 #of geodesic sphere to be sampled
):
    # This still needs to be debugged. There are occassions when it would infintely loop within the sampling process.

    triangle_indices = np.arange(len(triangles))

    nbrs = NearestNeighbors(n_neighbors=len(pointCloud)-1, algorithm='brute').fit(pointCloud[:,:3])
    distances, indices = nbrs.kneighbors(pointCloud[:,:3])
    # 1.a
    # Process ELEMENT-BY-ELEMENT randomly

    partial_pointcloud = []
    partial_triangles = []

    index_list = []
    iterations = 0

    t0= time.time()

    # Get a starting vertex
    get_new_triangle = np.random.choice(triangle_indices)
    #print("origin-triangle index",get_new_triangle,"(Remember, the triangle indices can be more than the point population.)")
    vertex = triangles[get_new_triangle,1]
    #print("origin-vertex",vertex)
    original_vertex = pointCloud[vertex]

    # Get the acceptable neighbors of the chosen point
    acceptable_point_neighbors = indices[vertex,np.where(distances[vertex]<radius)[0]]
    depletable_neighbors = np.copy(acceptable_point_neighbors)
    #print(len(acceptable_point_neighbors),"neighbors")
    acceptable_neighbor_distances = distances[vertex,np.where(distances[vertex]<radius)[0]]

    # While distance of points to be addded are less than 1,
    # get the indices of the connected items.
    prev_length = 0
    stopcount = 0

    while stopcount < int(0.9*len(acceptable_point_neighbors)):#depletable_neighbors.size != 0:

        included_triangle_indices = np.concatenate((
            np.where(triangles[:,0]==vertex)[0],
            np.where(triangles[:,1]==vertex)[0],
            np.where(triangles[:,2]==vertex)[0]),0)

        local_triangles = triangles[included_triangle_indices]    
        included_vertices = np.unique(local_triangles.flatten('C'))

        for in_vertex in np.asarray(included_vertices):
            if (len(index_list) > 0) and (in_vertex in index_list):
                # check if the current in_vetex has already been added
                # if yes, do not add to point_list and use the recorded index in replacing index in partial triangles

                np.place(local_triangles,local_triangles==in_vertex,index_list.index(in_vertex))
                continue

            if in_vertex in depletable_neighbors:
                # before adding to the partial lists, check if in the acceptable neighbor list.

                partial_pointcloud.append(pointCloud[in_vertex])
                depletable_neighbors = np.delete(depletable_neighbors,np.where(depletable_neighbors==in_vertex),0)
                index_list.append(in_vertex)
                np.place(local_triangles,local_triangles==in_vertex,len(partial_pointcloud)-1)
            else:
                # if not in acceptable, remove the triangle that has that in_vertex

                local_triangles = np.delete(local_triangles,np.unique(np.where(local_triangles==in_vertex)[0]),0)

        if len(partial_triangles) > 0:
            partial_triangles = np.concatenate((partial_triangles,local_triangles),0)
            partial_triangles = np.asarray(partial_triangles)
        else:
            partial_triangles = local_triangles

        if prev_length == len(partial_triangles):         
            stopcount +=1
        else:
            stopcount = 0
        prev_length = len(partial_triangles)

        if depletable_neighbors.size != 0:
            vertex = np.random.choice(depletable_neighbors)
        #print(vertex)
#    print("Time to get partial",time.time()-t0)
#    print("Partial triangles",len(partial_triangles))
#    print("Partial pointcloud",len(partial_pointcloud))

    partial_pointcloud = np.asarray(partial_pointcloud)
#    print("Remaining points",len(depletable_neighbors))
#    print(stopcount)    
    return partial_pointcloud, partial_triangles, original_vertex

def OLDgetPartialPointCloud(
    pointCloud,
    triangles,
    radius = 1
):

    triangle_indices = np.arange(len(triangles))

    nbrs = NearestNeighbors(n_neighbors=len(pointCloud)-1, algorithm='brute').fit(pointCloud[:,:3])
    distances, indices = nbrs.kneighbors(pointCloud[:,:3])
    # 1.b
    # list EVERYTHING, then update one-by-one

    partial_pointcloud = []
    partial_triangles = []
    
    #while len(triangle_indices)>0:
    t0 = time.time()
    # Get a starting vertex
    get_new_triangle = np.random.choice(triangle_indices)
    #print("origin-triangle index",get_new_triangle,"(Remember, the triangle indices can be more than the point population.)")
    vertex = triangles[get_new_triangle,1]
    #print("origin-vertex",vertex)
    original_vertex = pointCloud[vertex]

    # Get the acceptable neighbors of the chosen point
    acceptable_point_neighbors = indices[vertex,np.where(distances[vertex]<radius)[0]]
    depletable_triangles = np.copy(triangles)
    #print(len(acceptable_point_neighbors),"neighbors")
    acceptable_neighbor_distances = distances[vertex,np.where(distances[vertex]<radius)[0]]

    # While distance of points to be addded are less than 1,
    # get the indices of the connected items.
    prev_length = 0
    stopcount = 0

    # Get all triangles with the index in the neighbor list
    for vertex in acceptable_point_neighbors:    
        included_triangle_indices = np.concatenate((
            np.where(depletable_triangles[:,0]==vertex)[0],
            np.where(depletable_triangles[:,1]==vertex)[0],
            np.where(depletable_triangles[:,2]==vertex)[0]),0)

        local_triangles = depletable_triangles[np.unique(included_triangle_indices)]
        depletable_triangles = np.delete(depletable_triangles,np.unique(included_triangle_indices),0)

        if len(partial_triangles) > 0:
            partial_triangles = np.concatenate((partial_triangles,local_triangles),0)
            partial_triangles = np.asarray(partial_triangles)
        else:
            partial_triangles = local_triangles

    included_vertices = np.unique(partial_triangles.flatten('C'))
    index_list = []

    for in_vertex in included_vertices:
        if in_vertex in acceptable_point_neighbors:
            # before adding to the partial lists, check if in the acceptable neighbor list.

            partial_pointcloud.append(pointCloud[in_vertex])
            #depletable_neighbors = np.delete(depletable_neighbors,np.where(depletable_neighbors==in_vertex),0)
            index_list.append(in_vertex)
            np.place(partial_triangles,partial_triangles==in_vertex,len(partial_pointcloud)-1)
        else:
            # if not, remove associated triangles
            partial_triangles = np.delete(partial_triangles,np.unique(np.where(partial_triangles==in_vertex)[0]),0)

    partial_pointcloud = np.asarray(partial_pointcloud)
#    print("Remaining points",len(depletable_neighbors))
#    print(stopcount)    
    return partial_pointcloud, partial_triangles, original_vertex

# pointCollection --> object_number, vertices, vertex normals, polygons
#print(len(pointCollection),len(pointCollection[0]))

def getPartialPointCloud(
    pointCloud,
    triangles,
    radius = 1,
    vertex = [],
    verbose = False,
    return_indices = False
):

    triangle_indices = np.arange(len(triangles))
    
    if vertex == []:
        get_new_triangle = np.random.choice(triangle_indices)
    #rint("origin-triangle index",get_new_triangle,"(Remember, the triangle indices can be more than the point population.)")
        vertex = triangles[get_new_triangle,1]
        if verbose: print(" Computed origin-vertex",vertex)
    
    original_vertex = pointCloud[np.clip(vertex,0,len(pointCloud))]
    # makes sure that we don't get a point beyond the pC size

    nbrs = NearestNeighbors(n_neighbors=len(pointCloud)-1, algorithm='brute').fit(pointCloud[:,:3])
    distances, indices = nbrs.kneighbors(np.asarray([pointCloud[vertex,:3]]))
    # 1.b
    # list EVERYTHING, then update one-by-one

    partial_pointcloud = []
    partial_triangles = []
    
    #while len(triangle_indices)>0:
    t0 = time.time()
    # Get a starting vertex
    

    # Get the acceptable neighbors of the chosen point
    acceptable_point_neighbors = indices[0,np.where(distances[0]<radius)[0]]
    depletable_triangles = np.copy(triangles)
    #print(len(acceptable_point_neighbors),"neighbors")
    acceptable_neighbor_distances = distances[0,np.where(distances[0]<radius)[0]]

    # While distance of points to be addded are less than 1,
    # get the indices of the connected items.
    prev_length = 0
    stopcount = 0

    # Get all triangles with the index in the neighbor list
    for vertex in acceptable_point_neighbors:    
        included_triangle_indices = np.concatenate((
            np.where(depletable_triangles[:,0]==vertex)[0],
            np.where(depletable_triangles[:,1]==vertex)[0],
            np.where(depletable_triangles[:,2]==vertex)[0]),0)

        local_triangles = depletable_triangles[np.unique(included_triangle_indices)]
        depletable_triangles = np.delete(depletable_triangles,np.unique(included_triangle_indices),0)

        if len(partial_triangles) > 0:
            partial_triangles = np.concatenate((partial_triangles,local_triangles),0)
            partial_triangles = np.asarray(partial_triangles)
        else:
            partial_triangles = local_triangles

    included_vertices = np.unique(partial_triangles.flatten('C'))
    index_list = []

    for in_vertex in included_vertices:
        if in_vertex in acceptable_point_neighbors:
            # before adding to the partial lists, check if in the acceptable neighbor list.

            partial_pointcloud.append(pointCloud[in_vertex])
            #depletable_neighbors = np.delete(depletable_neighbors,np.where(depletable_neighbors==in_vertex),0)
            index_list.append(in_vertex)
            np.place(partial_triangles,partial_triangles==in_vertex,len(partial_pointcloud)-1)
        else:
            # if not, remove associated triangles
            partial_triangles = np.delete(partial_triangles,np.unique(np.where(partial_triangles==in_vertex)[0]),0)

    partial_pointcloud = np.asarray(partial_pointcloud)
#    print("Remaining points",len(depletable_neighbors))
#    print(stopcount) 

    if return_indices:
        return partial_pointcloud, partial_triangles, original_vertex, included_vertices
    else:
        return partial_pointcloud, partial_triangles, original_vertex

def getDelaunayTriangles(
    plane_params, # bestPlane, i.e. containing reference vertex and normal
    pointcloud,
    triangle_area_threshold = 0.1,
    phi = np.nan,
    strict = False,
    verbose = False
):
    d = -plane_params[0].dot(plane_params[1])
    
    PX = pointcloud[:,0]
    PY = pointcloud[:,1]
    PZ = pointcloud[:,2]  
    
    p_triangles = []

    # Generalize the bestPlane to one of the surface-axis to get a 3-point Delaunay match
    if np.isnan(phi):
        phi = math.fabs(plane_params[1][1]* 1./LA.norm(plane_params[1])) # y/r

    try:
        if math.degrees(math.acos(phi)) < 45 : # arc-cos(y/r) = phi < 45 --> horizontal
            # use floor (plane x-z) as origin mesh
            #PY = (-plane_params[1][0] * PX - plane_params[1][2] * PZ - d) * 1. /plane_params[1][1]

            surfaces_delaunay = Delaunay(np.stack((PX,PZ)).T)
            p_triangles=surfaces_delaunay.simplices
            #orientation = 'horizontal'
        else:
            #use vertical wall x-y as origin mesh
            #PZ = (-plane_params[1][0] * PX - plane_params[1][1] * PY - d) * 1. /plane_params[1][2]

            surfaces_delaunay = Delaunay(np.stack((PX,PY)).T)
            p_triangles=surfaces_delaunay.simplices
            #orientation = 'vertical'
    except:
        # Error in getting Delaunay triangles.
        pass
        
    if strict and len(p_triangles) != 0:
        v1 = pointcloud[p_triangles[:,0],:3] - pointcloud[p_triangles[:,1],:3]
        v2 = pointcloud[p_triangles[:,2],:3] - pointcloud[p_triangles[:,1],:3]
        area = np.abs(LA.norm(np.cross(v1,v2), axis = 1))*0.5
        
        v1_l = LA.norm(v1, axis = 1)
        v2_l = LA.norm(v2, axis = 1)
        
        if verbose:
            print("  getDelaunayTriangles: threshold = {:.3f}, mean area = {:.3f} ({:.3f})".format( #v1: ({.3f},{:.3f}); v2: ({:.3f},{:.3f})".format(
                triangle_area_threshold,
                #np.amin(LA.norm(v1, axis = 1)),
                #np.amax(LA.norm(v1, axis = 1)),
                #np.amin(LA.norm(v2, axis = 1)),
                #np.amax(LA.norm(v2, axis = 1))
                np.nanmean(area),
                np.nanstd(area)
            ))
            
        # Pick the triangles which are legitimately small enough. Too large means this triangles should not exist.
        # --> area < theshold (i.e. 0.1)
        # --> length of two sides should be less than twice the threshold to maintian the idea of bh/2.
        p_triangles = p_triangles[
            np.intersect1d(
                np.intersect1d(
                    np.where(area < (np.nanmean(area) + 2*np.nanstd(area)))[0],
                    np.where(v1_l < (np.nanmean(v1_l) + 2*np.nanstd(v1_l)))[0]
                ),
                np.where(v2_l < (np.nanmean(v2_l) + 2*np.nanstd(v2_l)))[0]
            )
        ]
        #p_pointCloud = pointcloud[np.unique(p_triangles.flatten())]

    return p_triangles

def getRansacPlanesOld(
    pointCloud,
    triangles,
    planes_to_find = 30, # number of planes to find
    threshold = 0.05,     # the point-plane distance threshold
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
                if LA.norm(point[3:]) == 0:
                    zero_normals += 1
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        # only add a point with zero_normal if very close to the plane
                        added_zero_normals += 1
                        testPoints.append(i)
                if abs(np.dot(testPlane[1],point[3:])/(LA.norm(testPlane[1])*LA.norm(point[3:]))) > (1-20*threshold):
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
                #np.concatenate((np.stack((PX,PY,PZ)).T,depletable_pc[bestPoints][:,3:6]),axis=1),
                np.concatenate((np.stack((PX,PY,PZ)).T,np.repeat([bestPlane[1]],len(PX),axis = 0)),axis=1),
                p_triangles,
                depletable_pc[sample]],
            )

            # Remove points of final plane from remaining candidate points
            depletable_pc = np.delete(depletable_pc,bestPoints,0)

    #print(len(pointCloud),"points, Time to extract",len(planes),"planes: ", time.time() - t0)
    #print(len(depletable_pc),"remaining points")
    #print(zero_normals,"with zero normals")
    #print(added_zero_normals,"added zero normals")
    #planeCollection.append([object_name, planes])
    plane_properties = np.asarray(plane_properties)
    return planes, plane_properties

def getRansacPlanes(
    pointCloud,
    #triangles,
    planes_to_find = 30, # number of planes to find
    threshold = 0.05,     # the point-plane distance threshold
    trials = 100,       # the number of RANSAC trials
    #strict = False
    #plane_group = 200     # number of nearby points per plane
):

    density = 0
    
    """
    if strict:
        v1 = pointCloud[triangles[:,0],:3] - pointCloud[triangles[:,1],:3]
        v2 = pointCloud[triangles[:,2],:3] - pointCloud[triangles[:,1],:3]
        area = np.abs(LA.norm(np.cross(v1,v2), axis = 1))*0.5
        density = len(pointCloud)/np.nansum(area)
    """
        
    planeCollection = []
    test_max = 10

    t0 = time.time()

    planes = []
    #plane_properties = []

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
                if LA.norm(testPlane[1])*LA.norm(point[3:]) == 0:
                    zero_normals += 1
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        # only add a point with zero_normal if very close to the plane
                        added_zero_normals += 1
                        testPoints.append(i)
                if abs(np.dot(testPlane[1],point[3:])/(LA.norm(testPlane[1])*LA.norm(point[3:]))) > max(0,(1-20*threshold)):
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
            
            phi = math.fabs(bestPlane[1][1]* 1./LA.norm(bestPlane[1])) # y/r
            
            if math.degrees(math.acos(phi)) < 45 : # arc-cos(y/r) = phi < 45 --> horizontal
                # use floor (plane x-z) as origin mesh
                NPY = (-bestPlane[1][0] * PX - bestPlane[1][2] * PZ - d) * 1. /bestPlane[1][1]
                #orientation = 'horizontal'
                acceptable_move = np.where(np.abs(PY-NPY)<10*threshold)[0]
                NPY = NPY[acceptable_move]
                NPX = PX[acceptable_move]
                NPZ = PZ[acceptable_move]
            else:
                #use vertical wall x-y as origin mesh
                NPZ = (-bestPlane[1][0] * PX - bestPlane[1][1] * PY - d) * 1. /bestPlane[1][2]
                #orientation = 'vertical'
                acceptable_move = np.where(np.abs(PZ-NPZ)<10*threshold)[0]
                NPZ = NPZ[acceptable_move]
                NPX = PX[acceptable_move]
                NPY = PY[acceptable_move]

            if len(NPX) == 0:
                if verbose: print("Emptied after strict acceptable points",PX.shape,NPX.shape)
                continue
                
            # Add final candidate plane to list of planes
            planes.append([
                [bestPlane ,phi, density],
                #np.concatenate((np.stack((PX,PY,PZ)).T,depletable_pc[bestPoints][:,3:6]),axis=1),
                np.concatenate((np.stack((NPX,NPY,NPZ)).T,np.repeat([bestPlane[1]],len(NPX),axis = 0)),axis=1),
                depletable_pc[sample]
            ])

            # Remove points of final plane from remaining candidate points
            depletable_pc = np.delete(depletable_pc,bestPoints,0)

    #print(len(pointCloud),"points, Time to extract",len(planes),"planes: ", time.time() - t0)
    #print(len(depletable_pc),"remaining points")
    #print(zero_normals,"with zero normals")
    #print(added_zero_normals,"added zero normals")
    #planeCollection.append([object_name, planes])
    #plane_properties = np.asarray(plane_properties)
    return planes

def getLOCALIZEDRansacPlanes(
    pointCloud,
    planes_to_find = 1, # number of planes to find
    threshold = 0.05,     # the point-plane distance threshold
    trials = 100,       # the number of RANSAC trials
    original_vertex = [],
    verbose = False
    #plane_group = 200     # number of nearby points per plane
):
    
    #density = 0

    planeCollection = []
    test_max = 10

    t0 = time.time()

    planes = []
    #plane_properties = []

    depletable_pc = np.copy(pointCloud)
    #print("true points",len(depletable_pc))

    zero_normals = 0
    added_zero_normals = 0
    
    # Getting the plane of the hitPoint
    #sample = np.random.randint(len(depletable_pc))

    firstPlane = [original_vertex[:3],original_vertex[3:]]
                 #np.cross(depletable_pc[near1,:3]-depletable_pc[sample,:3],
                 #        depletable_pc[near2,:3]-depletable_pc[sample,:3]
                 #        )]
    
    firstPoints = []
    d = -firstPlane[0].dot(firstPlane[1])
    
    for i, point in enumerate(depletable_pc): #[neighbours[sample]]
        if LA.norm(firstPlane[1])*LA.norm(point[3:]) == 0:
            zero_normals += 1
            if abs((np.dot(firstPlane[1],point[:3])+d)*1.0/LA.norm(firstPlane[1],ord = 2)) < threshold:
                # only add a point with zero_normal if very close to the plane
                added_zero_normals += 1
                firstPoints.append(i)
        if abs(np.dot(firstPlane[1],point[3:])/(LA.norm(firstPlane[1])*LA.norm(point[3:]))) > max(0,(1-20*threshold)):
            # if normals are close accept if near to the candidate plane
            if abs((np.dot(firstPlane[1],point[:3])+d)*1.0/LA.norm(firstPlane[1],ord = 2)) < threshold:
                firstPoints.append(i)
                
    d = -firstPlane[0].dot(firstPlane[1])
    
    if verbose: 
        print(depletable_pc.shape)
        print("First Points",len(firstPoints), firstPlane)
    PX = depletable_pc[firstPoints][:,0]
    PY = depletable_pc[firstPoints][:,1]
    PZ = depletable_pc[firstPoints][:,2]        
            
    phi = math.fabs(firstPlane[1][1]* 1./LA.norm(firstPlane[1])) # y/r

    # Add final candidate plane to list of planes
    planes.append([
        [firstPlane ,phi, 0],
        #np.concatenate((np.stack((PX,PY,PZ)).T,depletable_pc[bestPoints][:,3:6]),axis=1),
        np.concatenate((np.stack((PX,PY,PZ)).T,np.repeat([firstPlane[1]],len(PX),axis = 0)),axis=1),
        original_vertex
    ])
    
    # Remove points of first plane from remaining candidate points
    depletable_pc = np.delete(depletable_pc,firstPoints,0)
    
    all_planes = planes
    
    if planes_to_find - 1 > 0:
        additional_planes = getRansacPlanes(
            pointCloud = depletable_pc,
            planes_to_find = planes_to_find - 1,
            threshold = threshold,
            trials = trials
        )
        
        if len(additional_planes) != 0:        
            all_planes = planes + additional_planes

    return all_planes

def updatePlanesWithSubsumption(
    new_pointCloud,
    existing_pointCloud,
    planes_to_find = 30, # number of planes to find
    threshold = 0.05,     # the point-plane distance threshold
    trials = 100,
    verbose = False
):

    _, unq_idx = np.unique(
        np.round(existing_pointCloud[:,:3],decimals = 3),
        axis = 0,
        return_index=True
    )
    existing_pointCloud = existing_pointCloud[unq_idx]
    #existing_pointCloud[:,:3] = np.round(existing_pointCloud[:,:3],decimals = 5)
    #existing_pointCloud[:,3:] = np.round(existing_pointCloud[:,3:],decimals = 5)
    new_pointCloud = np.unique(new_pointCloud, axis = 0)
    #existing_pointCloud = np.unique(existing_pointCloud, axis = 0)
    
    depletable_pc = np.copy(new_pointCloud)
    depletable_existing = np.copy(existing_pointCloud)

    if verbose:
        print("New point cloud",depletable_pc.shape)
        print("Existing point cloud",depletable_existing.shape)
    
    planeCollection = []

    test_max = 10

    t0 = time.time()

    planes = []
    #plane_properties = []

    zero_normals = 0
    added_zero_normals = 0

    existing_normals = np.unique(np.round(existing_pointCloud[:,3:],decimals = 5), axis = 0)
    
    if verbose:
        print(" ",existing_normals.shape,"Normals")

    # Getting the planes
    for i_plane in np.arange(planes_to_find):
        #pass
        best_pc = []
        bestPoints = []
        trial = 0
        t1 = time.time()

        matched_idx_from_existing = []

        #neighbours, distance = getEuclideanNearestNeighbours(depletable_pc,3)

        #print(object_name,i_plane,len(depletable_pc))
        if len(depletable_pc) < 3:
            continue

        for normal in existing_normals:

            existing_pc_idx = np.where(np.round(depletable_existing[:,3:], decimals = 5) == normal)[0]
            
            if len(existing_pc_idx) == 0:
                continue

            sample = np.random.choice(existing_pc_idx)

            testPlane = [depletable_existing[sample,:3],depletable_existing[sample,3:]]

            #print(normal, sample, testPlane)

            testPoints = []
            d = -testPlane[0].dot(testPlane[1])
            for i, point in enumerate(depletable_pc): #[neighbours[sample]]
                if (LA.norm(testPlane[1])*LA.norm(point[3:])) == 0:
                    zero_normals += 1
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        # only add a point with zero_normal if very close to the plane
                        added_zero_normals += 1
                        testPoints.append(i)
                if abs(np.dot(testPlane[1],point[3:])/(LA.norm(testPlane[1])*LA.norm(point[3:]))) > max(0,(1-20*threshold)):
                    # if normals are close accept if near to the candidate plane
                    if abs((np.dot(testPlane[1],point[:3])+d)*1.0/LA.norm(testPlane[1],ord = 2)) < threshold:
                        testPoints.append(i)
                        
            #if len(testPoints) < 10:
            #    continue
            if len(testPoints) > len(bestPoints):#plane_group:
                trial += 1
                bestPlane = testPlane
                bestPoints = testPoints
                matched_idx_from_existing = existing_pc_idx

        #
        if trial > 0:
            
            #candidate_new_points = depletable_pc[bestPoints,:3]
            
            #nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(depletable_existing[matched_idx_from_existing,:3])
            #distances, indices = nbrs.kneighbors(candidate_new_points)
            
            best_pc = np.concatenate(
                (
                    depletable_existing[matched_idx_from_existing,:3],
                    depletable_pc[bestPoints,:3]#candidate_new_points[np.where(distances>0.1)[0]]
                ),
                axis = 0
            )
            
            best_pc = np.unique(np.around(best_pc,decimals=3),axis = 0)
            
            if verbose: print(" Added a ",bestPlane,"\n in {:.3f} seconds".format(time.time() - t1))

            d = -bestPlane[0].dot(bestPlane[1])
            
            phi = math.fabs(bestPlane[1][1]* 1./LA.norm(bestPlane[1])) # y/r

            # Add final candidate plane to list of planes
            planes.append([
                [bestPlane ,phi, 0],
                #np.concatenate((np.stack((PX,PY,PZ)).T,depletable_pc[bestPoints][:,3:6]),axis=1),
                np.concatenate((best_pc[:,:3],np.repeat([bestPlane[1]],len(best_pc),axis = 0)),axis=1),
                depletable_existing[sample]],
            )     

            # Remove points of final plane from remaining candidate points
            depletable_pc = np.delete(depletable_pc,bestPoints,0)
            depletable_existing = np.delete(depletable_existing,matched_idx_from_existing,0)

    if verbose:
        print(" ",len(planes)," updated planes.")
        print("Remaining from new",depletable_pc.shape)
        print("Remaining from exisiting",depletable_existing.shape)
        
    try:
        remaining_existing_normals = np.unique(np.round(depletable_existing[:,3:],decimals = 5), axis = 0)
    except:
        remaining_existing_normals = []

    if planes_to_find - len(planes) - len(remaining_existing_normals) > 1:
    
        more_planes = getRansacPlanes(
            depletable_pc,
            planes_to_find = planes_to_find - len(planes) - len(remaining_existing_normals),
            threshold = threshold,
            trials = trials
        )

        if verbose: print("  ",len(more_planes),"additional planes")
            
        planes = planes + more_planes

    if verbose and remaining_existing_normals.any(): print(remaining_existing_normals.shape,"Remaining normals")

    for normal in remaining_existing_normals:

        existing_pc_idx = np.where(np.round(depletable_existing[:,3:], decimals = 5) == normal)[0]
        
        current_plane = [
            depletable_existing[np.random.choice(existing_pc_idx),:3],
            normal
        ]
        phi = math.fabs(current_plane[1][1]* 1./LA.norm(current_plane[1]))
        
        planes.append([
            [current_plane,phi,0],
            depletable_existing[existing_pc_idx],
            [] # empty because already
        ])

    if verbose: print(len(planes),"Final planes")
        
    return planes


def getGeneralizedPointCloud(
    planes,
    triangle_area_threshold = 0.1,
    strict = False,
    verbose = False
):
    
    included_points = 0

    generalized_points = []
    generalized_triangles = []

    for plane_params, points, refpoint in planes:
        bestPlane = plane_params[0]
        phi = plane_params[1]
        density = plane_params[2]
        #points = plane_[1]
        #refpoint = plane_[2]
        
        #triangles = plane_[2]
        triangles = getDelaunayTriangles(
            bestPlane, 
            points,
            triangle_area_threshold = triangle_area_threshold,
            phi = phi,
            strict = strict,
            verbose = verbose
        )
        
        if len(triangles) == 0:
            if verbose: print("No triangles at",bestPlane,len(planes),points.shape)
            continue
            
        PX = points[:,0]
        PY = points[:,1]
        PZ = points[:,2]
            
        if strict:
            # Get area and point desity of the planes
            v1 = points[triangles[:,0],:3] - points[triangles[:,1],:3]
            v2 = points[triangles[:,2],:3] - points[triangles[:,1],:3]
            area = np.abs(LA.norm(np.cross(v1,v2), axis = 1))*0.5
            point_density = len(points)/np.nansum(area)
            
            if verbose: print("Plane area {:.3f}; point density {:.3f}.".format(np.nansum(area),point_density))
            
            if point_density < 0.75*density:
                continue

        included_points += len(points)

        # Populating the 
        if len(generalized_points)== 0:
            generalized_points = np.concatenate((np.stack((PX,PY,PZ)).T,np.tile(bestPlane[1],(len(PX),1))),axis=1)
            generalized_triangles = triangles[:,:3]
        else:
            generalized_triangles = np.append(generalized_triangles,triangles[:,:3]+len(generalized_points),axis=0)
            generalized_points = np.append(generalized_points,
                                           np.concatenate((np.stack((PX,PY,PZ)).T,np.tile(bestPlane[1],(len(PX),1))),axis=1),
                                           axis=0)
            
    return generalized_points, generalized_triangles


def getGeneralizedPointCloudOld(planes,plane_properties,plane_threshold = 0.75):
    included_points = 0

    generalized_points = []
    generalized_triangles = []

    for i, plane_ in enumerate(planes):
        plane = plane_[0]
        points = plane_[1]
        triangles = plane_[2]
        refpoint = plane_[3]

        """
        # Get area and point desity of the planes
            plane_area = getPointCloudArea(np.stack((PX,PY,PZ)).T,p_triangles)
            plane_properties.append([
                plane_area,
                len(bestPoints)/plane_area --> point density
            ])
        """
        # If point density is below 0.75 the min(median, mean) of the planes in the space.
        if plane_properties[i,1]<plane_threshold*min(np.mean(plane_properties[:,1]),np.median(plane_properties[:,1])):
            continue

        PX = points[:,0]
        PY = points[:,1]
        PZ = points[:,2]

        included_points += len(points)

        # Populating the 
        if len(generalized_points)== 0:
            generalized_points = np.concatenate((np.stack((PX,PY,PZ)).T,np.tile(plane[1],(len(PX),1))),axis=1)
            generalized_triangles = triangles[:,:3]
        else:
            generalized_triangles = np.append(generalized_triangles,triangles[:,:3]+len(generalized_points),axis=0)
            generalized_points = np.append(generalized_points,
                                           np.concatenate((np.stack((PX,PY,PZ)).T,np.tile(plane[1],(len(PX),1))),axis=1),
                                           axis=0)
            
    return generalized_points, generalized_triangles

# Getting the "spin image" descriptors for a given point cloud    
def getSpinImageDescriptors(_point_cloud,
                            resolution = 20,
                            normalize = True,
                            down_resolution = 3, #keypoint resolution
                            #localize = False,
                            local_radius = 1.0,
                            cylindrical_quantization = [10,20],
                            verbose = False,
                            old = False,
                            key_cap = 50, 
                            strict_cap = False
                           ):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    other_pc = np.copy(np_pointCloud)    
    
    if old:
        chosen_points = OLDgetQuantizedPointCloudOnly(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("Old:",chosen_points.shape)
    else:
        chosen_points = getQuantizedPointCloudOnly(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("New:",chosen_points.shape)

    chosen_points = np.delete(chosen_points,np.where(LA.norm(chosen_points[:,3:],axis=1)== 0)[0],0)
    
    if strict_cap and len(chosen_points) > key_cap:
        chosen_points = chosen_points[np.random.choice(len(chosen_points),key_cap)]
        if verbose: print("Capped chosen points to", key_cap)
    
    """
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Complete",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
                print("  IN Chosen Points before:",chosen_points.shape)
                #vertical = True

    """
    view_invariant_descriptor_cylinders = np.zeros(
        (np.append(len(chosen_points),
                   np.asarray(cylindrical_quantization)
                   #np.asarray(cylindrical_quantization*0.5*resolution,dtype = np.uint32)
                   #[int(resolution*cylindrical_quantization[0]),int(resolution*cylindrical_quantization[1])]
                  )
        ), dtype=np.float16)
    #    shape is (number of points, number of a bins, number of b bins)
    #print(view_invariant_descriptor_cylinders.shape)

    t0 = time.time()
    for i,c_p in enumerate(chosen_points):

        k_ps = other_pc[:,:3] - c_p[:3] # point (vertex) differences
        k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)==0)[0],0) # removing itself

        #if localize:
        #    k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)>local_radius)[0],0)

        theta = np.arccos(np.clip(np.sum(c_p[3:]*k_ps,axis=1)/(LA.norm(c_p[3:])*LA.norm(k_ps,axis=1)),-1,1))# normal at keypoint

        d_a = cylindrical_quantization[0]*LA.norm(k_ps,axis=1)*np.sin(theta)
        d_b = np.clip(cylindrical_quantization[1]*0.5*LA.norm(k_ps,axis=1)*np.cos(theta)+cylindrical_quantization[1]*0.5,
                      0,cylindrical_quantization[1]-1)
        
        # removing the points that are outside of the spin region
        k_ps = np.delete(k_ps,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)

        k_ps = np.delete(k_ps,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        
        a = np.ceil(np.clip(d_a,0,cylindrical_quantization[0]-1))
        b = np.ceil(np.clip(d_b,0,cylindrical_quantization[1]-1))

        for k,_ in enumerate(k_ps):
            
            if LA.norm(k_ps[k]) == 0:
                continue

            # skip points that are beyond the scope of the pre-defined descriptor
            if d_a[k] >= cylindrical_quantization[0]-1 or d_b[k] >= cylindrical_quantization[1]-1:
                continue
            if d_a[k]>a[k] or d_b[k]>b[k]:
                continue
            diff_a = a[k]-d_a[k]
            diff_b = b[k]-d_b[k]
            
            if diff_a > 1 or diff_b > 1:# or diff_a == 0 or diff_b == 0:
                continue
                
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])] += (diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])] += (1-diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])-1] += (diff_a)*(1-diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])-1] += (1-diff_a)*(1-diff_b)

            
    view_invariant_descriptors = []
    for descriptors in view_invariant_descriptor_cylinders:
        if normalize: # max-normalization
            if np.amax(descriptors) == 0:
                pass
            else:
                descriptors = descriptors/np.amax(descriptors)
        view_invariant_descriptors.append(descriptors.flatten('C'))

    #print("Time to get spin image descriptors",time.time()-t0)
        #print(view_invariant_descriptors.shape)
    """    
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)
        print("  IN Chosen Points after:",chosen_points.shape)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Chosen",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
    """    
    view_invariant_descriptors = np.asarray(view_invariant_descriptors)
    return view_invariant_descriptors, chosen_points, view_invariant_descriptor_cylinders

# Getting the "spin image" descriptors for a given point cloud    
def getSpinImageDescriptorsTest1(_point_cloud,
                            resolution = 20,
                            normalize = True,
                            down_resolution = 3, #keypoint resolution
                            #localize = False,
                            local_radius = 1.0,
                            cylindrical_quantization = [10,20],
                            verbose = False,
                            old = False,
                            key_cap = 50, 
                            strict_cap = False
                           ):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    other_pc = np.copy(np_pointCloud)
    
    t0 = time.time()
    
    if old:
        chosen_points = getQuantizedPointCloudOnly(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("Old:",chosen_points.shape)
    else:
        chosen_points = getQuantizedPointCloudOnlyTEST(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("New:",chosen_points.shape)

    chosen_points = np.delete(chosen_points,np.where(LA.norm(chosen_points[:,3:],axis=1)== 0)[0],0)
    
    t1 = time.time()
    
    if strict_cap and len(chosen_points) > key_cap:
        chosen_points = chosen_points[np.random.choice(len(chosen_points),key_cap)]
        if verbose: print("Capped chosen points to", key_cap)
    
    """
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Complete",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
                print("  IN Chosen Points before:",chosen_points.shape)
                #vertical = True

    """
    
    view_invariant_descriptor_cylinders = np.zeros(
        (np.append(len(chosen_points),
                   np.asarray(cylindrical_quantization)
                   #np.asarray(cylindrical_quantization*0.5*resolution,dtype = np.uint32)
                   #[int(resolution*cylindrical_quantization[0]),int(resolution*cylindrical_quantization[1])]
                  )
        ), dtype=np.float16)
    #    shape is (number of points, number of a bins, number of b bins)
    #print(view_invariant_descriptor_cylinders.shape)
    
    # N  = len(chosen points), M = len(other pc)

    for i,c_p in enumerate(chosen_points):

        k_ps = other_pc[:,:3] - c_p[:3] # point (vertex) differences; M
        k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)==0)[0],0) # removing itself; M-1

        #if localize:
        #    k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)>local_radius)[0],0)

        # len(theta) = M-1
        theta = np.arccos(np.clip(np.sum(c_p[3:]*k_ps,axis=1)/(LA.norm(c_p[3:])*LA.norm(k_ps,axis=1)),-1,1))# normal at keypoint

        
        d_a = cylindrical_quantization[0]*LA.norm(k_ps,axis=1)*np.sin(theta)
        d_b = np.clip(cylindrical_quantization[1]*0.5*LA.norm(k_ps,axis=1)*np.cos(theta)+cylindrical_quantization[1]*0.5,
                      0,cylindrical_quantization[1]-1)
        
        # removing the points that are outside of the spin region
        k_ps = np.delete(k_ps,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)

        k_ps = np.delete(k_ps,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        
        a = np.asarray(np.ceil(np.clip(d_a,0,cylindrical_quantization[0]-1)), dtype = np.uint)
        b = np.asarray(np.ceil(np.clip(d_b,0,cylindrical_quantization[1]-1)), dtype = np.uint)

        """for k,_ in enumerate(k_ps):
            
            if LA.norm(k_ps[k]) == 0:
                continue

            # skip points that are beyond the scope of the pre-defined descriptor
            if d_a[k] >= cylindrical_quantization[0]-1 or d_b[k] >= cylindrical_quantization[1]-1:
                continue
            if d_a[k]>a[k] or d_b[k]>b[k]:
                continue
            diff_a = a[k]-d_a[k]
            diff_b = b[k]-d_b[k]
            
            if diff_a > 1 or diff_b > 1:# or diff_a == 0 or diff_b == 0:
                continue
                
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])] += (diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])] += (1-diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])-1] += (diff_a)*(1-diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])-1] += (1-diff_a)*(1-diff_b)
        """
        # len(diff_*) ≤ M-1; where M = len(other pc)
        diff_a = a-d_a
        diff_b = b-d_b
        
        submap = np.zeros(
            (len(diff_a),
             cylindrical_quantization[0],
             cylindrical_quantization[1]),
            dtype=np.float16
        )
        
        submap[np.arange(len(diff_a)),a,b] += (diff_a)*(diff_b)
        submap[np.arange(len(diff_a)),a-1,b] += (1-diff_a)*(diff_b)
        submap[np.arange(len(diff_a)),a,b-1] += (diff_a)*(1-diff_b)
        submap[np.arange(len(diff_a)),a-1,b-1] += (1-diff_a)*(1-diff_b)
        
        view_invariant_descriptor_cylinders[i] += np.sum(submap, axis = 0)
        
        #view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])] += (diff_a)*(diff_b)
        #view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])] += (1-diff_a)*(diff_b)
        #view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])-1] += (diff_a)*(1-diff_b)
        #view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])-1] += (1-diff_a)*(1-diff_b)        
        
    view_invariant_descriptors = []

    for descriptors in view_invariant_descriptor_cylinders:
        if normalize: # max-normalization
            if np.amax(descriptors) == 0:
                pass
            else:
                descriptors = descriptors/np.amax(descriptors)
        view_invariant_descriptors.append(descriptors.flatten('C'))

    t2 = time.time()
    #print("Time to get spin image descriptors",time.time()-t0)
        #print(view_invariant_descriptors.shape)
    """    
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)
        print("  IN Chosen Points after:",chosen_points.shape)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Chosen",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
    """    
    view_invariant_descriptors = np.asarray(view_invariant_descriptors)
    
    if verbose: 
        print("  desc shape:",view_invariant_descriptor_cylinders.shape, " kp's shape",chosen_points.shape)
        print("  To get keypoints {:.3f} seconds, to get descriptors {:.3f} seconds".format(t1-t0,t2-t1))
    
    return view_invariant_descriptors, chosen_points, view_invariant_descriptor_cylinders, t1-t0, t2- t1


# Getting the "spin image" descriptors for a given point cloud    
def getSpinImageDescriptorsTryTest(_point_cloud,
                            resolution = 20,
                            normalize = True,
                            down_resolution = 3, #keypoint resolution
                            #localize = False,
                            local_radius = 1.0,
                            cylindrical_quantization = [10,20],
                            verbose = False,
                            old = False,
                            key_cap = 50, 
                            strict_cap = False
                           ):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    other_pc = np.copy(np_pointCloud)
    
    t0 = time.time()
    
    if old:
        chosen_points = OLDgetQuantizedPointCloudOnly(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("Old:",chosen_points.shape)
    else:
        chosen_points = getQuantizedPointCloudOnlyTEST(np_pointCloud,down_resolution,verbose=verbose)
        if verbose: print("New:",chosen_points.shape)

    chosen_points = np.delete(chosen_points,np.where(LA.norm(chosen_points[:,3:],axis=1)== 0)[0],0)
    
    t1 = time.time()
    
    if strict_cap and len(chosen_points) > key_cap:
        chosen_points = chosen_points[np.random.choice(len(chosen_points),key_cap)]
        if verbose: print("Capped chosen points to", key_cap)
    
    """
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Complete",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
                print("  IN Chosen Points before:",chosen_points.shape)
                #vertical = True

    """
    
    view_invariant_descriptor_cylinders = np.zeros(
        (np.append(len(chosen_points),
                   np.asarray(cylindrical_quantization)
                   #np.asarray(cylindrical_quantization*0.5*resolution,dtype = np.uint32)
                   #[int(resolution*cylindrical_quantization[0]),int(resolution*cylindrical_quantization[1])]
                  )
        ), dtype=np.float16)
    #    shape is (number of points, number of a bins, number of b bins)
    #print(view_invariant_descriptor_cylinders.shape)
    
    #for i,c_p in enumerate(chosen_points):

    k_ps = other_pc[:,np.newaxis,:3] - chosen_points[np.newaxis,:,:3] # point (vertex) differences
    #k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=-1)==0)[0],0) # removing itself
    #print("Chosen points:",chosen_points.shape, "Other PC: ", other_pc.shape, "KP shape:",k_ps.shape)

    theta = np.arccos(np.clip(np.sum(chosen_points[np.newaxis,:,:3]*k_ps,axis=-1)/(LA.norm(chosen_points[np.newaxis,:,:3])*LA.norm(k_ps,axis=-1)),-1,1))# normal at keypoint
    d_a = cylindrical_quantization[0]*LA.norm(k_ps,axis=-1)*np.sin(theta)
    d_b = np.clip(cylindrical_quantization[1]*0.5*LA.norm(k_ps,axis=-1)*np.cos(theta)+cylindrical_quantization[1]*0.5,
                  0,cylindrical_quantization[1]-1)
    
    #print("Theta:",theta.shape, "D_a", d_a.shape, "D_b", d_b.shape)
    
    i_a = np.ceil(d_a)
    i_b = np.ceil(d_b)

    diff_a = i_a - d_a
    diff_b = i_b - d_b
    
    mult = diff_a*diff_b
    
    #print("KP shape:",k_ps.shape, "new _a", diff_a.shape, "new _b", diff_b.shape, "mult shape", mult.shape)
    
    """
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])] += (diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])] += (1-diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])-1] += (diff_a)*(1-diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])-1] += (1-diff_a)*(1-diff_b)
    """

    for k, k_p in enumerate(k_ps):

        submap = np.zeros((len(chosen_points),
            max(cylindrical_quantization[0]+3,int(max(d_a[k]))),
            max(cylindrical_quantization[1]+3,int(max(d_b[k])))
        ))

        a_i = np.clip(np.searchsorted(np.arange(max(cylindrical_quantization[0]+2,int(max(i_a[k])))),i_a[k],side="right")-1,0,cylindrical_quantization[0]+2)
        b_i = np.clip(np.searchsorted(np.arange(max(cylindrical_quantization[1]+2,int(max(i_b[k])))),i_b[k],side="right")-1,0,cylindrical_quantization[1]+2)
        a_i_s = np.clip(np.searchsorted(np.arange(max(cylindrical_quantization[0]+2,int(max(i_a[k])))),i_a[k]-1,side="right")-1,0,cylindrical_quantization[0]+2)
        b_i_s = np.clip(np.searchsorted(np.arange(max(cylindrical_quantization[1]+2,int(max(i_b[k])))),i_b[k]-1,side="right")-1,0,cylindrical_quantization[1]+2)

        submap[np.arange(len(chosen_points)),a_i,b_i] += diff_a[k]*diff_b[k]    
        submap[np.arange(len(chosen_points)),a_i_s,b_i] += (1-diff_a[k])*diff_b[k]
        submap[np.arange(len(chosen_points)),a_i,b_i_s] += diff_a[k]*(1-diff_b[k])
        submap[np.arange(len(chosen_points)),a_i_s,b_i_s] += (1-diff_a[k])*(1-diff_b[k])

        view_invariant_descriptor_cylinders += submap[:,:cylindrical_quantization[0],:cylindrical_quantization[1]]

    view_invariant_descriptors = []

    for descriptors in view_invariant_descriptor_cylinders:
        if normalize: # max-normalization
            if np.amax(descriptors) == 0:
                pass
            else:
                descriptors = descriptors/np.amax(descriptors)
        view_invariant_descriptors.append(descriptors.flatten('C'))

    t2 = time.time()
    #print("Time to get spin image descriptors",time.time()-t0)
        #print(view_invariant_descriptors.shape)
    """    
    if verbose:
        unique_normals = np.unique(chosen_points[:,3:],axis=0)
        print("  IN Chosen Points after:",chosen_points.shape)

        for normal in unique_normals:

            if np.round(np.abs(np.dot(normal,[0,1,0]))) == 0:
                print("  IN Chosen",object_name,"Vertical",np.abs(np.dot(normal,[0,1,0])),normal,LA.norm(normal,2))
    """    
    view_invariant_descriptors = np.asarray(view_invariant_descriptors)
    
    if verbose: 
        print("\n  desc shape:",view_invariant_descriptor_cylinders.shape, " kp's shape",chosen_points.shape)
        print("  To get keypoints {:.3f} seconds, to get descriptors {:.3f} seconds".format(t1-t0,t2-t1))
    
    return view_invariant_descriptors, chosen_points, view_invariant_descriptor_cylinders, t1-t0, t2- t1



# Getting the "spin image" descriptors for a given point cloud
# Using the z instead of theta.
def getSpinImageDescriptorsZ(_point_cloud,
                            resolution = 20,
                            normalize = True,
                            down_resolution = 3,
                            #localize = False,
                            local_radius = 1.0,
                            cylindrical_quantization = [10,20]
                           ):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    other_pc = np.copy(np_pointCloud)    
    
    #print(np_pointCloud.shape)
    chosen_points = getQuantizedPointCloudOnly(np_pointCloud,down_resolution)
    chosen_points = np.delete(chosen_points,np.where(LA.norm(chosen_points[:,3:6],axis=1)== 0)[0],0)
    #print(chosen_points.shape)

    view_invariant_descriptor_cylinders = np.zeros(
        (np.append(len(chosen_points),
                   np.asarray(cylindrical_quantization)
                   #np.asarray(cylindrical_quantization*0.5*resolution,dtype = np.uint32)
                   #[int(resolution*cylindrical_quantization[0]),int(resolution*cylindrical_quantization[1])]
                  )
        ), dtype=np.float16)
    #    shape is (numer of points, number of a bins, number of b bins)
    #print(view_invariant_descriptor_cylinders.shape)

    t0 = time.time()
    for i,c_p in enumerate(chosen_points):

        k_ps = other_pc[:,:3] - c_p[:3] # point (vertex) differences
        k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)==0)[0],0)

        #if localize:
        #    k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)>local_radius)[0],0)

        theta = np.arccos(np.clip(np.sum(c_p[3:]*k_ps,axis=1)/(LA.norm(c_p[3:])*LA.norm(k_ps,axis=1)),-1,1))

        d_a = cylindrical_quantization[0]*LA.norm(k_ps,axis=1)*np.sin(theta)
        d_b = cylindrical_quantization[1]*LA.norm(k_ps,axis=1)*np.cos(theta)
#        d_b = np.clip(cylindrical_quantization[1]*0.5*LA.norm(k_ps,axis=1)*np.cos(theta)+cylindrical_quantization[1]*0.5,
#                       0,cylindrical_quantization[1]-1)
        
        k_ps = np.delete(k_ps,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_a>=cylindrical_quantization[0]-1)[0],0)

        k_ps = np.delete(k_ps,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_a = np.delete(d_a,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        d_b = np.delete(d_b,np.where(d_b>=cylindrical_quantization[1]-1)[0],0)
        
        a = np.ceil(np.clip(d_a,0,cylindrical_quantization[0]-1))
        b = np.ceil(np.clip(d_b,0,cylindrical_quantization[1]-1))

        for k,_ in enumerate(k_ps):
            
            if LA.norm(k_ps[k]) == 0:
                continue

            # skip points that are beyond the scope of the pre-defined descriptor
            if d_a[k] >= cylindrical_quantization[0]-1 or d_b[k] >= cylindrical_quantization[1]-1:
                continue
            if d_a[k]>a[k] or d_b[k]>b[k]:
                continue
            diff_a = a[k]-d_a[k]
            diff_b = b[k]-d_b[k]
            
            if diff_a > 1 or diff_b > 1:# or diff_a == 0 or diff_b == 0:
                continue
                
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])] += (diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])] += (1-diff_a)*(diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k]),int(b[k])-1] += (diff_a)*(1-diff_b)
            view_invariant_descriptor_cylinders[i,int(a[k])-1,int(b[k])-1] += (1-diff_a)*(1-diff_b)

            
    view_invariant_descriptors = []
    for descriptors in view_invariant_descriptor_cylinders:
        if normalize: # max-normalization
            if np.amax(descriptors) == 0:
                pass
            else:
                descriptors = descriptors/np.amax(descriptors)
        view_invariant_descriptors.append(descriptors.flatten('C'))

    #print("Time to get spin image descriptors",time.time()-t0)
        #print(view_invariant_descriptors.shape)
        
    view_invariant_descriptors = np.asarray(view_invariant_descriptors)
    return view_invariant_descriptors, chosen_points, view_invariant_descriptor_cylinders

# Getting the descriptors for a given point cloud    
def getSelfSimilarityDescriptors(_point_cloud,
                                 scale = 20,
                                 normalize = True,
                                 localize = False,
                                 local_radius = 1.0,
                                 minima = False,
                                 curve_radius = 0,
                                 spherical_quantization = [6,8,6]
                                ):
    
    np_pointCloud = np.asarray(np.copy(_point_cloud))
    
    try:
        # Get the local self-similarity for all points based on the neighbor size, i.e. scale/2
        #local sim has: 'x','y','z','nx','ny','nz','cx','cy','cz','c','s'        
        if np.max(curve_radius,0) == 0:
            local_sim, k_neighbors = getLocalSelfSimilarity(np_pointCloud,scale=scale)
        else:
            local_sim, k_neighbors = getDynamicLocalSelfSimilarity(np_pointCloud,curve_radius,20)
    except Exception as ex:
        print("getSelfSimilarityDescriptors: Error in getting selfsimilarity descriptors:",ex)
        pass

    try:
        # Comparing the local curv values and getting the indices of those
        # which are the maximum within its locality, i.e. neighbor size
        local_curv_maxima_check = local_sim[:,-2][:,None] > local_sim[:,-2][k_neighbors]
        chosen_indices = np.asarray(np.where(np.all(local_curv_maxima_check,axis=1)==True)[0],dtype=np.uint16)
        
        if minima:
            local_curv_minima_check = local_sim[:,-2][:,None] < local_sim[:,-2][k_neighbors]
            chosen_indices = np.asarray(np.where(np.all(local_curv_minima_check,axis=1)==True)[0],dtype=np.uint16)
        
        #'i','x','y','z','nx','ny','nz','cx','cy','cz','c','s'
        # Adding the index (to the complete pointcloud)
        chosen_keypoints = np.concatenate((chosen_indices[:,None],local_sim[chosen_indices,:]),axis=1)
        # Removing keypoints with 0 magnitude normals.
        chosen_keypoints = np.delete(chosen_keypoints,np.where(LA.norm(chosen_keypoints[:,4:7],axis=1)== 0)[0],0)
        #print("Chosen keypoints size",chosen_keypoints.shape)
        
        view_invariant_descriptor_sphere = np.zeros((np.append(np.append(len(chosen_keypoints),spherical_quantization),2)))
                                            #np.zeros((len(chosen_keypoints),6,8,6,2)) # i, r, p, t, (sim, count)
        view_invariant_descriptors = []        
        #print(view_invariant_descriptors.shape)
        
        for i,c_kp in enumerate(chosen_keypoints):
            
            other_pc = np.delete(local_sim,int(c_kp[0]),0)
            #print(local_sim.shape,other_pc.shape)
            k_ps = other_pc[:,:3] - c_kp[1:4] # keypoint differences
            
            if localize:
                k_ps = np.delete(k_ps,np.where(LA.norm(k_ps,axis=1)>local_radius)[0],0)
            
            #print(i,"k_ps",k_ps.shape)
            
            d_z = np.sum(c_kp[4:7]*k_ps,axis=1)[:,None]*c_kp[4:7]# normal at keypoint
            d_x = np.sum(c_kp[7:10]*k_ps,axis=1)[:,None]*c_kp[7:10] # principal curvature at keypoint
            d_y = np.sum(np.cross(c_kp[4:7],c_kp[7:10])*k_ps,axis=1)[:,None]*np.cross(c_kp[4:7],c_kp[7:10]) # cross of normal and curvature
            #d_xyz = np.concatenate((d_x,d_y,d_z),axis=1)
            #print("D_XYZ",d_xyz.shape)
            
            d_r = np.asarray(np.clip(np.floor((spherical_quantization[0]-1)*LA.norm(k_ps,axis = 1)/local_radius),0,(spherical_quantization[0]-1)))
            d_p = np.floor((spherical_quantization[1]-1)*np.arccos(np.clip(LA.norm(d_x,axis=1)/LA.norm(d_x + d_y,axis=1),-1,1))/(2*math.pi))
            d_t = np.floor((spherical_quantization[2]-1)*np.arccos(np.clip(LA.norm(d_z,axis=1)/LA.norm(d_x + d_z + d_y,axis=1),-1,1))/math.pi)
            #print(d_r.shape,d_p.shape,d_t.shape)
            
            for k,_ in enumerate(k_ps):
                if LA.norm(k_ps[k]) == 0:
                    continue
                try:
                    i_r = int(d_r[k])
                    i_p = int(d_p[k])
                    i_t = int(d_t[k])
                    view_invariant_descriptor_sphere[i,i_r,i_p,i_t,0] = view_invariant_descriptor_sphere[i,i_r,i_p,i_t,0]*view_invariant_descriptor_sphere[i,i_r,i_p,i_t,1]
                    view_invariant_descriptor_sphere[i,i_r,i_p,i_t,1] += 1
                    view_invariant_descriptor_sphere[i,i_r,i_p,i_t,0] += other_pc[k,-1]
                    view_invariant_descriptor_sphere[i,i_r,i_p,i_t,0] = view_invariant_descriptor_sphere[i,i_r,i_p,i_t,0]/view_invariant_descriptor_sphere[i,i_r,i_p,i_t,1]
                except Exception as e1:
                    print("Chosen keypoints",chosen_keypoints.shape)
                    print(c_kp[0],"current reference keypoint",c_kp[1:4],
                          "\nfrom local_sim",local_sim[int(c_kp[0]),:4])
                    print("current other keypoint",k_ps[k],other_pc[k,:3])
                    print(object_name,i,k,d_p[k],d_t[k],"Error:",e1)
                    #print(i_r,i_p,i_t)
                    print("x,y,z:",other_pc[k,1:4],LA.norm(other_pc[k,1:4]))
                    print("nx,ny,nz:",other_pc[k,4:7],LA.norm(other_pc[k,4:7]))
                    print("cx,cy,cz:",other_pc[k,7:10],LA.norm(other_pc[k,7:10]))
                    print("d_z:",d_z[k],LA.norm(d_z[k]))
                    print("d_x:",d_x[k],LA.norm(d_x[k]))
                    print("d_z(dot)d_x:",np.degrees(np.arccos(np.sum(d_z[k]*d_x[k])/LA.norm(d_z[k]*d_x[k]))))
                    print("d_y:",d_y[k],LA.norm(d_y[k]))
                    print("Len k_ps",LA.norm(k_ps[k]))
                    print("Len d_(x,y,z)",LA.norm(d_x[k] + d_z[k] + d_y[k]))
                    print("|d_x|/|d_xy|",LA.norm(d_x[k])/LA.norm(d_x[k] + d_y[k]),
                          "->",np.arccos(LA.norm(d_x[k])/LA.norm(d_x[k] + d_y[k])))
                    print("|d_z|/|d_xyz|",LA.norm(d_z[k])/LA.norm(d_x[k] + d_z[k]+ d_y[k]),
                          "->",np.arccos(LA.norm(d_z[k])/LA.norm(d_x[k] + d_z[k]+ d_y[k])))
                    #print(d_x)
                    #print(d_y)
                    print(d_p,d_t)
                    break
            
        for descriptor in view_invariant_descriptor_sphere:
            #print("before",descriptor.shape)
            #descriptor = descriptor[:,:,:,0].flatten('C')
            #print("after",descriptor.shape)
            view_invariant_descriptors.append(descriptor[:,:,:,0].flatten('C'))
        
        view_invariant_descriptors = np.asarray(view_invariant_descriptors)    
#        print(chosen_keypoints.shape,view_invariant_descriptors.shape)
#        ss_descriptors.append([object_name,
#                               chosen_keypoints,
#                               view_invariant_descriptors])
        
    except Exception as ex:
        print("getSelfSimilarityDescriptors: Object with shape {}, Error message: {}".format(np_pointCloud.shape,ex))
        pass
    
    if normalize:
        #print(view_invariant_descriptors.shape)
        view_invariant_descriptors = view_invariant_descriptors/np.amax(view_invariant_descriptors,1)[:,np.newaxis]
        #print(view_invariant_descriptors.shape)
    return view_invariant_descriptors, chosen_keypoints, local_sim, k_neighbors


def getLocalSelfSimilarity(_point_cloud, scale = 20, thresh_max=20):

    ##1 normal_projection is the projection of the normal vector of the nearby points
    #   on to the normal of the reference (i.e. n_pointCloud[:,3:]);
    #   shape is (length of point clouds, neighbors, 3)

    neighbor_size = max(int(scale/2),1)
    
    #print("getSelfSimilarityDescriptors of point-cloud with size {} and {} neighbors".format(_point_cloud.shape,neighbor_size))

    if neighbor_size >= len(_point_cloud):
        print('getLocalSelfSimilarity: Computed neighbor size {} is greater than point cloud size ({})'.format(neighbor_size,len(_point_cloud)))
        return
    
    try:
        k_neighbors, _ = getEuclideanNearestNeighbours(_point_cloud, neighbor_size)
    except:
        return

    nearby_points = _point_cloud[k_neighbors,:3]
    diff_nrby_pts = _point_cloud[:,:3][:,None] - nearby_points

    try:
        curvatures = getPrincipalCurvaturesHY(_point_cloud,neighbor_size,['max','max'], k_neighbors)[:,-4:]
    except:
        return
    
    try:
        nearby_normals = _point_cloud[k_neighbors,3:6]
        refrnc_normals = np.reshape(np.repeat(_point_cloud[:,3:6],neighbor_size,axis=0),nearby_normals.shape)

        normal_product = np.sum(np.multiply(refrnc_normals,nearby_normals),axis=2)
    except Exception as e1:
        print("getLocalSelfSimilarity: Error while getting normals,", e1)
        return

    #normal_projection = np.multiply(refrnc_normals,np.reshape(np.repeat(normal_product,3,axis=1),nearby_normals.shape))
    normal_similarity = (math.pi - np.arccos(np.clip(normal_product,-1,1)))/math.pi
    #print("Normal Similarity",normal_similarity.shape)

    curvature_similarity = 1 - np.abs(curvatures[k_neighbors,-1] - curvatures[:,-1][:,None])
    #print("Curvature Similarity",curvature_similarity.shape)

    local_similarity = np.mean(0.5*normal_similarity + 0.5*curvature_similarity,axis=1)
    #print("Local Similarity",local_similarity.shape)
    
    keypoints_with_local_similarity = np.concatenate((_point_cloud,curvatures,local_similarity[:,None]),axis=1)
    # 'x','y','z','nx','ny','nz','cx','cy','cz','c','similarity'
    
    return keypoints_with_local_similarity, k_neighbors

def getDynamicLocalSelfSimilarity(_point_cloud, local_radius = 0.1, thresh_max=20):

    ##1 normal_projection is the projection of the normal vector of the nearby points
    #   on to the normal of the reference (i.e. n_pointCloud[:,3:]);
    #   shape is (length of point clouds, neighbors, 3)
    
    #print("getSelfSimilarityDescriptors of point-cloud with size {} and {} neighbors".format(_point_cloud.shape,neighbor_size))
    
    try:
        k_neighbors, distances = getEuclideanNearestNeighbours(_point_cloud[:,:3], len(_point_cloud)-1)
        neighbor_size = int(np.median(np.argmax(distances > local_radius,1)))

        k_neighbors = k_neighbors[:,:neighbor_size]
        distances = distances[:,:neighbor_size]
        #print("DynamicLSS:",k_neighbors.shape,distances.shape)
    except:
        print("Error in nearest neighbors")
        return

    nearby_points = _point_cloud[k_neighbors,:3]
    diff_nrby_pts = _point_cloud[:,:3][:,None] - nearby_points

    try:
        curvatures = getPrincipalCurvaturesHY(_point_cloud,neighbor_size,['max','max'], k_neighbors)[:,-4:]
    except:
        return
    
    try:
        nearby_normals = _point_cloud[k_neighbors,3:6]
        refrnc_normals = np.reshape(np.repeat(_point_cloud[:,3:6],neighbor_size,axis=0),nearby_normals.shape)

        normal_product = np.sum(np.multiply(refrnc_normals,nearby_normals),axis=2)
    except Exception as e1:
        print("getLocalSelfSimilarity:Error while getting normals:", e1)
        return

    #normal_projection = np.multiply(refrnc_normals,np.reshape(np.repeat(normal_product,3,axis=1),nearby_normals.shape))
    normal_similarity = (math.pi - np.arccos(np.clip(normal_product,-1,1)))/math.pi
    #print("Normal Similarity",normal_similarity.shape)

    curvature_similarity = 1 - np.abs(curvatures[k_neighbors,-1] - curvatures[:,-1][:,None])
    #print("Curvature Similarity",curvature_similarity.shape)

    local_similarity = np.mean(0.5*normal_similarity + 0.5*curvature_similarity,axis=1)
    #print("Local Similarity",local_similarity.shape)
    
    keypoints_with_local_similarity = np.concatenate((_point_cloud,curvatures,local_similarity[:,None]),axis=1)
    # 'x','y','z','nx','ny','nz','cx','cy','cz','c','similarity'
    
    return keypoints_with_local_similarity, k_neighbors


def getMatches(arr1, arr2, n=2, thresh_max=20, range_to_match=np.arange(3)):
    
    arr1 = arr1[:,range_to_match] #np.asarray(ss_descriptors[4][1])[:,:3]  #
    arr2 = arr2[:,range_to_match] #np.asarray(ss_descriptors[8][1])[:,:3]  #

    # Transform it to get a new matrix that repeats the points to prepare for nearest neighbour matching.
    #desired_shape = (point_cloud_coord.shape[0],point_cloud_coord.shape[0],point_cloud_coord.shape[1])
    arr1_r = np.reshape(np.repeat(arr1,len(arr2),0),(len(arr1),len(arr2),len(range_to_match)))
    arr2_r = np.reshape(np.repeat(arr2,len(arr1),0),(len(arr2),len(arr1),len(range_to_match)))
    
    #print("Arr1",arr1.shape,"->",arr1_r.shape)
    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    arr2_r = arr2_r.transpose(1,0,2)
    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    difference = LA.norm(arr1_r - arr2_r,axis=2)
    #print(difference.shape)
    matches = np.argsort(np.clip(difference,0,thresh_max))
    
    difference.sort(axis=1)
    bestmatches = matches[:,:n]

    return bestmatches, difference[:,:n]

def getQueryBiasedMatches(arr1, arr2, n=2, thresh_max=20, range_to_match=np.arange(3)):
    
    arr1 = arr1[:,range_to_match] #np.asarray(ss_descriptors[4][1])[:,:3]  #
    arr2 = arr2[:,range_to_match] #np.asarray(ss_descriptors[8][1])[:,:3]  #

    # Transform it to get a new matrix that repeats the points to prepare for nearest neighbour matching.
    #desired_shape = (point_cloud_coord.shape[0],point_cloud_coord.shape[0],point_cloud_coord.shape[1])
    arr1_r = np.reshape(np.repeat(arr1,len(arr2),0),(len(arr1),len(arr2),len(range_to_match)))
    arr2_r = np.reshape(np.repeat(arr2,len(arr1),0),(len(arr2),len(arr1),len(range_to_match)))
    
    #print("Arr1",arr1.shape,"->",arr1_r.shape)
    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    arr2_r = arr2_r.transpose(1,0,2)
    arr2_r = np.multiply(arr2_r,arr1_r>0)

    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    difference = LA.norm(arr1_r - arr2_r,axis=2)
    #print(difference.shape)
    matches = np.argsort(np.clip(difference,0,thresh_max))
    
    difference.sort(axis=1)
    bestmatches = matches[:,:n]

    return bestmatches, difference[:,:n]

def getLocalBiasedMatches(arr1, arr2, n=2, thresh_max=20, range_to_match=np.arange(3), bias = np.arange(1,0,-0.18)):
    
    arr1 = arr1[:,range_to_match] #np.asarray(ss_descriptors[4][1])[:,:3]  #
    arr2 = arr2[:,range_to_match] #np.asarray(ss_descriptors[8][1])[:,:3]  #

    # Transform it to get a new matrix that repeats the points to prepare for nearest neighbour matching.
    #desired_shape = (point_cloud_coord.shape[0],point_cloud_coord.shape[0],point_cloud_coord.shape[1])
    arr1_r = np.reshape(np.repeat(arr1,len(arr2),0),(len(arr1),len(arr2),len(range_to_match)))
    arr2_r = np.reshape(np.repeat(arr2,len(arr1),0),(len(arr2),len(arr1),len(range_to_match)))
    
    #print("Arr1",arr1.shape,"->",arr1_r.shape)
    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    arr2_r = arr2_r.transpose(1,0,2)
    #print("Arr2",arr2.shape,"->",arr2_r.shape)

    difference = arr1_r - arr2_r
    #print("Difference shape",difference.shape)
    local_biased_difference = difference*np.repeat(bias[:6]/np.amax(bias[:6]),48,axis=0)
    #print("Biased Difference shape",local_biased_difference.shape)
    l2_difference = LA.norm(local_biased_difference,axis=2)
    #print(difference.shape)
    matches = np.argsort(np.clip(l2_difference,0,thresh_max))
    
    l2_difference.sort(axis=1)
    bestmatches = matches[:,:n]

    return bestmatches, l2_difference[:,:n]

# Using the algorithm from Matlab

def getCurvatures(_point_cloud,
                  neighbor_size,
                  mode = ['min','min'], # 0: for the principal curvature vector, 1: for the curvature
                  k_neighbors = [],
                  thresh_max=20):
    
    #print("getCurvatures Mode: vector =",mode[0],", curv =",mode[1])

    ##1 normal_projection is the projection of the normal vector of the nearby points
    #   on to the normal of the reference (i.e. n_pointCloud[:,3:]);
    #   shape is (length of point clouds, neighbors, 3)

    #neighbor_size = max(int(scale/2),1)
    #print("Neighbor size",neighbor_size)
    if neighbor_size >= len(_point_cloud):
        print('Computed neighbor size {} is greater than point cloud size ({})'.format(neighbor_size,len(_point_cloud)))
        return
    
    if len(k_neighbors) == 0:
        k_neighbors, _ = getEuclideanNearestNeighbours(_point_cloud, neighbor_size)

    #nearby_normals = _point_cloud[k_neighbors,3:]
    #refrnc_normals = np.reshape(np.repeat(_point_cloud[:,3:],neighbor_size,axis=0),nearby_normals.shape)

    #normal_product = np.sum(np.multiply(refrnc_normals,nearby_normals),axis=2)
    #normal_projection = np.multiply(refrnc_normals,np.reshape(np.repeat(normal_product,3,axis=1),nearby_normals.shape))

    nearby_points = _point_cloud[k_neighbors,:3]
    diff_nrby_pts = _point_cloud[:,:3][:,None] - nearby_points
    
    #print("Difference size",diff_nrby_pts.shape)
    
    covariance_matrix = np.zeros((_point_cloud.shape[0],6))
    #print("Cov_Mat size",covariance_matrix.shape)
    
    covariance_matrix[:,0] = np.sum(diff_nrby_pts[:,:,0]*diff_nrby_pts[:,:,0],axis=1)
    covariance_matrix[:,1] = np.sum(diff_nrby_pts[:,:,0]*diff_nrby_pts[:,:,1],axis=1)
    covariance_matrix[:,2] = np.sum(diff_nrby_pts[:,:,0]*diff_nrby_pts[:,:,2],axis=1)
    covariance_matrix[:,3] = np.sum(diff_nrby_pts[:,:,1]*diff_nrby_pts[:,:,1],axis=1)
    covariance_matrix[:,4] = np.sum(diff_nrby_pts[:,:,1]*diff_nrby_pts[:,:,2],axis=1)
    covariance_matrix[:,5] = np.sum(diff_nrby_pts[:,:,2]*diff_nrby_pts[:,:,2],axis=1)
    
    covariance_matrix = covariance_matrix/neighbor_size
    #print("Cov_Mat size",covariance_matrix.shape)

    curvature = []
    curvature_vector = []

    ##3 We solve for the covariance matrix element by element and get the curvature vector

    for cov in covariance_matrix:
        local_cov_mat = [[cov[0],cov [1],cov[2]],
                             [cov[1],cov [3],cov[4]],
                             [cov[2],cov [4],cov[5]]]
        local_cov_mat = np.asarray(local_cov_mat)
        w,v = LA.eig(local_cov_mat)
        if mode[0] == 'max':
            curvature_vector.append(v[:,np.argmax(w)])
        else: #default to min
            curvature_vector.append(v[:,np.argmin(w)])
        if mode[1] == 'max':
            curvature.append(w[np.argmax(w)]/np.sum(w))
        else:
            curvature.append(w[np.argmin(w)]/np.sum(w))
        
    curvature_vector = np.asarray(curvature_vector)
    curvature = np.asarray(curvature)

    # 'x','y','z','nx','ny','nz','cx','cy','cz','c'
    keypoints_with_curvature = np.concatenate((_point_cloud,curvature_vector,curvature[:,None]),1)#[np.where(np.multiply(local_maxima_check[:,0]==True,local_maxima_check[:,1]==True))]
    #print("Resulting size of curvatures",keypoints_with_curvature.shape)
    return keypoints_with_curvature

# Using the algorithm from Matlab

def getPrincipalCurvatures(_point_cloud,
                           neighbor_size,
                           mode = ['min','min'], # 0: for the principal curvature vector, 1: for the curvature
                           k_neighbors = [],
                           thresh_max=20):
        ##1 normal_projection is the projection of the normal vector of the nearby points
        #   on to the normal of the reference (i.e. n_pointCloud[:,3:]);
        #   shape is (length of point clouds, neighbors, 3)

    #print("getPrincipalCurvatures Mode: vector =",mode[0],"curv =",mode[1])
    #neighbor_size = max(int(scale/2),1)
    #print("Neighbor size",neighbor_size)
    if neighbor_size >= len(_point_cloud):
        print('Computed neighbor size {} is greater than point cloud size ({})'.format(neighbor_size,len(_point_cloud)))
        return
    
    if len(k_neighbors) == 0:
        k_neighbors, _ = getEuclideanNearestNeighbours(_point_cloud, neighbor_size)

    nearby_normals = _point_cloud[k_neighbors,3:]
    #refrnc_normals = np.reshape(np.repeat(_point_cloud[:,3:],neighbor_size,axis=0),nearby_normals.shape)

    #normal_product = np.sum(np.multiply(refrnc_normals,nearby_normals),axis=2)
    #normal_projection = np.multiply(refrnc_normals,np.reshape(np.repeat(normal_product,3,axis=1),nearby_normals.shape))

    #nearby_points = _point_cloud[k_neighbors,:3]
    #diff_nrby_pts = _point_cloud[:,:3][:,None] - nearby_points
    
    #print("Difference size",diff_nrby_pts.shape)
    
    covariance_matrix = np.zeros((_point_cloud.shape[0],6))
    #print("Cov_Mat size",covariance_matrix.shape)
    
    covariance_matrix[:,0] = np.sum(nearby_normals[:,:,0]*nearby_normals[:,:,0],axis=1)
    covariance_matrix[:,1] = np.sum(nearby_normals[:,:,0]*nearby_normals[:,:,1],axis=1)
    covariance_matrix[:,2] = np.sum(nearby_normals[:,:,0]*nearby_normals[:,:,2],axis=1)
    covariance_matrix[:,3] = np.sum(nearby_normals[:,:,1]*nearby_normals[:,:,1],axis=1)
    covariance_matrix[:,4] = np.sum(nearby_normals[:,:,1]*nearby_normals[:,:,2],axis=1)
    covariance_matrix[:,5] = np.sum(nearby_normals[:,:,2]*nearby_normals[:,:,2],axis=1)
    
    covariance_matrix = covariance_matrix/neighbor_size
    #print("Cov_Mat size",covariance_matrix.shape)

    curvature = []
    curvature_vector = []

    ##3 We solve for the covariance matrix element by element and get the curvature vector

    for cov in covariance_matrix:
        local_cov_mat = [[cov[0],cov [1],cov[2]],
                             [cov[1],cov [3],cov[4]],
                             [cov[2],cov [4],cov[5]]]
        local_cov_mat = np.asarray(local_cov_mat)
        w,v = LA.eig(local_cov_mat)
        if mode[0] == 'max':
            curvature_vector.append(v[:,np.argmax(w)])
        else: #default to min
            curvature_vector.append(v[:,np.argmin(w)])
        if mode[1] == 'max':
            curvature.append(w[np.argmax(w)]/np.sum(w))
        else:
            curvature.append(w[np.argmin(w)]/np.sum(w))

    curvature_vector = np.asarray(curvature_vector)
    curvature = np.asarray(curvature)

    # 'x','y','z','nx','ny','nz','cx','cy','cz','c'
    keypoints_with_Pcurvature = np.concatenate((_point_cloud,curvature_vector,curvature[:,None]),1)#[np.where(np.multiply(local_maxima_check[:,0]==True,local_maxima_check[:,1]==True))]
    #print("Resulting size of curvatures",keypoints_with_Pcurvature.shape)
    return keypoints_with_Pcurvature

# Using the algorithm from Huang and You

def getPrincipalCurvaturesHY(_point_cloud,
                             neighbor_size,
                             mode = ['max','max'], # 0: for the principal curvature vector, 1: for the curvature
                             k_neighbors = [],
                             thresh_max=20):
        ##1 normal_projection is the projection of the normal vector of the nearby points
        #   on to the normal of the reference (i.e. n_pointCloud[:,3:]);
        #   shape is (length of point clouds, neighbors, 3)

    #print("getPrincipalCurvaturesHY Mode: vector =",mode[0],"curv =",mode[1])
    
    #neighbor_size = max(int(scale/2),1)
    #print("Neighbor size",neighbor_size)
    
    if neighbor_size >= len(_point_cloud):
        print('Computed neighbor size {} is greater than point cloud size ({})'.format(neighbor_size,len(_point_cloud)))
        return
    
    if len(k_neighbors) == 0:
        k_neighbors, _ = getEuclideanNearestNeighbours(_point_cloud, neighbor_size)

    try:
        nearby_normals = _point_cloud[k_neighbors,3:]
        refrnc_normals = np.reshape(np.repeat(_point_cloud[:,3:],neighbor_size,axis=0),nearby_normals.shape)
    except Exception as e1:
        print(_point_cloud.shape,k_neighbors.shape)
        print("getPrincipalCurvaturesHY: Error during reshape",e1)
        return
    
    try:
        normal_product = np.sum(np.multiply(refrnc_normals,nearby_normals),axis=2)
        normal_projection = np.multiply(refrnc_normals,np.reshape(np.repeat(normal_product,3,axis=1),nearby_normals.shape))
        tangnt_projection = nearby_normals - normal_projection
    except Exception as e2:
        print("getPrincipalCurvaturesHY: Error during projection",e2)
        return

    #nearby_points = _point_cloud[k_neighbors,:3]
    #diff_nrby_pts = _point_cloud[:,:3][:,None] - nearby_points
    
    #print("Projection size",tangnt_projection.shape)
    
    covariance_matrix = np.zeros((_point_cloud.shape[0],6))
    #print("Cov_Mat size",covariance_matrix.shape)
    
    covariance_matrix[:,0] = np.sum(tangnt_projection[:,:,0]*tangnt_projection[:,:,0],axis=1)
    covariance_matrix[:,1] = np.sum(tangnt_projection[:,:,0]*tangnt_projection[:,:,1],axis=1)
    covariance_matrix[:,2] = np.sum(tangnt_projection[:,:,0]*tangnt_projection[:,:,2],axis=1)
    covariance_matrix[:,3] = np.sum(tangnt_projection[:,:,1]*tangnt_projection[:,:,1],axis=1)
    covariance_matrix[:,4] = np.sum(tangnt_projection[:,:,1]*tangnt_projection[:,:,2],axis=1)
    covariance_matrix[:,5] = np.sum(tangnt_projection[:,:,2]*tangnt_projection[:,:,2],axis=1)
    
    covariance_matrix = covariance_matrix/neighbor_size
    #print("Cov_Mat size",covariance_matrix.shape)

    curvature = []
    curvature_vector = []

    ##3 We solve for the covariance matrix element by element and get the curvature vector

    for cov in covariance_matrix:
        local_cov_mat = [[cov[0],cov [1],cov[2]],
                             [cov[1],cov [3],cov[4]],
                             [cov[2],cov [4],cov[5]]]
        local_cov_mat = np.asarray(local_cov_mat)
        w,v = LA.eig(local_cov_mat)
        if mode[0] == 'max':
            curvature_vector.append(v[:,np.argmax(w)])
        else: #default to min
            curvature_vector.append(v[:,np.argmin(w)])
        if mode[1] == 'max':
            curvature.append(w[np.argmax(w)]/np.sum(w))
        else:
            curvature.append(w[np.argmin(w)]/np.sum(w))

    curvature_vector = np.asarray(curvature_vector)
    curvature = np.asarray(curvature)

    # 'x','y','z','nx','ny','nz','cx','cy','cz','c'
    keypoints_with_Pcurvature = np.concatenate((_point_cloud,curvature_vector,curvature[:,None]),1)#[np.where(np.multiply(local_maxima_check[:,0]==True,local_maxima_check[:,1]==True))]
    #print("Resulting size of curvatures",keypoints_with_Pcurvature.shape)
    return np.real(keypoints_with_Pcurvature)

# scores = [ radius, 
#           [ res (or iteration),
#            [ obj, [miss-match score]]
#           ] 
#          ]
def getErrorRate(qpr_scores, rank = 1):
    qpr_error_rate = []
    qpr_errors = []
    for radius, _scores in qpr_scores:
        _errors, _errors_map = getRankedErrorMaps(_scores, rank = rank)
        size = _errors_map.shape[-1]
        mean, interval = mean_confidence_interval(_errors[:,2]/size,confidence = 0.95)
        qpr_error_rate.append([
            radius,
            np.mean(_errors[:,1]/size), # mean unweighted scores,
            np.mean(_errors[:,2]/size), # mean weighted scores,
            np.std(_errors[:,2]/size), # std
            mean,
            interval
        ])
        qpr_errors.append([
            radius,
            _errors,
            _errors_map
        ])
        
    return qpr_error_rate, qpr_errors

def getRankedErrorMaps(_scores, rank = 1):
    
    _errors = []
    _errors_map = []

    for i_scores, o_scores in enumerate(_scores):
        
        res = o_scores[0]
        scores = o_scores[1]

        _error_map = np.zeros(scores.shape[0])

        uw_errors = 0

        for i,h in enumerate(scores[:,:,0]):
                if i == np.argmax(h):
                    uw_errors += 0
                else:
                    #print(i,np.argsort(h)[-5:])
                    uw_errors += 1


        adjusted_scores = scores[:,:,0]*scores[:,:,-1]/scores[:,:,1]
        bayesian_adjusted_scores = adjusted_scores/np.sum(adjusted_scores,axis=1)
        
        w_errors = 0

        for i,h in enumerate(adjusted_scores):
                if i in np.argsort(h)[-rank:]:
                    w_errors += 0
                else:         
                    _error_map[i] = 1
                    #print(i,np.argsort(h)[-5:])
                    w_errors += 1

        b_errors = 0

        for i,h in enumerate(bayesian_adjusted_scores):
                if i == np.argmax(h):
                    b_errors += 0
                else:            
                    #print(i,np.argsort(h)[-5:])
                    b_errors += 1

        _errors.append([
            res,
            uw_errors,
            w_errors,
            b_errors
        ])

        _errors_map.append(_error_map)

    _errors = np.asarray(_errors)
    _errors_map = np.asarray(_errors_map)
    
    return _errors, _errors_map

def downSampleDescriptors(descriptors,factor=2,cylindrical_shape = np.asarray([10,20])):
    
    # reshaping the 2-D array to a 3-D array, just like the original cylindrical SI descriptor
    cylindrical_d = np.reshape(descriptors,(np.concatenate(([len(descriptors)],cylindrical_shape),axis=0)))
    
    # downsampling the 3-D cylindrical descriptor by the given factor, default is 2
    downsampled_d = cylindrical_d[:,::factor,::factor] + cylindrical_d[:,1::factor,::factor] + cylindrical_d[:,::factor,1::factor] + cylindrical_d[:,1::factor,1::factor]
    
    # normalizing the 3-D descriptors so that max descriptor value is 1.
    downsampled_d = downsampled_d/np.amax(downsampled_d,axis=(1,2))[:,np.newaxis,np.newaxis]
    
    # reshaping it back to a 2-D descriptor
    downsampled_d = np.reshape(downsampled_d,(downsampled_d.shape[0],np.prod(downsampled_d.shape[1:])))
    
    return downsampled_d

def rotatePointCloud(pointCloud, theta = (np.pi)/3, axis = 1, verbose = False):
    
    rotation_matrix = np.asarray([
        [
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)],
        ],
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ],
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
    ])
    
    n_pointCloud = np.copy(pointCloud) # x, y, z
    
    n_pointCloud[:,:3] = np.matmul(pointCloud[:,:3],rotation_matrix[axis])
    n_pointCloud[:,3:] = np.matmul(pointCloud[:,3:],rotation_matrix[axis])

    return n_pointCloud