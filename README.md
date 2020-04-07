# 3D Spatial Privacy

[Background](#background)

[Sample Code](#sample-code)

[Dataset](#dataset)


# ABSTRACT
Augmented reality (AR) or mixed reality (MR) platforms require spatial understanding to detect objects or surfaces, often including their structural (i.e. spatial geometry) and photometric (e.g. color, and texture) attributes, to allow applications to place virtual or synthetic objects seemingly "anchored'' on to real world objects; in some cases, even allowing interactions between the physical and virtual objects. These functionalities requires AR/MR platforms to capture the 3D spatial information with high resolution and frequency; however, this poses unprecedented risks to user privacy. Aside from objects being detected, spatial information also reveals the location of the user with high specificity, e.g. in which part of the house the user is. In this work, we propose to leverage *spatial generalizations* coupled with *conservative releasing* to provide spatial privacy while maintaining data utility. We simulate user movement within spaces which reveals more of their space as they move around. Then, we designed an inference attacker to which the proposed spatial privacy approach can be evaluated against. Results show that revealing no more than 11 generalized planes--accumulated from revealed spaces with large enough radius, i.e. *r*≤1.0m--can make an adversary fail in identifying the spatial location of the user for at least half of the time. Furthermore, if the accumulated spaces are of smaller radius, i.e. *r*≤ 0.5m, we can release up to 29 generalized planes while enjoying both better data utility and privacy.

# BACKGROUND
<p align="center">
  <img src="images/adversary-model-pipeline-v6-revised-a.png" width = "500">
  <br>
  <b>Figure 1:</b> A generic information flow for a desired MR functionality G with an attacker J which can perform adversarial inference off line to determine the space the user is in as well as the objects within the space: (1) adversarial inference modeling or learning from, say, historical 3D data, and (2) adversarial inference or matching over currently released 3D data<br>
</p>

<p align="center">
  <img src="images/adversary-model-pipeline-v6-revised-b.png" width="500">
  <br>
  <b>Figure 2:</b> Inserting an intermediate privacy-preserving mechanism M which aims to prevent spatial inference<br>
</p>

AR/MR platforms such as [Google ARCore](https://developers.google.com/ar/), [Apple ARKit](https://developer.apple.com/documentation/arkit), and [Windows Mixed Reality API](https://www.microsoft.com/en-au/windows/windows-mixed-reality) requires spatial understanding of the user environment in order to deliver virtual augmentations that seemingly inhabit the real world, and, in some immersive examples, even interact with physical objects. The captured spatial information is stored digitally as a spatial map or graph of 3D points, called a *point cloud*, which is accompanied by mesh information to indicate how the points, when connected, represent surfaces and other structures in the user environment. However, these 3D spatial maps that may contain sensitive information can be stored and accessed by a potential adversary (as shown in Fig. 1), and be further utilized for functionalities beyond the application's intended function such as aggressive localized advertisements. And, so far, there are *no* mechanisms in place that ensure user data privacy in MR platforms.

In light of this, first, we present an attacker that not only recognizes the general space, i.e. *inter-space*, but also infers the user’s location within the space, i.e. *intra-space*. To construct the attacker, we build up on existing place recognition methods that have been applied on 3D lidar data and modify it to the scale on which 3D data is captured by MR platforms. We demonstrate how easy it is to extend these 3D recognition methods to be used as an attacker in the MR scenario. Then, we present *spatial plane generalizations* with *conservative plane releasing* as a simple privacy approach which we insert as an intermediary layer of protection as shown in Fig. 2.

# SAMPLE CODE

Pre-requisites
* Python
* Numpy
* Scipy
* HDF5
* Bz2

Sample data is available [here](https://drive.google.com/drive/folders/1IMVuLJxuKeV9HchGY1Wet5IabK2NS4hc) (Download and unpack the ZIP files). Put the testing_samples and testing_results directory within the head directory of this repo after cloning.

The notebook [3D-spatial-privacy-testing](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-testing.ipynb) contains a step-by-step replication of our work but at a smaller scale, i.e. less sample iterations. It uses prepared sample data for various scenarios (i.e. inside testing_samples). The notebook [3D-spatial-privacy-generate-samples](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-generate-samples.ipynb) can be used to generate new samples with varying parameters. As one can inspect, we vary the following parameters on both Raw spaces and [RANSAC] generalized spaces: 
1. the size, i.e radius, of the revealed space
2. the number of successively released partial spaces
3. the number of *generalized* planes released

A separate notebook, i.e. [3D-spatial-privacy-results](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-results.ipynb), just plots the available sample results.

## Manual to [3D-spatial-privacy-generate-samples](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-generate-samples.ipynb)

Step 0: Extract the point cloud from the OBJ files.

Step 0.1: Compute the NN-trees with a set of neighbor size, i.e. nn_range. We use this to generate sample partial spaces.

Step 1: Compute the descriptors of the spaces from the point cloud.

Step 2.1: Generate partial spaces.

Step 2.2: Generate successive partial spaces.


## Manual to [3D-spatial-privacy-testing](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-testing.ipynb)

Step 1.1: Test RAW partial spaces.

Step 1.2: Test RANSAC-generalized partial spaces.

Step 1.3: Results of partial spaces testing.

Step 2.1: Test successive partial spaces.

Step 2.2: Results of successive partial spaces.

Step 3.1: Test successive partial spaces with conservative releasing.

Step 3.2: Results of successive partial spaces with conservative releasing.

## Manual to [3D-spatial-privacy-results](https://github.com/spatial-privacy/spatial-privacy/blob/master/3D-spatial-privacy-results.ipynb)

Note: We are using the uploaded results in bz2 files for this notebook. The results in regular pickle files are not available iin this repo. But, of course, you can rerun the testing to produce new pickle files and use it in this notebook. The file access to the regular pickle files are just commented out, so you can just uncomment them to use them.

Step 1: Results of partial spaces testing.

Step 2: Results of successive partial spaces.

Step 3: Results of successive partial spaces with conservative releasing.

## Sample results

<p align="center">
  <img src="plots/partial-spaces.png" width = "700">
  <br>
  <b>Figure 3:</b> Average inter- and intra-space privacy of partial spaces with varying radius.<br>
</p>

<p align="center">
  <img src="plots/successive-partial-spaces.png" width = "700">
  <br>
  <b>Figure 4:</b> Average inter- and intra-space privacy of successively released partial spaces with increasing number of releases.<br>
</p>


# DATASET
<p align="center">
  <img src="images/all_spaces_sample_3.png" width="500">
  <br>
  <b>Figure 5:</b> 3D point clouds of the 7 collected environments (left); a 3D surface of a sample space (bottom-right), and its 2D-RGB view (top-right)<br>
</p>

For our dataset, we gathered real 3D point cloud data using the Microsoft HoloLens in various environments to demonstrate the leakage from actual human-scale spaces in which an MR device is usually used. As shown in Fig. 5, our collected environments include the following spaces: a work space, a reception area, an office kitchen or pantry, an apartment, a drive way, a hall way, and a stair well.

# PROPOSED MECHANISM

<p align="center">
  <img src="images/partial-releases-2.png" width="300">
  <br>
  <b>Figure 6a:</b> Sample partial spaces of a bigger space<br>
    <img src="images/partial-generalized-releases-2.png" width="275">
  <br>
  <b>Figure 6b:</b> Generalizing the partial spaces<br>
    <img src="images/partial-conservative-releases-2.png" width="275">
  <br>
  <b>Figure 6c:</b> Conservative release of generalized planes<br>
</p>

In this work, we present *conservative plane releasing*, where we limit the number of planes a generalization produces. Fig. 6a shows an example set of planes that are released after RANSAC generalization of the revealed partial raw spaces (in Fig. 6b). Then, we can limit the maximum allowable planes that can be released, say, a maximum of 3 planes in total. As we can see in Fig. 6c, both partial releases produces only 3 planes.
