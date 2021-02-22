# ESBN
Exposure-Structure Blending Network for High Dynamic Range Imaging of Dynamic Scenes\
Sang-hoon Lee, Haesoo Chung, Nam Ik Cho

## Introduction
This repository provides a code for an HDR imaging algorithm and result images described in our accepted IEEE ACCESS paper.

## Usage
1. Prepare the Kalantari dataset
2. Make tfrecord files for training samples using "make_tfrecord.py"
3. Train the alignment networks and the merging network using "training_align_high.py", "training_align_low.py" and "training_fusion.py"
4. Reconstruct the aligned images and fused hdr images using "reconstruction_align2.py" and "reconstruction_fusion.py"
* You should change the file pathes in the codes.

## Abstract
This paper presents a deep end-to-end network for high dynamic range (HDR) imaging of dynamic scenes with background and foreground motions. Generating an HDR image from a sequence of multi-exposure images is a challenging process when the images have misalignments by being taken in a dynamic situation. Hence, recent methods first align the multi-exposure images to the reference by using patch matching, optical flow, homography transformation, or attention module before the merging. In this paper, we propose a deep network that synthesizes the aligned images as a result of blending the information from multi-exposure images, because explicitly aligning photos with different exposures is inherently a difficult problem. Specifically, the proposed network generates under/over-exposure images that are structurally aligned to the reference, by blending all the information from the dynamic multiexposure images. Our primary idea is that blending two images in the deep-feature-domain is effective for synthesizing multi-exposure images that are structurally aligned to the reference, resulting in betteraligned images than the pixel-domain blending or geometric transformation methods. Specifically, our alignment network consists of a two-way encoder for extracting features from two images separately, several convolution layers for blending deep features, and a decoder for constructing the aligned images. The proposed network is shown to generate the aligned images with a wide range of exposure differences very well and thus can be effectively used for the HDR imaging of dynamic scenes. Moreover, by adding a simple merging network after the alignment network and training the overall system end-to-end, we obtain a performance gain compared to the recent state-of-the-art methods.

## Citation
```
@article{lee2020exposure,
  title={Exposure-Structure Blending Network for High Dynamic Range Imaging of Dynamic Scenes},
  author={Lee, Sang-Hoon and Chung, Haesoo and Cho, Nam Ik},
  journal={IEEE Access},
  volume={8},
  pages={117428--117438},
  year={2020},
  publisher={IEEE}
}
```
