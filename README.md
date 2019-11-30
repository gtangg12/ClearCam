# ClearCam: Realtime Imaging through Fog

Current solutions to imaging through fog such as infrared imaging or extracting signals from collected photon distributions are costly and ineffective. We propose a pipeline that directly removes scattering effects from stereo video input. It is based on the Atmospheric Scattering Model, which models scattering as a function of depth and atmospheric parameters. We compute the depth map in realtime via robust stereo algorithms implemented on the GPU and a refine it via a novel post-processing algorithm that compensates for the poor performance of stereo algorithms in low-visibility conditions. We estimate atmospheric parameters using machine learning. Putting the depth and parameters together, we are able to solve for the dehazed image. Our results achieve an average SSIM of 0.95 with respect to the clear references compared to hazy inputs' 0.63.

In addition to the main pipeline, we also explore many different factors the project brought to light. See Project Outline for more details. For more a more detailed description of the entire project, see project.pdf.

## Getting Started

These instructions will allow you to run a demo of the main pipeline on MacOS.

### Prerequisites

You must have the following libraries. To install them, run the folowing commands:

```
pip3 install opencv-python
pip3 install tensorflow
pip3 install keras
brew install cmake
brew install opencv
```

In addition, MacOS lacks the #include <bits/stdc++.h> header. Do the following to add it

```
cd /usr/local/include/
mkdir bits
```
Copy the file from here into bits: https://gist.github.com/reza-ryte-club/97c39f35dab0c45a5d924dd9e50c445f.

Next do
```
cmake .
make
```

### Running Demo
```
sh run.txt
```

This will compute the disparity map from the stereo images in demo/. The result will be displayed; press return to continue. Now, the parameters will be estimated using the model from final_model/. The hazy image, dehazed image, and clear reference (right image of stereo images) will be displayed. If you only see one image, the others are hidden behind it, so just drag them apart. 

## Project Outline

### main.cpp
Computes the disparity from stereo images.

### stereo_data
Contains the Middlebury 2006 Dataset that we used to syntheize hazy data, which are stored in hazy_images/. The respective computed disparity maps you see in project.pdf can be found in disparity_maps/.

### final_model
Contains the code for and the final machine learning model we used in estimating the beta parameter.

### parameter_fitting
Contains code of different machine learning approaches to beta estimation. 

### outdoor
Contains code of non machine learning approaches to beta estimation that is better suited in outdoor conditions.

### middlebury_haze
Contains code for synthezing haze into clear references as well as approaches to synthesize non-homogeneous fog via noise insertion.

