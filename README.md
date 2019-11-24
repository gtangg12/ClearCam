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

Copy the file from here: https://gist.github.com/reza-ryte-club/97c39f35dab0c45a5d924dd9e50c445f.

### Running Demo
```
sh run.txt
```

## Project Outline

### main.cpp

### stereo_data

### final_model

### parameter_fitting

### outdoor

### middlebury_haze

