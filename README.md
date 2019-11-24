# ClearCam: Realtime Imaging through Fog

Current solutions to imaging through fog such as infrared imaging or extracting signals from collected photon distributions are costly and ineffective. We propose a pipeline that directly removes scattering effects from stereo video input. It is based on the Atmospheric Scattering Model, which models scattering as a function of depth and atmospheric parameters. We compute the depth map in realtime via robust stereo algorithms implemented on the GPU and a refine it via a novel post-processing algorithm that compensates for the poor performance of stereo algorithms in low-visibility conditions. We estimate atmospheric parameters using machine learning. Putting the depth and parameters together, we are able to solve for the dehazed image. Our results achieve an average SSIM of 0.95 with respect to the clear references compared to hazy inputs' 0.63.

In addition to the main pipeline, we also explore many different factors the project brought to light. See Project Outline for more details. For more a more detailed description of the entire project, see project.pdf.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

End with an example of getting some data out of the system or using it for a little demo

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

