# Voxelmap
A deep learning framework for patient-specific 3D respiratory motion modelling and volumetric imaging

The key idea behind this framework is that 2D views provide hints about 3D motion. Patient-specific geometric correspondences can be learned from pre-treatment 4D imaging data. Image registration and forward-projection can be used to generate the desired 3D deformation vector fields (DVFs) and 2D projections, which are then used to train a deep neural network. During treatment, a trained neural network can be used to provide insights regarding 3D internal patient anatomy from 2D images acquired in real-time. In particular, the predicted 3D DVF can be used to warp pre-treatment 3D images and contours to provide real-time volumetric imaging as well as the 3D positions of the target and surrounding organs-at-risk.

![Proposed clinical workflow](https://github.com/Image-X-Institute/Voxelmap/blob/main/Workflow.jpg)

This task can be approached in a variety of ways, yielding a number of different network architectures. Here, in every case, we use a residual network with an encoding arm(s) that generates a low-dimensional feature representation of the input images that is then decoded to predict the desired 3D DVF. We also use scaling and squaring layers to integrate the output of the neural network to encourage diffeomorphic mappings.

![Networks](https://github.com/Image-X-Institute/Voxelmap/blob/main/Networks.jpg)

Here we provide code for 5 different neural networks. train_a and test_a are used to train and test Network A respectively, and so on. This repository has benefitted greatly from the excellent Voxelmorph repository. You can check out their work here: https://github.com/voxelmorph/voxelmorph

The inventors have filed a PCT for this work (WO2023215936A1). This code is free to use for academic, non-commercial use. For commercial use, please contact nicholas.hindley@sydney.edu.au.
