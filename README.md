# Voxelmap
A deep learning framework for patient-specific 3D respiratory motion modelling and volumetric imaging

The key idea behind this framework is that 2D views provide hints about 3D motion. Patient-specific geometric correspondences can be learned from pre-treatment 4D imaging data. Image registration and forward-projection can be used to generate the desired 3D deformation vector fields (DVFs) and 2D projections, which are then used to train a deep neural network. During treatment, a trained neural network can be used to provide insights regarding 3D internal patient anatomy from 2D images acquired in real-time.

![Proposed clinical workflow](https://github.com/Image-X-Institute/Voxelmap/blob/main/Workflow.jpg)

This task can be approached in a variety of ways, yielding a number of different neural networks. Here, in every case, we use a residual network with an encoding arm(s) that generates a low-dimensional feature representation of the input images that is then decoded to predict the desired 3D DVF. We also use scaling and squaring layers to integrate the output of the neural network to encourage diffeomorphic mappings.

![Networks](https://github.com/Image-X-Institute/Voxelmap/blob/main/Networks.jpg)

Here we provide code for 5 different neural networks. train_a and test_a are used to train and test Network A respectively, and so on. This repository has benefitted greatly from the excellent Voxelmorph repository. You can check out their work here: https://github.com/voxelmorph/voxelmorph
