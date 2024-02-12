# Voxelmap
A deep learning framework for patient-specific 3D respiratory motion modelling

The key idea behind this framework is that 2D views provide hints about 3D motion. Patient-specific geometric correspondences can be learned from pre-treatment 4D imaging data. Image registration and forward-projection can be used to generate the desired 3D deformation vector fields (DVFs) and 2D projections, which are then used to train a deep neural network.

![Proposed clinical workflow](https://github.com/Image-X-Institute/Voxelmap/blob/main/Workflow.jpg)
