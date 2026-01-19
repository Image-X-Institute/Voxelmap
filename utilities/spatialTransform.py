# from voxelmorph repository (https://github.com/voxelmorph/voxelmorph)
from utilities import layers
from utilities.modelio import LoadableModel, store_config_args

class Network(LoadableModel):

    @store_config_args
    def __init__(self, vol_shape):
        """
        Parameters:
            vol_shape: Input volume shape. e.g. (128, 128, 128)
        """
        super().__init__()

        # configure transformer
        self.transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, source_vol, pos_flow):
        '''
        Parameters:
            source_vol: Source volume tensor.
            pos_flow: Input deformation vector field
        '''

        # warp image with flow field
        y_source = self.transformer(source_vol, pos_flow)

        # return warped image
        return y_source
