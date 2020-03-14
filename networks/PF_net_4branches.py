"""
The proposed architecture is a modification of:

https://github.com/DuFanXin/deep_residual_unet

"""


from keras.models import *
from keras.layers import Input, Conv3D, UpSampling3D, BatchNormalization, \
                         Activation, add, concatenate


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='selu')(res_path)
    res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), 
                      padding='same', strides=strides[0])(res_path)
    
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='selu')(res_path)
    res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3, 3), 
                      padding='same', strides=strides[1])(res_path)
    # Create residual unit
    shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)
    
    # Add residual unit with main path 
    res_path = add([shortcut, res_path])
    return res_path


def encoder(x, filters_1):
    to_decoder = []
    
    # Create first unit
    main_path = Conv3D(filters=filters_1, kernel_size=(3, 3, 3), 
                       padding='same', strides=(1, 1, 1))(x) 
    
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='selu')(main_path)
    main_path = Conv3D(filters=filters_1, kernel_size=(3, 3, 3), 
                       padding='same', strides=(1, 1, 1))(main_path)

    shortcut = Conv3D(filters=filters_1, kernel_size=(1, 1, 1), 
                      strides=(1, 1, 1))(x)
    
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    
    to_decoder.append(main_path)
    
    # Create second unit
    main_path = res_block(main_path, [filters_1*2, filters_1*2], 
                          [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters_1*4, filters_1*4], 
                          [(2, 2, 2), (1, 1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, b_sum0, b_sum1, b_sum2, filters_1):
    
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    main_path = concatenate([main_path, b_sum2], axis=4)
    main_path = res_block(main_path, [filters_1*4, filters_1*4], [(1, 1, 1), (1, 1, 1)])


    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, b_sum1], axis=4)    
    main_path = res_block(main_path, [filters_1*2, filters_1*2], [(1, 1, 1), (1, 1, 1)])

   
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    main_path = concatenate([main_path, b_sum0], axis=4) 
    main_path = res_block(main_path, [filters_1, filters_1], [(1, 1, 1), (1, 1, 1)])

    return main_path


def build_PF_net(input_shape0, input_shape1, input_shape2, input_shape3, 
                    filters_1=5):
    """
    Creates the PoreFlow-Net
    - input_shapeX is an array of 4-dims (i.e. (None,None,None,1))
    - filters_1 is an integer with the number of filters in the first layer
      this is doubled after each residual unit
    """
    
    
    # Create the placeholders for the inputs
    inputs0 = Input( shape=input_shape0 )
    inputs1 = Input( shape=input_shape1 )
    inputs2 = Input( shape=input_shape2 )
    inputs3 = Input( shape=input_shape3 )
    
    # Create the encoder branches 
    branch0 = encoder( inputs0, filters_1 )
    branch1 = encoder( inputs1, filters_1 )
    branch2 = encoder( inputs2, filters_1 )
    branch3 = encoder( inputs3, filters_1 )
    
    # Concatenate the residual units of e/branch for the skip connections 
    branch_sum_0 = concatenate( [branch0[0], branch1[0], branch2[0], 
                                 branch3[0] ], axis=4)
    
    branch_sum_1 = concatenate( [branch0[1], branch1[1], branch2[1], 
                                 branch3[1] ], axis=4)
    
    branch_sum_2 = concatenate( [branch0[2], branch1[2], branch2[2], 
                                 branch3[2] ], axis=4)
  
    # Create bridge between encoder and decoder
    path = res_block(branch_sum_2, [filters_1*8, filters_1*8, filters_1*8], 
                     [(2, 2, 2), (1, 1, 1)])

    # Create decoder branch
    path = decoder(path, branch_sum_0, branch_sum_1, branch_sum_2, filters_1)

    # Last filter, this outputs the velocity in Z-direction
    # for pressure or the full velocity tensor, one could change the 
    # number of filters to > 1
    path = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='selu')(path)

    return Model(inputs=[inputs0,inputs1,inputs2,inputs3], outputs=path)














