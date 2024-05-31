"""
A collection of models we'll use to attempt to classify videos.
"""
import numpy as np
import tensorflow as tf
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Input, Dense, Flatten, Dropout, ZeroPadding3D,Activation
from keras.layers.recurrent import LSTM
from keras.layers import Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.layers import Lambda
from keras.layers.merge import concatenate, average
from collections import deque
import sys
import os.path
from keras.engine.topology import Layer
#from spp.SpatialPyramidPooling import SpatialPyramidPooling
import keras.backend as K




def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def mat_mul(A, B):
    return tf.matmul(A, B)

class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def slice(x, h1,h2):
    return x[:,h1:h2]

class ResearchModels():
    #1536
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=1536):



        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            #self.model = load_model(self.saved_model)
            self.input_shape = (seq_length, features_length)
            self.model = self.stream()
            self.model.load_weights(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
            #self.model.load_weights(os.path.join('data', 'checkpoints','lstm-features.051-0.559.hdf5'), by_name=True)

        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
            #self.model.load_weights(os.path.join('data', 'checkpoints','mlp-features.001-0.410.hdf5'), by_name=True)


        elif model == '2stream':
            print("Loading 2stream.")
            self.input_shape = (seq_length, features_length)
            self.model = self.stream()
            #self.model.load_weights(os.path.join('data', 'checkpoints','mlp-features.022-0.446.hdf5'), by_name=True)
            #self.model.load_weights(os.path.join('data', 'checkpoints','point-point.001-0.588.hdf5'), by_name=True)
            #self.model.load_weights(os.path.join('data', 'checkpoints','trajectory-trajectory.048-0.462.hdf5'), by_name=True)
            #for layer in self.model.layers[:-9]:
            #layer.trainable = False

        elif model == 'trajectory':
            print("Loading tra_pointnet.")
            #self.input_shape = (seq_length, features_length)
            self.model = self.trajectory()
            #self.model.load_weights(os.path.join('data', 'checkpoints','trajectory-trajectory.266-0.761.hdf5'), by_name=True)
            #self.model.load_weights(os.path.join('data', 'checkpoints','08trajectory-trajectory.152-0.766.hdf5'), by_name=True)
            #self.model.load_weights(os.path.join('data', 'checkpoints','02trajectory-trajectory.063-0.794.hdf5'), by_name=True)
           # for layer in self.model.layers[:-12]:
        #print layer
        #   layer.trainable = False
        elif model == 'point':
            print("Loading pointnet.")
            #self.input_shape = (seq_length, features_length)
            self.model = self.point()
            #self.model.load_weights(os.path.join('data', 'checkpoints','point-point.545-0.840.hdf5'), by_name=True)
        elif model == '3stream':
            print("Loading 3stream.")
            self.input_shape = (seq_length, features_length)
            self.model = self.t_stream()
            #self.model.load_weights(os.path.join('data', 'checkpoints','3stream-3stream.475-0.549.hdf5'), by_name=True)
            self.model.load_weights(os.path.join('data', 'checkpoints','mlp-features.083-0.577.hdf5'), by_name=True)
            self.model.load_weights(os.path.join('data', 'checkpoints','point-point.050-0.814.hdf5'), by_name=True)
            self.model.load_weights(os.path.join('data', 'checkpoints','trajectory-trajectory.061-0.762.hdf5'), by_name=True)
            for layer in self.model.layers[:-13]:
                print (layer)
                layer.trainable = False
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=0.01, decay=0.000001)
        #optimizer = SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=True)
#        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,
#                           metrics=metrics)
#        self.model.compile(loss={'t_out_1':'categorical_crossentropy',
#                                 't_out_2':'categorical_crossentropy'},
#                           loss_weights={'t_out_1':0.7, 't_out_2':0.3},
#                           optimizer=optimizer,
#                           metrics=metrics)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(get_flops(self.model))
        print(self.model.summary())

 

    def trajectory(self):
        num_tra = 1024

        # number of categories
        k = 6
        dim_tra = 7
        
        # ------------------------------------ Pointnet Architecture
        # input_Transformation_net
        input_tra = Input(shape=(num_tra, dim_tra), name='tra_input_1')


        # forward net
    
        g2 = Convolution1D(32, 3, strides=2, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_21')(input_tra)
        g2 = BatchNormalization(name='tra_batch_normalization_21')(g2)
        g2 = Convolution1D(64, 3, strides=2, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_22')(g2)
        g2 = BatchNormalization(name='tra_batch_normalization_22')(g2)

        # feature transform net
        f2 = Convolution1D(128, 1, activation='relu', name='tra_conv1d_23')(g2)
        f2 = BatchNormalization(name='tra_batch_normalization_23')(f2)
        f2 = Convolution1D(256, 1, activation='relu', name='tra_conv1d_24')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_24')(f2)
        #f2 = Convolution1D(1024, 1, activation='relu', name='tra_conv1d_25')(f2)
        #f2 = BatchNormalization(name='tra_batch_normalization_25')(f2)
    
        # global_feature
        #global_feature2 = MaxPooling1D(pool_size=num_tra, name='tra_maxpooling1d_1')(f2)



        #f2 = Convolution1D(1024, 3, strides=1,  activation='relu', name='tra_conv1d_5')(f2)
        #f2 = BatchNormalization(name='tra_batch_normalization_5')(f2)
    
        #f2 = SpatialPyramidPooling([1])(f2)
        #f2 = MaxPooling1D(pool_size=255, name='tra_maxpooling1d_1')(f2)
        f2 = Flatten(name='tra_flatten_20')(f2)

        y2 = Dense(512, activation='relu', name='tra_dense_1')(f2)
        y2 = BatchNormalization(name='tra_batch_normalization_6')(y2)
        y2 = Dropout(rate=0.7, name='tra_dropout_1')(y2)
        y2 = Dense(256, activation='relu', name='tra_dense_2')(y2)
        y2 = BatchNormalization(name='tra_batch_normalization_7')(y2)
        y2 = Dropout(rate=0.7, name='tra_dropout_2')(y2)
        y2 = Dense(k, activation='relu', name='tra_dense_3')(y2)
        #prediction = Flatten(name='tra_flatten_1')(y2)
        prediction = y2

        #model = Model(inputs=input_tra, outputs=prediction)


        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='cnn_lam_1')(prediction)
        prediction1 = Dense(2, activation='softmax', name='cnn_out_1')(prediction1)

        prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='cnn_lam_2')(prediction)
        prediction2 = Dense(4, activation='softmax', name='cnn_out_2')(prediction2)
        model = Model(inputs=input_tra, outputs=[prediction1, prediction2])

        return model

 

    def point(self):
        num_points = 4096

        # number of categories
        k = 6
        dim_points = 3
        
        # ------------------------------------ Pointnet Architecture
        # input_Transformation_net
        input_points = Input(shape=(num_points, dim_points), name='pcl_input_1')
        

        x = Convolution1D(64, 1, activation='relu', input_shape=(num_points, 3), name='pclit_conv1d_1')(input_points)
        x = BatchNormalization(name='pclit_batch_normalization_1')(x)
        x = Convolution1D(128, 1, activation='relu', name='pclit_conv1d_2')(x)
        x = BatchNormalization(name='pclit_batch_normalization_2')(x)
        x = Convolution1D(1024, 1, activation='relu', name='pclit_conv1d_3')(x)
        x = BatchNormalization(name='pclit_batch_normalization_3')(x)
        x = MaxPooling1D(pool_size=num_points, name='pclit_maxpooling1d_1')(x)
        x = Dense(512, activation='relu', name='pclit_dense_1')(x)
        x = BatchNormalization(name='pclit_batch_normalization_4')(x)
        x = Dense(256, activation='relu', name='pclit_dense_2')(x)
        x = BatchNormalization(name='pclit_batch_normalization_5')(x)
        x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)], name='pclit_dense_3')(x)
        input_T = Reshape((3, 3))(x)

        # forward net
        g = MatMul(name='pcl_mul_1')([input_points, input_T])
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_1')(g)
        g = BatchNormalization(name='pcl_batch_normalization_1')(g)
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_2')(g)
        g = BatchNormalization(name='pcl_batch_normalization_2')(g)

        # feature transform net
        f = Convolution1D(64, 1, activation='relu', name='pclft_conv1d_1')(g)
        f = BatchNormalization(name='pclft_batch_normalization_1')(f)
        f = Convolution1D(128, 1, activation='relu', name='pclft_conv1d_2')(f)
        f = BatchNormalization(name='pclft_batch_normalization_2')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pclft_conv1d_3')(f)
        f = BatchNormalization(name='pclft_batch_normalization_3')(f)
        f = MaxPooling1D(pool_size=num_points, name='pclft_maxpooling1d_1')(f)
        f = Dense(512, activation='relu', name='pclft_dense_1')(f)
        f = BatchNormalization(name='pclft_batch_normalization_4')(f)
        f = Dense(256, activation='relu', name='pclft_dense_2')(f)
        f = BatchNormalization(name='pclft_batch_normalization_5')(f)
        f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)], name='pclft_dense_3')(f)
        feature_T = Reshape((64, 64))(f)
        g = MatMul(name='pcl_mul_2')([g, feature_T])


        f = Convolution1D(64, 1, activation='relu', name='pcl_conv1d_3')(g)
        f = BatchNormalization(name='pcl_batch_normalization_3')(f)
        f = Convolution1D(128, 1, activation='relu', name='pcl_conv1d_4')(f)
        f = BatchNormalization(name='pcl_batch_normalization_4')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pcl_conv1d_5')(f)
        f = BatchNormalization(name='pcl_batch_normalization_5')(f)
        
        # global_feature
        global_feature = MaxPooling1D(pool_size=num_points, name='pcl_maxpooling1d_1')(f)
        #global_feature = LSTM(1024, return_sequences=False,dropout=0.5)(f)

        y = Dense(512, activation='relu', name='pcl_dense_1')(global_feature)
        y = BatchNormalization(name='pcl_batch_normalization_6')(y)
        y = Dropout(rate=0.7, name='pcl_dropout_1')(y)
        y = Dense(256, activation='relu', name='pcl_dense_2')(y)
        y = BatchNormalization(name='pcl_batch_normalization_7')(y)
        y = Dropout(rate=0.7, name='pcl_dropout_2')(y)
        y = Dense(k, activation='relu', name='pcl_dense_3')(y)
        prediction = Flatten(name='pcl_flatten_1')(y)

        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='cnn_lam_1')(prediction)
        prediction1 = Dense(2, activation='softmax', name='cnn_out_1')(prediction1)
    
        prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='cnn_lam_2')(prediction)
        prediction2 = Dense(4, activation='softmax', name='cnn_out_2')(prediction2)
        model = Model(inputs=input_points, outputs=[prediction1, prediction2])



        return model



    def t_stream_latefusion(self):
    # number of categories
        k = 1
    # -------------------------------------------------------------------------------------------------------------------CNN
        feature_input = Input(shape=self.input_shape, name='cnn_input_1')
        fx = Flatten(name='cnn_flatten_1')(feature_input)
        fx = Dense(512, activation='relu', name='cnn_dense_1')(fx)
        fx = Dropout(0.5, name='cnn_dropout_1')(fx)
        fx = Dense(512, activation='relu', name='cnn_dense_2')(fx)
        fx = Dropout(0.5, name='cnn_dropout_2')(fx)
        fx = Dense(k, activation='sigmoid', name='cnn_dense_3')(fx)

    # ------------------------------------------------------------------------------------------------------------- pointcloud
        num_points = 4096

        dim_points = 3
    # input_Transformation_net
        input_points = Input(shape=(num_points, dim_points), name='pcl_input_1')

        # forward net
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_1')(input_points)
        g = BatchNormalization(name='pcl_batch_normalization_1')(g)
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_2')(g)
        g = BatchNormalization(name='pcl_batch_normalization_2')(g)

        # feature transform net
        f = Convolution1D(64, 1, activation='relu', name='pcl_conv1d_3')(g)
        f = BatchNormalization(name='pcl_batch_normalization_3')(f)
        f = Convolution1D(128, 1, activation='relu', name='pcl_conv1d_4')(f)
        f = BatchNormalization(name='pcl_batch_normalization_4')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pcl_conv1d_5')(f)
        f = BatchNormalization(name='pcl_batch_normalization_5')(f)
        
        # global_feature
        global_feature = MaxPooling1D(pool_size=num_points, name='pcl_maxpooling1d_1')(f)
        c = Dense(512, activation='relu', name='pcl_dense_1')(global_feature)
        c = BatchNormalization(name='pcl_batch_normalization_6')(c)
        c = Dropout(0.7, name='pcl_dropout_1')(c)
        c = Dense(256, activation='relu', name='pcl_dense_2')(c)
        c = BatchNormalization(name='pcl_batch_normalization_7')(c)
        c = Dropout(rate=0.7, name='pcl_dropout_2')(c)
        c = Dense(k, activation='sigmoid', name='pcl_dense_3')(c)
        c = Flatten(name='pcl_flatten_1')(c)
        # -------------------------------------------------------------------------------------------------------------- trajectory
        num_tra = 1024

        dim_tra = 7
        # input_Transformation_net
        input_tra = Input(shape=(num_tra, dim_tra), name='tra_input_1')

        # forward net
        g2 = Convolution1D(64, 1, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_1')(input_tra)
        g2 = BatchNormalization(name='tra_batch_normalization_1')(g2)
        g2 = Convolution1D(64, 1, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_2')(g2)
        g2 = BatchNormalization(name='tra_batch_normalization_2')(g2)

        # feature transform net
        f2 = Convolution1D(64, 1, activation='relu', name='tra_conv1d_3')(g2)
        f2 = BatchNormalization(name='tra_batch_normalization_3')(f2)
        f2 = Convolution1D(128, 1, activation='relu', name='tra_conv1d_4')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_4')(f2)
        f2 = Convolution1D(1024, 1, activation='relu', name='tra_conv1d_5')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_5')(f2)
        
        # global_feature
        global_feature2 = MaxPooling1D(pool_size=num_tra, name='tra_maxpooling1d_1')(f2)
        c2 = Dense(512, activation='relu', name='tra_dense_1')(global_feature2)
        c2 = BatchNormalization(name='tra_batch_normalization_6')(c2)
        c2 = Dropout(0.7, name='tra_dropout_1')(c2)
        c2 = Dense(256, activation='relu', name='tra_dense_2')(c2)
        c2 = BatchNormalization(name='tra_batch_normalization_7')(c2)
        c2 = Dropout(rate=0.7, name='tra_dropout_2')(c2)
        c2 = Dense(k, activation='sigmoid', name='tra_dense_3')(c2)
        c2 = Flatten(name='tra_flatten_1')(c2)



        y3 = average([fx, c, c2], name='t_average_1')

        prediction = y3


        model = Model(inputs=[feature_input, input_points, input_tra], outputs=prediction)

        return model

    def t_stream(self):
    # number of categories
        k = 6
    # -------------------------------------------------------------------------------------------------------------------CNN
        feature_input = Input(shape=self.input_shape, name='cnn_input_1')
        fx = Flatten(name='cnn_flatten_1')(feature_input)
        fx = Dense(512, activation='relu', name='cnn_dense_1')(fx)
        fx = Dropout(0.5, name='cnn_dropout_1')(fx)
        fx = Dense(512, activation='relu', name='cnn_dense_2')(fx)
        fx = Dropout(0.5, name='cnn_dropout_2')(fx)
        fx = Reshape((-1, 512), name='cnn_reshape_1')(fx)

    # ------------------------------------------------------------------------------------------------------------- pointcloud
        num_points = 4096

        dim_points = 3
    # input_Transformation_net
        input_points = Input(shape=(num_points, dim_points), name='pcl_input_1')


        x = Convolution1D(64, 1, activation='relu', input_shape=(num_points, 3), name='pclit_conv1d_1')(input_points)
        x = BatchNormalization(name='pclit_batch_normalization_1')(x)
        x = Convolution1D(128, 1, activation='relu', name='pclit_conv1d_2')(x)
        x = BatchNormalization(name='pclit_batch_normalization_2')(x)
        x = Convolution1D(1024, 1, activation='relu', name='pclit_conv1d_3')(x)
        x = BatchNormalization(name='pclit_batch_normalization_3')(x)
        x = MaxPooling1D(pool_size=num_points, name='pclit_maxpooling1d_1')(x)
        x = Dense(512, activation='relu', name='pclit_dense_1')(x)
        x = BatchNormalization(name='pclit_batch_normalization_4')(x)
        x = Dense(256, activation='relu', name='pclit_dense_2')(x)
        x = BatchNormalization(name='pclit_batch_normalization_5')(x)
        x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)], name='pclit_dense_3')(x)
        input_T = Reshape((3, 3))(x)

        # forward net
        g = MatMul(name='pcl_mul_1')([input_points, input_T])
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_1')(g)
        g = BatchNormalization(name='pcl_batch_normalization_1')(g)
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_2')(g)
        g = BatchNormalization(name='pcl_batch_normalization_2')(g)

        # feature transform net
        f = Convolution1D(64, 1, activation='relu', name='pclft_conv1d_1')(g)
        f = BatchNormalization(name='pclft_batch_normalization_1')(f)
        f = Convolution1D(128, 1, activation='relu', name='pclft_conv1d_2')(f)
        f = BatchNormalization(name='pclft_batch_normalization_2')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pclft_conv1d_3')(f)
        f = BatchNormalization(name='pclft_batch_normalization_3')(f)
        f = MaxPooling1D(pool_size=num_points, name='pclft_maxpooling1d_1')(f)
        f = Dense(512, activation='relu', name='pclft_dense_1')(f)
        f = BatchNormalization(name='pclft_batch_normalization_4')(f)
        f = Dense(256, activation='relu', name='pclft_dense_2')(f)
        f = BatchNormalization(name='pclft_batch_normalization_5')(f)
        f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)], name='pclft_dense_3')(f)
        feature_T = Reshape((64, 64))(f)
        g = MatMul(name='pcl_mul_2')([g, feature_T])


        f = Convolution1D(64, 1, activation='relu', name='pcl_conv1d_3')(g)
        f = BatchNormalization(name='pcl_batch_normalization_3')(f)
        f = Convolution1D(128, 1, activation='relu', name='pcl_conv1d_4')(f)
        f = BatchNormalization(name='pcl_batch_normalization_4')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pcl_conv1d_5')(f)
        f = BatchNormalization(name='pcl_batch_normalization_5')(f)
        
        
        # global_feature
        global_feature = MaxPooling1D(pool_size=num_points, name='pcl_maxpooling1d_1')(f)
        c = Dense(512, activation='relu', name='pcl_dense_1')(global_feature)
        c = BatchNormalization(name='pcl_batch_normalization_6')(c)
        c = Dropout(0.5, name='pcl_dropout_1')(c)
        c = Reshape((-1, 512), name='pcl_reshape_1')(c)

        # -------------------------------------------------------------------------------------------------------------- trajectory
        num_tra = 1024

        dim_tra = 7
        # input_Transformation_net
        input_tra = Input(shape=(num_tra, dim_tra), name='tra_input_1')

        # forward net
        g2 = Convolution1D(64, 3, strides=2, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_1')(input_tra)
        g2 = BatchNormalization(name='tra_batch_normalization_1')(g2)
        g2 = Convolution1D(128, 3, strides=2, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_2')(g2)
        g2 = BatchNormalization(name='tra_batch_normalization_2')(g2)

        # feature transform net
        f2 = Convolution1D(256, 1, activation='relu', name='tra_conv1d_3')(g2)
        f2 = BatchNormalization(name='tra_batch_normalization_3')(f2)
        f2 = Convolution1D(512, 1, activation='relu', name='tra_conv1d_4')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_4')(f2)
        #f2 = Convolution1D(1024, 1, activation='relu', name='tra_conv1d_5')(f2)
        #f2 = BatchNormalization(name='tra_batch_normalization_5')(f2)
        
        # global_feature
        #global_feature2 = MaxPooling1D(pool_size=num_tra, name='tra_maxpooling1d_1')(f2)
        global_feature2 = Flatten(name='tra_flatten_0')(f2)
        c2 = Dense(512, activation='relu', name='tra_dense_1')(global_feature2)
        c2 = BatchNormalization(name='tra_batch_normalization_6')(c2)
        c2 = Dropout(0.5, name='tra_dropout_1')(c2)
        c2 = Reshape((-1, 512), name='tra_reshape_1')(c2)



        y3 = concatenate([fx, c, c2], name='t_concatenate_1')

        y3 = Dense(512, activation='relu', name='t_dense_1')(y3)
        y3 = BatchNormalization(name='t_batch_normalization_1')(y3)
        y3 = Dropout(rate=0.7, name='t_dropout_1')(y3)
        y3 = Dense(256, activation='relu', name='t_dense_2')(y3)
        y3 = BatchNormalization(name='t_batch_normalization_2')(y3)
        y3 = Dropout(rate=0.7, name='t_dropout_2')(y3)
        y3 = Dense(k, activation='relu', name='t_dense_3')(y3)


        y3 = Flatten(name='t_flatten_1')(y3)

        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='t_lam_1')(y3)
        prediction1 = Dense(2, activation='softmax', name='t_out_1')(prediction1)

        prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='t_lam_2')(y3)
        prediction2 = Dense(4, activation='softmax', name='t_out_2')(prediction2)
        model = Model(inputs=[feature_input, input_points, input_tra], outputs=[prediction1, prediction2])

        return model


 

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        #model = Sequential()
        #model.add(LSTM(2048, return_sequences=False,input_shape=self.input_shape,dropout=0.5))
        #model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(1, activation='sigmoid'))

        k = 6

        feature_input = Input(shape=self.input_shape)
        fx = LSTM(2048, return_sequences=False,dropout=0.5)(feature_input)
        fx = Dense(512, activation='relu')(fx)
        fx = Dropout(0.5)(fx)
        #y = Dense(k, activation='sigmoid')(fx)
        #prediction = y
        

        #model = Model(inputs=feature_input, outputs=prediction)


        y = Dense(k, activation='relu')(fx)
        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='cnn_lam_1')(y)
        prediction1 = Dense(2, activation='softmax', name='cnn_out_1')(prediction1)

        #prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='cnn_lam_2')(y)
        #prediction2 = Dense(4, activation='softmax', name='cnn_out_2')(prediction2)
        model = Model(inputs=feature_input, outputs=prediction1)
        return model



    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        #model = Sequential()
        #model.add(Flatten(input_shape=self.input_shape))
        #model.add(Dense(512))
        #model.add(Dropout(0.5))
        #model.add(Dense(512))
        #model.add(Dropout(0.5))
        #model.add(Dense(1, activation='sigmoid'))
        k = 6

        feature_input = Input(shape=self.input_shape, name='cnn_input_1')
        fx = Flatten(name='cnn_flatten_1')(feature_input)
        fx = Dense(512, activation='relu', name='cnn_dense_1')(fx)
        fx = Dropout(0.5, name='cnn_dropout_1')(fx)
        fx = Dense(512, activation='relu', name='cnn_dense_2')(fx)
        fx = Dropout(0.5, name='cnn_dropout_2')(fx)
        #y = Dense(k, activation='sigmoid', name='cnn_dense_3')(fx)
        y = Dense(k, activation='relu', name='cnn_dense_3')(fx)
        #prediction = y

        #model = Model(inputs=feature_input, outputs=prediction)



        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='cnn_lam_1')(y)
        prediction1 = Dense(2, activation='sigmoid', name='cnn_out_1')(prediction1)

        prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='cnn_lam_2')(y)
        prediction2 = Dense(4, activation='softmax', name='cnn_out_2')(prediction2)
        model = Model(inputs=feature_input, outputs=[prediction1, prediction2])

        return model

    def stream(self):

    # number of categories
        k = 1
    # -------------------------------------------------------------------------------------------------------------------CNN
        feature_input = Input(shape=self.input_shape, name='cnn_input_1')
        fx = Flatten(name='cnn_flatten_1')(feature_input)
        fx = Dense(512, activation='relu', name='cnn_dense_1')(fx)
        fx = Dropout(0.5, name='cnn_dropout_1')(fx)
        fx = Dense(512, activation='relu', name='cnn_dense_2')(fx)
        fx = Dropout(0.5, name='cnn_dropout_2')(fx)
        fx = Reshape((-1, 512), name='cnn_reshape_1')(fx)

    # ------------------------------------------------------------------------------------------------------------- pointcloud
        num_points = 4096

        dim_points = 3
        # input_Transformation_net
        input_points = Input(shape=(num_points, dim_points), name='pcl_input_1')

        # forward net
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_1')(input_points)
        g = BatchNormalization(name='pcl_batch_normalization_1')(g)
        g = Convolution1D(64, 1, input_shape=(num_points, dim_points), activation='relu', name='pcl_conv1d_2')(g)
        g = BatchNormalization(name='pcl_batch_normalization_2')(g)

        # feature transform net
        f = Convolution1D(64, 1, activation='relu', name='pcl_conv1d_3')(g)
        f = BatchNormalization(name='pcl_batch_normalization_3')(f)
        f = Convolution1D(128, 1, activation='relu', name='pcl_conv1d_4')(f)
        f = BatchNormalization(name='pcl_batch_normalization_4')(f)
        f = Convolution1D(1024, 1, activation='relu', name='pcl_conv1d_5')(f)
        f = BatchNormalization(name='pcl_batch_normalization_5')(f)
        
        # global_feature
        global_feature = MaxPooling1D(pool_size=num_points, name='pcl_maxpooling1d_1')(f)
        c = Dense(512, activation='relu', name='pcl_dense_1')(global_feature)
        c = BatchNormalization(name='pcl_batch_normalization_6')(c)
        c = Dropout(0.5, name='pcl_dropout_1')(c)
        c = Reshape((-1, 512), name='pcl_reshape_1')(c)

        # -------------------------------------------------------------------------------------------------------------- trajectory
        num_tra = 1024

        dim_tra = 7
        # input_Transformation_net
        input_tra = Input(shape=(num_tra, dim_tra), name='tra_input_1')

        # forward net
        g2 = Convolution1D(64, 1, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_1')(input_tra)
        g2 = BatchNormalization(name='tra_batch_normalization_1')(g2)
        g2 = Convolution1D(64, 1, input_shape=(num_tra, dim_tra), activation='relu', name='tra_conv1d_2')(g2)
        g2 = BatchNormalization(name='tra_batch_normalization_2')(g2)

        # feature transform net
        f2 = Convolution1D(64, 1, activation='relu', name='tra_conv1d_3')(g2)
        f2 = BatchNormalization(name='tra_batch_normalization_3')(f2)
        f2 = Convolution1D(128, 1, activation='relu', name='tra_conv1d_4')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_4')(f2)
        f2 = Convolution1D(1024, 1, activation='relu', name='tra_conv1d_5')(f2)
        f2 = BatchNormalization(name='tra_batch_normalization_5')(f2)
        
        # global_feature
        global_feature2 = MaxPooling1D(pool_size=num_tra, name='tra_maxpooling1d_1')(f2)
        c2 = Dense(512, activation='relu', name='tra_dense_1')(global_feature2)
        c2 = BatchNormalization(name='tra_batch_normalization_6')(c2)
        c2 = Dropout(0.5, name='tra_dropout_1')(c2)
        c2 = Reshape((-1, 512), name='tra_reshape_1')(c2)



        y3 = concatenate([fx, c2], name='t_concatenate_1')

        y3 = Dense(512, activation='relu', name='t_dense_1')(y3)
        y3 = BatchNormalization(name='t_batch_normalization_1')(y3)
        y3 = Dropout(rate=0.7, name='t_dropout_1')(y3)
        y3 = Dense(256, activation='relu', name='t_dense_2')(y3)
        y3 = BatchNormalization(name='t_batch_normalization_2')(y3)
        y3 = Dropout(rate=0.7, name='t_dropout_2')(y3)
        y3 = Dense(k, activation='sigmoid', name='t_dense_3')(y3)
        y3 = Flatten(name='t_flatten_1')(y3)
        #prediction = Flatten(name='t_flatten_1')(y3)

        prediction1 = Lambda(slice, arguments={'h1':0, 'h2':2}, name='t_lam_1')(y3)
        prediction1 = Dense(2, activation='softmax', name='t_out_1')(prediction1)

        #prediction2 = Lambda(slice, arguments={'h1':2, 'h2':6}, name='t_lam_2')(y3)
        #prediction2 = Dense(4, activation='softmax', name='t_out_2')(prediction2)

        model = Model(inputs=[feature_input, input_tra], outputs=prediction1)
        #model = Model(inputs=[feature_input, input_tra], outputs=prediction)

        return model