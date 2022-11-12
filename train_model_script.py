from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import pandas as pd

# optimizer = Adam(lr=0.001) 


def load_train(path):
    
    labels = pd.read_csv(path+'/labels.csv')
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                      validation_split=0.2,
                                      horizontal_flip=True)
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path+'/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345) 

    return  train_gen_flow

def load_test(path):
    
    labels = pd.read_csv(path+'/labels.csv')
    
    test_datagen = ImageDataGenerator(rescale=1./255,
                                      validation_split=0.2)
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path+'/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345) 

    return  test_gen_flow


def create_model(input_shape):
    
    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                        # weights='imagenet',
                    include_top=False)
    # backbone.trainable = False
    

    model = Sequential()

    model.add(backbone)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.compile(optimizer=Adam(lr=0.0001), loss='mse',
                  metrics=['mae'])    

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=10,
                steps_per_epoch=None, validation_steps=None):
    
    callback = EarlyStopping(monitor='loss', patience=1)
    
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, 
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              callbacks=[callback],
              verbose=2)
    
    return model




# Found 7591 validated image filenames.
# Found 1518 validated image filenames.
# 2022-10-10 15:15:27.371809: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
# 2022-10-10 15:15:27.409584: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2099995000 Hz
# 2022-10-10 15:15:27.411660: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4d83cb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2022-10-10 15:15:27.411694: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2022-10-10 15:15:27.641352: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x43e3fd0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2022-10-10 15:15:27.641399: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0
# 2022-10-10 15:15:27.643833: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-10-10 15:15:27.643901: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-10-10 15:15:27.643911: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-10-10 15:15:27.643944: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-10-10 15:15:27.643956: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-10-10 15:15:27.643966: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-10-10 15:15:27.643977: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-10-10 15:15:27.643986: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-10-10 15:15:27.648410: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-10-10 15:15:27.653731: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-10-10 15:15:31.103144: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2022-10-10 15:15:31.103196: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
# 2022-10-10 15:15:31.103207: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
# 2022-10-10 15:15:31.107883: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2022-10-10 15:15:31.107937: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10240 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8b:00.0, compute capability: 7.0)
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 238 steps, validate for 48 steps
# Epoch 1/4
# 2022-10-10 15:15:54.978898: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-10-10 15:15:57.154295: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 238/238 - 75s - loss: 218.1275 - mae: 10.4028 - val_loss: 667.4081 - val_mae: 20.7438
# Epoch 2/4
# 238/238 - 47s - loss: 60.3296 - mae: 5.8738 - val_loss: 392.3885 - val_mae: 14.7134
# Epoch 3/4
# 238/238 - 48s - loss: 31.7256 - mae: 4.2693 - val_loss: 101.3237 - val_mae: 7.5688
# Epoch 4/4
# 238/238 - 48s - loss: 21.0959 - mae: 3.4571 - val_loss: 46.8885 - val_mae: 5.4170
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 48/48 - 8s - loss: 46.8885 - mae: 5.4170
# Test MAE: 5.4170




# WITH subset='training' in first loader func
# Found 6073 validated image filenames.
# Found 1518 validated image filenames.
# 2022-10-10 15:22:20.769058: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
# 2022-10-10 15:22:20.776097: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2099995000 Hz
# 2022-10-10 15:22:20.776843: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x50eb430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2022-10-10 15:22:20.776872: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2022-10-10 15:22:21.001460: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-32GB, Compute Capability 7.0
# 2022-10-10 15:22:21.003833: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
# 2022-10-10 15:22:21.001420: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x474b0d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# pciBusID: 0000:8b:00.0 name: Tesla V100-SXM2-32GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 31.75GiB deviceMemoryBandwidth: 836.37GiB/s
# 2022-10-10 15:22:21.003898: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-10-10 15:22:21.003911: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-10-10 15:22:21.003937: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2022-10-10 15:22:21.003946: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2022-10-10 15:22:21.003955: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2022-10-10 15:22:21.003964: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2022-10-10 15:22:21.003972: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2022-10-10 15:22:21.008239: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2022-10-10 15:22:21.008303: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2022-10-10 15:22:21.419419: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2022-10-10 15:22:21.419472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
# 2022-10-10 15:22:21.419480: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
# 2022-10-10 15:22:21.423833: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2022-10-10 15:22:21.423887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10240 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:8b:00.0, compute capability: 7.0)
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
#   ...
# WARNING:tensorflow:sample_weight modes were coerced from
#     to  
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# Train for 190 steps, validate for 48 steps
# Epoch 1/4
# 2022-10-10 15:22:32.766915: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2022-10-10 15:22:33.149714: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 190/190 - 51s - loss: 216.3325 - mae: 10.5860 - val_loss: 787.3729 - val_mae: 22.9980
# Epoch 2/4
# Epoch 3/4
# 190/190 - 41s - loss: 58.7829 - mae: 5.7724 - val_loss: 712.9997 - val_mae: 21.6180
# 190/190 - 41s - loss: 30.2008 - mae: 4.2540 - val_loss: 292.4623 - val_mae: 12.5172
# Epoch 4/4
# 190/190 - 40s - loss: 20.5208 - mae: 3.4843 - val_loss: 106.2605 - val_mae: 7.8207
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 48/48 - 8s - loss: 106.2605 - mae: 7.8207
# Test MAE: 7.8207