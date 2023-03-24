import tensorflow as tf
# check if tensorflow can see the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# check if tensorflow can use the GPU
tf.debugging.set_log_device_placement(True)