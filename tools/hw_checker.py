import tensorflow as tf
import GPUtil

def check():
  print("Tensorflow version ")
  print(tf.version)
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tf.distribute.experimental.TPUStrategy(tpu)
  except ValueError:
    print('ERROR: Not connected to a TPU runtime!')
    GPUs = GPUtil.getGPUs()
    print('GPU count: ' + str(len(GPUs)))