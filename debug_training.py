# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

def main(pipeline_config_path_n = None, 
         num_train_steps_n = None, 
         eval_on_train_data_n = False,
         sample_1_of_n_eval_examples_n = None,
         sample_1_of_n_eval_on_train_examples_n = 5, 
         model_dir_n=None,
         checkpoint_dir_n = None,
         eval_timeout_n = 3600,
         use_tpu_n = False,
         tpu_name_n = None,
         num_workers_n = 1,
         checkpoint_every_n_n = 1000,
         record_summaries_n = True):

  if pipeline_config_path_n is None or model_dir_n is None or num_train_steps_n is None:
    print("Model dir and pipeline config path must be provided.")
  else:
    if checkpoint_dir_n:
      model_lib_v2.eval_continuously(
          pipeline_config_path=pipeline_config_path_n,
          model_dir=model_dir_n,
          train_steps=num_train_steps_n,
          sample_1_of_n_eval_examples=sample_1_of_n_eval_examples_n,
          sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples_n),
          checkpoint_dir=checkpoint_dir_n,
          wait_interval=300, timeout=eval_timeout_n)
    else:
      if use_tpu_n:
        # TPU is automatically inferred if tpu_name is None and
        # we are running under cloud ai-platform.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name_n)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
      elif num_workers_n > 1:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
      else:
        strategy = tf.compat.v2.distribute.MirroredStrategy()

      with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config_path_n,
            model_dir=model_dir_n,
            train_steps=num_train_steps_n,
            use_tpu=use_tpu_n,
            checkpoint_every_n=checkpoint_every_n_n,
            record_summaries=record_summaries_n)


model_name = "my_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8_3"
pipeline_config_path = f"TensorFlow/workspace/training_demo/models/{model_name}/pipeline.config" #'Path to pipeline config '
num_train_steps = 25000 # 'Number of train steps.'
eval_on_train_data = False # 'Enable evaluating on train data (only supported in distributed training).' (CHYBA NIE UŻYWANE)
sample_1_of_n_eval_examples = None # 'Will sample one of every n eval input examples, where n is provided.'
sample_1_of_n_eval_on_train_examples = 5 # 'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.'
model_dir = f"TensorFlow/workspace/training_demo/models/{model_name}" # 'Path to output model directory where event and checkpoint files will be written.
checkpoint_dir = None # 'Path to directory holding a checkpoint. If `checkpoint_dir` is provided, this binary operates in eval-only mode, writing resulting metrics to `model_dir`.'
eval_timeout = 3600 # 'Number of seconds to wait for an evaluation checkpoint before exiting.'
use_tpu = False # 'Whether the job is executing on a TPU.'
tpu_name = None # 'Name of the Cloud TPU for Cluster Resolvers.'
num_workers = 1 # 'When num_workers > 1, training uses MultiWorkerMirroredStrategy. When num_workers = 1 it uses MirroredStrategy.'
checkpoint_every_n = 1000 # 'Integer defining how often we checkpoint.'
record_summaries = True # 'Whether or not to record summaries defined by the model or the training pipeline. This does not impact the summaries of the loss values which are always recorded.'

main(pipeline_config_path_n = pipeline_config_path, 
     num_train_steps_n = num_train_steps, 
     eval_on_train_data_n = eval_on_train_data,
     sample_1_of_n_eval_examples_n = sample_1_of_n_eval_examples,
     sample_1_of_n_eval_on_train_examples_n = sample_1_of_n_eval_on_train_examples, 
     model_dir_n = model_dir,
     checkpoint_dir_n = checkpoint_dir,
     eval_timeout_n = eval_timeout,
     use_tpu_n = use_tpu,
     tpu_name_n = tpu_name,
     num_workers_n = num_workers,
     checkpoint_every_n_n = checkpoint_every_n,
     record_summaries_n = record_summaries)