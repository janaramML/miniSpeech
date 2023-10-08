"""
Deploy ptr-trainied mini-speech-command tiny tensor

"""
import json
import tarfile
import pathlib
import tempfile
import numpy as np
import tensorflow as tf
import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.contrib.download import download_testdata
from tvm.relay.backend import Executor, Runtime


# set environment variable
MODEL_PATH = "./quant_tiny_model.tflite"
input_shape = (1,98, 40, 1)
data_type ="int8"
INPUT_NAME = "input_0"
use_physical_hw = False
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.micro.testing.get_target("crt")
BOARD=""
SERIAL=""
# Use the AOT executor rather than graph or vm executors. Don't use unpacked API or C calling style.
EXECUTOR = Executor("aot")
batch_size = 1
dataset_url= "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
tvm_generated_files_dir = "./TVM_GENERATED_FILES/"
SAMPLE_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
SAMPLE_PATH = download_testdata(SAMPLE_URL, "keyword_spotting_int8_6.pyc.npy", module="data")
labels = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]


def download_data(data_dir):
  """
  Function to load the train and test dataset
  """
  if not data_dir.exists():
    tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir=data_dir)
    

######################################################################
# Load a TFLite model
# -------------------

import os
tflite_model_file = os.path.join(MODEL_PATH)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)   


######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {INPUT_NAME: data_type}
shape_dict = {INPUT_NAME: input_shape}

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

print("Relay IR:\n", mod)

######################################################################
# Compile the Relay module
# ------------------------


with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    aot_module = relay.build(mod, target=TARGET, runtime=RUNTIME, executor=EXECUTOR,params=params)


def build_project(mod):

    from sys import get_coroutine_origin_tracking_depth
    # built thin app layer on top of function for running
    # Get a temporary path where we can store the tarball (since this is running as a tutorial).

    temp_dir = tvm.contrib.utils.tempdir()
    model_tar_path = temp_dir / "model.tar"
    export_model_library_format(mod, model_tar_path)

    with tarfile.open(model_tar_path, "r:*") as tar_f:
       print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

    # TVM also provides a standard way for embedded platforms to automatically generate a standalone
    # project, compile and flash it to a target, and communicate with it using the standard TVM RPC
    # protocol. The Model Library Format serves as the model input to this process. When embedded
    # platforms provide such an integration, they can be used directly by TVM for both host-driven
    # inference and autotuning . This integration is provided by the
    # `microTVM Project API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_,
    #
    # Embedded platforms need to provide a Template Project containing a microTVM API Server (typically,
    # this lives in a file ``microtvm_api_server.py`` in the root directory). Let's use the example ``host``
    # project in this tutorial, which simulates the device using a POSIX subprocess and pipes:

    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
    project_options = {}  # You can use options to provide platform-specific options through TVM.

    #  For physical hardware, you can try out the Zephyr platform by using a different template project
    #  and options:


    if use_physical_hw:
        template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
        project_options = {
            "project_type": "host_driven",
            "board": BOARD,
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
            "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
        }

    # Create a temporary directory
    temp_dir = tvm.contrib.utils.tempdir(tvm_generated_files_dir)
    aot_generated_project_dir = temp_dir / "aot_generated_project"
    print(aot_generated_project_dir)
    aot_generated_project = tvm.micro.generate_project(
        template_project_path, aot_module, aot_generated_project_dir, project_options
    )

    # Build and flash the project
    aot_generated_project.build()
    aot_generated_project.flash()

    return aot_generated_project


######################################################################
# Create TVM runtime and do inference
# -----------------------------------


def get_tvm_accuracy(project, sample, ground_truth):
    print("\n")
    # create module
    with tvm.micro.Session(transport_context_manager=project.transport()) as session:
       aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
       aot_executor.get_input(INPUT_NAME).copyfrom(sample)
       # run
       import time
       timeStart = time.time()
       aot_executor.run()
       timeEnd = time.time()
       print("Inference time: %f" % (timeEnd - timeStart))
       # get output
       result = aot_executor.get_output(0).numpy()
       print(f"Label is `{labels[np.argmax(result)]}` with index `{np.argmax(result)}`")
       # print top-1
       #top1 = np.argmax(tvm_output[0])
       #print("Top-1: %s (ID: %d)" % (block.classes[top1], top1))
       # print top-5
       top5 = result[0].argsort()[-5:][::-1]
       print("TVM's prediction:", top5)
       check_top1 = 0
       check_top5 = 0
       for top_id in range(5):
          if top_id == 0 and top5[top_id] == ground_truth:
             check_top1 = 1
          if top5[top_id] == ground_truth:
             check_top5 = 1
          print("Top-%d: %s (ID: %d) " % (top_id+1, labels[top5[top_id]], top5[top_id]))
       return check_top1, check_top5


aot_project = build_project(aot_module)
print("\nStart to estimate the accuracy using TVM")
