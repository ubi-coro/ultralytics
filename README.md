# Train and Test Event-based Yolo Detection Model

[ultralytics](https://github.com/ultralytics/ultralytics) | [PROPHESEE](https://docs.prophesee.ai/stable/tutorials/other/train_and_test_event_based_yolo_detection_model.html)

Read the [Ultralytics Documentation](https://docs.ultralytics.com/).

## Introduction

The goal of this tutorial is to demonstrate how to quickly leverage the existing popular frame-based neural networks for event-based vision with minimum modifications. Typically, these frame-based networks input RGB images and make predictions in different formats depending on the task. Event frames, such as event histograms, encode similar visual information as RGB images and can be used as their substitutions for the input of the networks.

In this tutorial, we use yolov8 as an example to show how to train and evaluate object detection models with event histograms instead of RGB images. The main differences between the properties of event histograms and RGB images are:

- An event histogram has 2 channels (for ON and OFF polarities) while a RGB image has 3 channels.
- Event histograms are often continuous and stored together in a `.npy` file when they are converted from an events stream. RGB images, by contrast, are separately stored and often irrelevant to each other.

The modification of the source code mainly targets these two issues.

## Prepare the Source Code of Yolov8

Download the source code of [yolov8](https://github.com/ultralytics/ultralytics) by:

```bash
git clone git@github.com:ubi-coro/ultralytics.git
```

## Prepare the Events Dataset for Object Detection

1. Make sure **Metavision SDK** is [properly installed](https://docs.prophesee.ai/stable/installation/index.html#chapter-installation).
2. **Make some recordings** with an event camera (for example with [Metavision Studio](https://docs.prophesee.ai/stable/metavision_studio/index.html#chapter-metavision-studio)). The events are stored in [RAW files](https://docs.prophesee.ai/stable/data/file_formats/raw.html#chapter-data-file-formats-raw).
3. Convert the **RAW event files into HDF5 tensor files** made of event histograms using [Generate HDF5 sample](https://docs.prophesee.ai/stable/samples/modules/ml/generate_hdf5.html#chapter-samples-ml-generate-hdf5). Those files have the extension `.h5`.
4. **Generate labels** for each event histogram.
   - Each label should contain at least the information of timestamp, position of the bounding box of the object and the object class ID.
   - *Timestamp* is used to associate the label with the event histogram. The unit of the timestamp(`ts`) is µs.
   - The *labels* should be saved in a `numpy` array with a customized structured data type: `dtype=[('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1')]`. The location of the bounding box(`x, y, w, h`) is in pixel unit, without normalization. *(For example, a label `[500000, 127., 165., 117., 156., 0]` means it corresponds to the `50th` event histogram if the event histogram is generated with a time interval of `10ms`. There is an object, which corresponds to class `0`, in the scene. The top-left anchor of the bounding box to frame the object is `[165., 127.]` and the height and width of the box is `156.` and `117.`, respectively.)*
   - The labels corresponding to the event histograms in a `xxx.h5` file should be saved in a `xxx_bbox.npy` file, namely changing the suffix from `.h5` to `_bbox.npy`. And the `xxx.h5` file and `xxx_bbox.npy` file should be placed in the same folder.
   - We provide a [labelling tool](https://support.prophesee.ai/portal/en/kb/articles/test-machine-learning-labeling-tool) to facilitate the process. You can get access to it if you are a Prophesee customer by [creating your Knowledge Center account](https://www.prophesee.ai/resources-access-request/).
5. **Group** your event histogram and label files into `train`, `val`, `test` folders.

## Prepare Your Python Environment

We need a python environment that fulfills the [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) listed by yolov8. Once you download the source code, the `requirements.txt` can be found in the root directory. To install *(it is recommended to do it in a virtual environment)*:

```bash
pip install -r requirements.txt
```

Besides, `h5py` package needs to be installed to read the events histograms saved in the `.h5` files:

```bash
pip install h5py
```

## Train a Detection Model

1. Create a `.yaml` file and specify your dataset path and class names:

    ```yaml
    path: YOUR_DATASET_PATH # dataset root dir
    train: train  # train images (relative to 'path') 128 images
    val: val  # val images (relative to 'path') 128 images
    test:  # test images (optional)

    # Classes
    names:
    0: xxx
    1: yyy
    2: zzz
    ```

2. Create a `.py` file and you can train your network with just three lines of code:

    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')
    results = model.train(data='YOUR_YAML_FILE.yaml', amp=False, epochs=10)
    ```

    The network will be trained for only 10 epochs. Increase the number if you want to train more.

## Make Predictions with the Trained Detection Model

The following example shows how to run the detection model over a sequence of event histograms to predict the bounding boxes and the classes. The bounding boxes and classes are shown on the images, which are saved as a video.

```python
from ultralytics import YOLO
import cv2
import h5py
import numpy as np

model = YOLO('YOUR_TRAINED_MODEL.pt')  # load a trained model

events_file = h5py.File("YOUR_TEST_EVENTS_FILE.h5", "r+")
num_ev_histograms, _, height, width = events_file['data'].shape

out = cv2.VideoWriter('OUTPUT_FOLDER/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (width, height))
for idx in range(num_ev_histograms):
    ev_histo = np.transpose(events_file['data'][idx], (1, 2, 0))
    results = model(ev_histo)  # return a generator of Results objects
    annotated_frame = results[0].plot()
    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
out.release()
```