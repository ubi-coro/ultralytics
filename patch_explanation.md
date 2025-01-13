# The explanations of the modifications

In `ultralytics/data/base.py`:

1. Add `import h5py` at the beginning of the file.

2. Modify the `get_img_files` function of the BaseDataset:

    First, change

    ```python
    im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    ```

    to

    ```python
    im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() == "h5")
    ```

    because our event histograms are saved in .h5 files and do not have the endings like `.jpg`, `.png`, etc.

    Then, comment these two lines as their usage is not longer valid:

    ```python
    # if self.fraction < 1:
    #     im_files = im_files[:round(len(im_files) * self.fraction)]
    ```

3. Modify the `_init_` function of the `BaseDataset`:

    First, change

    ```python
    self.labels = self.get_labels()
    ```

    to

    ```python
    self.labels = self.get_events_labels()
    ```

    as later we will add this `get_events_labels` function in the `YOLODataset` class to load the labels.

    Then, comment this line as we do not store individual images in `.npy` files:

    ```python
    # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
    ```

4. Modify the `load_image` function of the `BaseDataset`:

    First, comment this piece of code as we will not load any RGB images:

    ```python
    #"""Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    #im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    #if im is None:  # not cached in RAM
    #    if fn.exists():  # load npy
    #        im = np.load(fn)
    #    else:  # read image
    #        im = cv2.imread(f)  # BGR
    #        if im is None:
    #            raise FileNotFoundError(f'Image Not Found {f}')
    ```

    Then, add this piece of code below to read event histograms:

    ```python
    file_idx, frame_idx = self.labels[i]["im_file"]
    ev_frame_identifier = self.im_files[file_idx] + "_frame_" + str(frame_idx)
    with h5py.File(self.im_files[file_idx], "r") as h5_file:
        im = h5_file["data"][frame_idx].transpose(1, 2, 0)
    ```

    - We need to put the channel dimension of the event histogram to the last dimension so that it can be processed by `OpenCV` functions such as `cv2.resize`.
    - `ev_frame_identifier` is a string recording the location of the event histogram.

    Next, change the line

    ```python
    return im, (h0, w0), im.shape[:2]
    ```

    to

    ```python
    return im, (h0, w0), im.shape[:2], ev_frame_identifier
    ```

    and comment the line

    ```python
    # return self.ims[i], self.im_hw0[i], self.im_hw[i]
    ```

5. Modify the `set_rectangle` function of the `BaseDataset`:

    Comment this line as it is no longer valid:

    ```python
    # self.im_files = [self.im_files[i] for i in irect]
    ```

6. Modify the `get_image_and_label` function of the `BaseDataset`:

    Change this line

    ```python
    label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
    ```

    to

    ```python
    label['img'], label['ori_shape'], label['resized_shape'], label['im_file'] = self.load_image(index)
    ```

    because we also return ev_frame_identifier in the load_image function.

In `ultralytics/data/dataset.py`:

1. Add `import h5py` at the beginning of the file.

2. Add a function called `get_events_labels` to the class `YOLODataset`:

    ```python
    def get_events_labels(self):
      self.label_files = [x.rsplit('.', 1)[0] + '_bbox.npy' for x in self.im_files]
      all_labels = []
      for file_idx in range(len(self.im_files)):
          with h5py.File(self.im_files[file_idx], "r") as h5_file:
              num_frames, _, height, width = h5_file["data"].shape
          labels = np.load(self.label_files[file_idx])
          x_normalized, y_normalized, w_normalized, h_normalized = labels['x'] / width, labels['y'] / height, labels['w'] / width, labels['h'] / height
          x_centered_normalized = x_normalized + w_normalized / 2
          y_centered_normalized = y_normalized + h_normalized / 2
          bboxes_normalized = np.stack((x_centered_normalized, y_centered_normalized, w_normalized, h_normalized), axis=1)
          classes = labels["class_id"]
          timestamps = labels['ts']
          for frame_idx in range(num_frames):
              box_idxs = np.nonzero(timestamps == (frame_idx+1) * 1e5)[0]
              if len(box_idxs) > 0:
                  all_labels.append(
                      dict(
                          im_file=[file_idx, frame_idx],
                          shape=(height, width),
                          cls=classes[box_idxs, np.newaxis],  # n, 1
                          bboxes=bboxes_normalized[box_idxs, :],  # n, 4
                          segments=[],
                          keypoints=None,
                          normalized=True,
                          bbox_format='xywh'))
      return all_labels
    ```

    - The previous section mentioned that the labels corresponding to the event histograms in a `xxx.h5` file should be saved in a `xxx_bbox.npy` file in the same folder. So the `self.label_files` here stored the paths to all the label files.
    - It also mentioned that the `x, y` fields of the bounding boxes refer to the top-left corner. But in yolov8, they refer to the center of the bounding box. So we need to set `x=x+w/2, y=y+h/2`. Also the fields `x,y,w,h` need to be normalized with respect to the `height` and `width` of the event histograms in align with the convention of yolov8.
    - A label is matched with an event histogram by its timestamp, i.e. its `ts` field. For example, if the timestamp of a label is `500000`, it corresponds to the `50th` event histogram, assuming the event histograms are generated with a time interval of `10ms`. If you use a different time interval to generate events, please change the `1e5` in the code to your time interval.
    - Each label is constructed as a dictionary and it contains the information of the location, shape of the event histogram, the class ID and the location of the bounding boxes in the event histogram.

In `ultralytics/data/augment.py`:

Comment these augmentations in the `v8_transforms` function:

```python
# Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
# CopyPaste(p=hyp.copy_paste),

# MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
# Albumentations(p=1.0),
# RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
```

since they are no longer valid with respect to event histograms.

In `ultralytics/utils/plotting.py`:

1. Add these lines right after the imports of libraries:

    ```python
    BG_COLOR = np.array([30, 37, 52], dtype=np.uint8)
    POS_COLOR = np.array([216, 223, 236], dtype=np.uint8)
    NEG_COLOR = np.array([64, 126, 201], dtype=np.uint8)
    ```

    and add a function called `viz_histo_binarized` somewhere in the file:

    ```python
    def viz_histo_binarized(im):
      """
      Visualize binarized histogram of events

      Args:
          im (np.ndarray): Array of shape (2,H,W)

      Returns:
          output_array (np.ndarray): Array of shape (H,W,3)
      """
      img = np.full(im.shape[-2:] + (3,), BG_COLOR, dtype=np.uint8)
      y, x = np.where(im[0] > 0)
      img[y, x, :] = POS_COLOR
      y, x = np.where(im[1] > 0)
      img[y, x, :] = NEG_COLOR
      return img
    ```

    The function can be utilized to visualize the event histograms as RGB images.

2. Modify the `plot_images` function by changing this line:

    ```python
    im = im.transpose(1, 2, 0)
    ```

    to

    ```python
    im = viz_histo_binarized(im.copy())
    ```

    to convert an event histogram to a 3-channel image which can be properly visualized.

In `ultralytics/engine/results.py`:

1. Add import of `viz_histo_binarized` from `ultralytics.utils.plotting`:

    ```python
    from ultralytics.utils.plotting import Annotator, colors, save_one_box, viz_histo_binarized
    ```

2. Change the following line in the function `plot`:

    ```python
    deepcopy(self.orig_img if img is None else img),
    ```

    to

    ```python
    deepcopy(viz_histo_binarized(self.orig_img.transpose(2, 0, 1)) if img is None else viz_histo_binarized(img.transpose(2, 0, 1))),
    ```

    The reason is the same as above: converting an event histogram to a 3-channel image which can be properly visualized.

In `ultralytics/cfg/models/v8/yolov8.yaml`:

Add the line:

```yaml
ch: 2
```

to specify the input is expected to have two channels because the event histogram has two channels.

In `ultralytics/engine/predictor.py`:

Change the line:

```python
self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
```

to

```python
self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 2, *self.imgsz))
```

because the network now expects the input has 2 channels not 3.

In `ultralytics/engine/validator.py`:

Change the line:

```python
model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
```

to

```python
model.warmup(imgsz=(1 if pt else self.args.batch, 2, imgsz, imgsz))  # warmup
```