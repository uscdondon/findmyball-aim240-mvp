## YOLO Dataset Structure

This folder prepares the AIM240 FindMyBall project for future YOLO object detection training.

YOLO training is not implemented yet. Images and labels should be created manually or exported from a labeling tool such as CVAT, Roboflow, Label Studio, or LabelImg.

Expected layout:

```text
data/yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  dataset.yaml
```

Each image should have a matching `.txt` label file with the same base filename. Labels use YOLO format:

```text
class_id x_center y_center width height
```

All coordinate values are normalized from `0` to `1`. Class `0` means `golf_ball`.
