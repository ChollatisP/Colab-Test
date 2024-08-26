from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

from roboflow import Roboflow
rf = Roboflow(api_key="AaOodq2geAVIQQOZVNjW")
project = rf.workspace("gh-sormt").project("model-detect-water-on-floor")
version = project.version(7)
dataset = version.download("yolov8")