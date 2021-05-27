# Hand Tracker
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wolterlw/hand_tracking/master)

Simple Python wrapper for Google's [Mediapipe Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md) pipeline.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

These are required to use the HandTracker module

```
numpy
opencv
tensorflow
```
To run the Jupyter demo you'll also need `jupyter` and `matplotlib`

### Setup

Download models using load_models.sh, or load them manually from [metalwhale's repo](https://github.com/metalwhale/hand_tracking/) and put them inside models dir.
### Anchors

To get the SSD anchors I've written a C++ program that executes the `SsdAnchorsCalculator::GenerateAnchors` function from [this calculator](https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc).
As there's no reason to modify provided anchors I do not include it into the repository, but you can find the script [here](https://gist.github.com/wolterlw/6f1ebc49230506f8e9ce5facc5251d4f)

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.   
Big thanks to @metalwhale for removing cusom operation dependencies.


## Compile
    1. fatal error: pyconfig.h: No such file or directory
        - 'find /usr/include -name pyconfig.h'
        - 'export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.6m/"'
    2. compile with g++
        - 'g++ -fPIC -shared -o gesture_recognition.so gesture_recognition.cpp -lpython3.6m -lboost_python3 -lboost_numpy3'
        - reference link: https://blog.csdn.net/fxjtoday/article/details/6059874?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic
    3. Issue to be solve:
        - (i)   counter on detection output
        - (ii)  flip right hand to left?                (done)
        - (iii) Integrate with ROS
        - (iv)  Write into class 
        - (v)   Multi hand detection function
