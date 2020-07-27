/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef MODULES_PERCEPTION_OBSTACLE_CAMERA_DETECTOR_YOLOV4_CAMERA_DETECTOR_H_
#define MODULES_PERCEPTION_OBSTACLE_CAMERA_DETECTOR_YOLOV4_CAMERA_DETECTOR_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <caffe/caffe.hpp>
#include "modules/perception/obstacle/camera/detector/yolov4_camera_detector/image_opencv.h"
#include "modules/perception/obstacle/camera/interface/base_camera_detector.h"
#include "modules/perception/obstacle/camera/detector/yolov4_camera_detector/yolo_layer.h"

namespace apollo {
namespace perception {

// struct bbox_t{
//     unsigned int x,y,w,h;   //(x,y) - top-left corner, (w,h) - width & height of bounded box
//     float prob;             // confidence - probability that the object was found correctly
//     unsigned int obj_id;    // class of object - from range [0,classes - 1]
// };

class YoloV4CameraDetector : public BaseCameraDetector {
 public:
  YoloV4CameraDetector() : BaseCameraDetector() {}

  virtual ~YoloV4CameraDetector() {}

  bool Init();
  bool Init(const CameraDetectorInitOptions &options =
                CameraDetectorInitOptions()) override {return false;};

  bool Detect(const cv::Mat &frame, const CameraDetectorOptions &options,
              std::vector<std::shared_ptr<VisualObject>> *objects) override {return false;};

  bool Detect(const cv::Mat &mat, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects);

  bool Multitask(const cv::Mat &frame, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects);

  bool Lanetask(const cv::Mat &frame, cv::Mat *mask) override {return false;};

  bool Extract(std::vector<std::shared_ptr<VisualObject>> *objects) {return false;}

  std::string Name() const override;

 protected:
  shared_ptr<Net<float> > m_net;
  Blob<float> * m_net_input_data_blobs;
  vector<Blob<float>*> m_blobs;

  float m_thresh = 0.001;
  int m_classes = 80; //coco classes
};

REGISTER_CAMERA_DETECTOR(YoloV4CameraDetector);

}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_CAMERA_DETECTOR_YOLOV4_CAMERA_DETECTOR_H_
