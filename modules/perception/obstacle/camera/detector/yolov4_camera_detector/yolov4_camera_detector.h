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

#include <yolo_caffe/yolo_caffe.hpp>
#include "modules/perception/obstacle/camera/common/visual_object.h"
#include "modules/perception/obstacle/camera/interface/base_camera_detector.h"
#include "image_opencv.h"
#include "yolo_layer.h"
namespace apollo {
namespace perception {

class YoloV4CameraDetector : public BaseCameraDetector {
 public:
  YoloV4CameraDetector() : BaseCameraDetector() {}

  virtual ~YoloV4CameraDetector() {}

  bool Init();
  bool Init(const CameraDetectorInitOptions &options) override {return false;};
  // bool Init(const CameraDetectorInitOptions &options =
  //               CameraDetectorInitOptions()) override {return false;};

  bool Detect(const cv::Mat &frame, const CameraDetectorOptions &options,
              std::vector<std::shared_ptr<VisualObject>> *objects) override {return false;};

  bool Detect(const cv::Mat &mat, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects);

  bool Multitask(const cv::Mat &frame, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects);
  bool Multitask(const cv::Mat& frame,
                         const CameraDetectorOptions& options,
                         std::vector<std::shared_ptr<VisualObject>>* objects,
                         cv::Mat* mask) {
    *mask = cv::Mat(384, 960, CV_32FC1);
    return true;
  }
  bool Lanetask(const cv::Mat &frame, cv::Mat *mask) override {
    *mask = cv::Mat(384, 960, CV_32FC1);
    return true;
    }

  bool Extract(std::vector<std::shared_ptr<VisualObject>> *objects) {return false;}

  std::string Name() const override;

 protected:
  shared_ptr<yolo_caffe::Net<float> > m_net;
  yolo_caffe::Blob<float> * m_net_input_data_blobs;
  std::vector<Blob<float>*> m_blobs;

  float m_thresh = 0.001;
  int m_classes = 80; //coco classes
};

REGISTER_CAMERA_DETECTOR(YoloV4CameraDetector);

}  // namespace perception
}  // namespace apollo

#endif  // MODULES_PERCEPTION_OBSTACLE_CAMERA_DETECTOR_YOLOV4_CAMERA_DETECTOR_H_
