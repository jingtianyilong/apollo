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

#include "modules/perception/obstacle/camera/detector/yolov4_camera_detector/yolov4_camera_detector.h"

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

#include "modules/common/log.h"
#include "modules/perception/common/perception_gflags.h"
#include "modules/perception/obstacle/camera/interface/base_camera_detector.h"


DEFINE_string(test_dir,
              "/apollo/modules/perception/data/yolo_camera_detector_test/",
              "test data directory");

namespace apollo {
namespace perception {


class YoloV4CameraDetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RegisterFactoryYoloV4CameraDetector();
  }
};

TEST_F(YoloV4CameraDetectorTest, model_init_test) {

  YoloV4CameraDetector *camera_detector =
      new YoloV4CameraDetector;
  CHECK_NOTNULL(camera_detector);
  CHECK(camera_detector->Init());

}

TEST_F(YoloV4CameraDetectorTest, multi_task_test) {
  YoloV4CameraDetector *camera_detector = new YoloV4CameraDetector;
  CHECK(camera_detector->Init());
  CHECK_EQ(camera_detector->Name(), "YoloV4CameraDetector");

  const std::string image_file = FLAGS_test_dir + "test.jpg";
  ADEBUG << "test image file: " << image_file;

  cv::Mat frame = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
  CHECK_NOTNULL(frame.data);

  CHECK_EQ(camera_detector->Multitask(frame, 0.45f, NULL), false);

  std::vector<std::shared_ptr<VisualObject>> objects;
  CHECK(camera_detector->Multitask(frame, 0.45f, &objects));
  ADEBUG << "#objects detected = " << objects.size();

  CHECK_EQ(objects.size(), 1);  // Related to current model and threshold
}

}  // namespace perception
}  // namespace apollo

