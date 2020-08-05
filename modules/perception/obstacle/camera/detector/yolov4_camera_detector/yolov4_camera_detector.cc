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

// #include "/apollo/modules/perception/obstacle/camera/detector/yolov4_camera_detector/yolov4_camera_detector.h"
// #include "/apollo/modules/perception/obstacle/camera/detector/yolov4_camera_detector/cuda.h"
#include "yolov4_camera_detector.h"
#include "cuda.h"


#include <yolo_caffe/yolo_caffe.hpp>
#include <vector>
#include <iostream>
#include "modules/perception/common/perception_gflags.h"
#include "glog/logging.h"

namespace apollo {
namespace perception {

using std::string;
using std::vector;

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

bool YoloV4CameraDetector::Init() {
  // load yolo camera detector config file to proto
  const string prototxt = "/apollo/caffe-yolov4/prototxt/yolov4.prototxt";
  const string caffemodel = "/apollo/caffe-yolov4/yolov4.caffemodel";
  YoloCaffe::SetDevice(FLAGS_obs_camera_detector_gpu);
  YoloCaffe::set_mode(YoloCaffe::GPU);
  YoloCaffe::DeviceQuery();
  AWARN << "Caffe mode: " << YoloCaffe::mode;

  /* load and init network. */
  m_net.reset(new Net<float>(prototxt, TEST));
  m_net->CopyTrainedLayersFrom(caffemodel);
  AINFO << "net inputs numbers is " << m_net->num_inputs();
  AINFO << "net outputs numbers is " << m_net->num_outputs();

  CHECK_EQ(m_net->num_inputs(), 1) << "Network should have exactly one input.";

  m_net_input_data_blobs = m_net->input_blobs()[0];
  AINFO << "input data layer channels is  " << m_net_input_data_blobs->channels();
  AINFO << "input data layer width is  " << m_net_input_data_blobs->width();
  AINFO << "input data layer height is  " << m_net_input_data_blobs->height();
  return true;
}

bool YoloV4CameraDetector::Multitask(
    const cv::Mat &frame, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects) {
  if (objects == nullptr) {
    AERROR << "'objects' is a null pointer.";
    return false;
  }
  Detect(frame, thresh, objects);
  
  return true;
}

bool YoloV4CameraDetector::Detect(const cv::Mat &mat, float thresh, std::vector<std::shared_ptr<VisualObject>> *objects){
    //convert mat to image
    if(mat.data == NULL)
        throw std::runtime_error("Mat is empty");
    YoloCaffe::SetDevice(FLAGS_obs_camera_detector_gpu);
    YoloCaffe::set_mode(YoloCaffe::GPU);
    image im = mat_to_image(mat);
    image sized = letterbox_image(im,m_net_input_data_blobs->width(),m_net_input_data_blobs->height());

    //copy data from cpu to gpu
    int size = m_net_input_data_blobs->channels()*m_net_input_data_blobs->width()*m_net_input_data_blobs->height();
    cuda_push_array(m_net_input_data_blobs->mutable_gpu_data(),sized.data,size);

    //clean blobs
    m_blobs.clear();
        
    int nboxes = 0;
    detection *dets = NULL;
    yolo_caffe::Timer det_time;
    det_time.Start();
    // forward
    m_net->Forward();
    det_time.Stop();
    AWARN << "Running detection: " << det_time.MilliSeconds() << " ms" << YoloCaffe::mode();

    for(int i =0;i<m_net->num_outputs();++i){
        m_blobs.push_back(m_net->output_blobs()[i]);
    }

    dets = get_detections(m_blobs,im.w,im.h,
        m_net_input_data_blobs->width(),m_net_input_data_blobs->height(),m_thresh, m_classes, &nboxes);
    AWARN << "Got some detection ressults: " << nboxes;
    //deal with results
    for (int i = 0; i < nboxes; ++i) {
        box b = dets[i].bbox;
        int obj_id = max_index(dets[i].prob, m_classes);
        float score = std::max(dets[i].objectness, dets[i].prob[obj_id]);

        float const prob = dets[i].prob[obj_id];
        int det_id = 0;
        if (prob > thresh)
        {
            std::shared_ptr<VisualObject> bbox(new VisualObject);
            bbox->id = det_id;
            bbox->score = score;
            float x = std::max((double)0, (b.x - b.w / 2.)*im.w);
            float y = std::max((double)0, (b.y - b.h / 2.)*im.h);
            float w = b.w*im.w;
            float h = b.h*im.h;
            bbox->upper_left = Eigen::Vector2f(x,y);
            bbox->lower_right = Eigen::Vector2f(x+w,y+h);
            bbox->height = 1.5;
            bbox->width = 1.6000000238418579;
            bbox->length = 3.877000093460083;
            bbox->velocity = Eigen::Vector3f(-10.0f, 0.0f, 0.0f);
            // bbox->direction = Eigen::Vector3f(1.57f, 0.0f, 0.0f);
            AWARN << "FOUND " << obj_id;
            switch (obj_id)
            {
            case 0 :
                bbox->type = ObjectType::PEDESTRIAN;
                break;
            case 7 :
                bbox->type = ObjectType::VEHICLE;
                break;
            case 2 :
                bbox->type = ObjectType::VEHICLE;
            default:
                bbox->type = ObjectType::UNKNOWN;
                break;
            }
            // bbox->type = static_cast<ObjectType>(obj_id);
            std::vector<float> temp_probs(dets[i].prob, dets[i].prob+80);
            bbox->type_probs = temp_probs;
            bbox->object_feature.clear();
            objects->push_back(bbox);
            det_id++;
        }
    }
    free_detections(dets,nboxes);
    free_image(sized);
    free_image(im);
    AWARN << "Free images detections";
    return true;
}

string YoloV4CameraDetector::Name() const { return "YoloV4CameraDetector"; }

}  // namespace perception
}  // namespace apollo
