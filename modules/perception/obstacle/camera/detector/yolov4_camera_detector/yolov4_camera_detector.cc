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
#include "modules/perception/obstacle/camera/detector/yolov4_camera_detector/cuda.h"



#include <caffe/caffe.hpp>
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

  Caffe::set_mode(Caffe::GPU);

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
    const cv::Mat &frame, float thresh, std::vector<std::shared_ptr<bbox_t>> *objects) {
  if (objects == nullptr) {
    AERROR << "'objects' is a null pointer.";
    return false;
  }
  Detect(frame, thresh, objects);
  return true;
}

bool YoloV4CameraDetector::Detect(const cv::Mat &mat, float thresh, std::vector<std::shared_ptr<bbox_t>> *objects){
    //convert mat to image
    if(mat.data == NULL)
        throw std::runtime_error("Mat is empty");
    image im = mat_to_image(mat);
    image sized = letterbox_image(im,m_net_input_data_blobs->width(),m_net_input_data_blobs->height());

    //copy data from cpu to gpu
    int size = m_net_input_data_blobs->channels()*m_net_input_data_blobs->width()*m_net_input_data_blobs->height();
    cuda_push_array(m_net_input_data_blobs->mutable_gpu_data(),sized.data,size);

    //clean blobs
    m_blobs.clear();
        
    int nboxes = 0;
    detection *dets = NULL;

    // forward
    m_net->Forward();
    for(int i =0;i<m_net->num_outputs();++i){
        m_blobs.push_back(m_net->output_blobs()[i]);
    }

    dets = get_detections(m_blobs,im.w,im.h,
        m_net_input_data_blobs->width(),m_net_input_data_blobs->height(),m_thresh, m_classes, &nboxes);

    //deal with results
    for (int i = 0; i < nboxes; ++i) {
        box b = dets[i].bbox;
        int const obj_id = max_index(dets[i].prob, m_classes);
        float const prob = dets[i].prob[obj_id];

        if (prob > thresh)
        {
            std::shared_ptr<bbox_t> bbox(new bbox_t);
            bbox->x = std::max((double)0, (b.x - b.w / 2.)*im.w);
            bbox->y = std::max((double)0, (b.y - b.h / 2.)*im.h);
            bbox->w = b.w*im.w;
            bbox->h = b.h*im.h;
            bbox->obj_id = obj_id;
            bbox->prob = prob;

            objects->push_back(bbox);
        }
    }
    free_detections(dets,nboxes);
    free_image(sized);
    free_image(im);
    return true;
}

string YoloV4CameraDetector::Name() const { return "YoloV4CameraDetector"; }

}  // namespace perception
}  // namespace apollo
