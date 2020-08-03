/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */

#ifndef __BOX_YOLOV4_H_
#define __BOX_YOLOV4_H_
#include "yolo_layer.h"


void do_nms_sort(detection *dets, int total, int classes, float thresh);


#endif
