load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "yolov4_camera_detector",
    data = [
        "//caffe-yolov4:model_data",
    ],
    srcs = [
        "build/x86_64/lib/libyolov4-plugin.so",
        "yolov4_camera_detector.cc",
        ],
    hdrs = [
        "yolov4_camera_detector.h",
        "yolo_layer.h",
        "image.h",
        "image_opencv.h",
        "cuda.h",
        "box.h",
        "blas.h",
        "activations.h",
        ],
    deps = [
        "//modules/common",
        "//modules/perception/common",
        "//modules/perception/obstacle/camera/common:util",
        "//modules/perception/obstacle/camera/interface:interface",
        "//modules/perception/lib/base:base",
        "//modules/perception/obstacle/base",
        "@yolo_caffe//:lib",
        "@eigen",
    ],
)

cc_test(
    name = "yolov4_camera_detector_test",
    size = "small",
    srcs = [
        "yolov4_camera_detector_test.cc",
    ],
    data = [
        "//caffe-yolov4:model_data",
        "//modules/perception:perception_data",
    ],
    deps = [
        ":yolov4_camera_detector",
        "@gtest//:main",
        "@yolo_caffe//:lib",
        "@eigen",
    ],
)

cpplint()
