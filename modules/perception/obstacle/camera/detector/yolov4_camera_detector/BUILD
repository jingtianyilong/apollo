load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "yolov4_camera_detector",
    srcs = ["yolov4_camera_detector.cc"],
    hdrs = ["yolov4_camera_detector.h",
            "yolo_layer.h",
            "image_opencv.h",
            "image.h",
            "detector.h",
            "cuda.h",
            "box.h",
            "blas.h",
            "activations.h"],
    deps = [
        "//modules/perception/obstacle/camera/interface",
        "//modules/perception/lib/base",
        "@caffe//:lib",
        "@eigen",
    ],
)

cpplint()
