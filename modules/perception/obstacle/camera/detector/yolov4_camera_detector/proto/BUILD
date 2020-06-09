syntax = "proto2"

load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "yolov4_camera_proto",
    deps = [
        ":yolov4_camera_proto_lib",
    ],
)

proto_library(
    name = "yolov4_camera_proto_lib",
    srcs = [
        "yolov4.proto",
    ],
)

cpplint()
