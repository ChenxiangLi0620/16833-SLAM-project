//
// Created by ubuntu on 3/16/23.
//
#ifndef YOLOV8_HPP
#define YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
using namespace det;

class YOLOv8 {
public:
    YOLOv8();
    ~YOLOv8();

    //add by jhw
    void GetImage(cv::Mat & RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();
    std::vector<cv::Rect2i> mvPersonArea = {};
    //--

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs);
    static void          draw_objects(
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    std::vector<int> hardNMS(std::vector<Object> &input, std::vector<Object> &output, float iou_threshold, unsigned int topk);
    bool cmp(Object &obj1, Object &obj2);
    static float iou_of(const Object &obj1, const Object &obj2);

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

public:
    //add by jhw
    cv::Mat mRGB;
    std::vector<cv::Rect2i> mvDynamicArea;
    std::vector<std::string> mvDynamicNames;
    std::vector<std::string> mClassnames;

    std::map<std::string, std::vector<cv::Rect2i>> mmDetectMap;
    std::vector<std::vector<unsigned int>> COLORS;
    //--
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};


#endif  // JETSON_DETECT_YOLOV8_HPP
