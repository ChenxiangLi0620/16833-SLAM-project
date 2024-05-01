#include"yolov8.hpp"

YOLOv8::YOLOv8()
{
    std::ifstream file("/home/jhw/YOLOv8_ORBSLAM3/YOLOv8_ORB_SLAM3/yolov8n.engine", std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    mmDetectMap = std::map<std::string, std::vector<cv::Rect2i>>();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);


        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
    this->make_pipe(true);

    this->mClassnames = {
            "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
            "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
            "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
            "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
            "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
            "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
            "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
            "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
            "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
            "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
            "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
            "teddy bear",     "hair drier", "toothbrush"};
    this->mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};

    COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

}

YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YOLOv8::ClearImage()
{
    mRGB = 0;
}

void YOLOv8::ClearArea()
{
    mvPersonArea.clear();
}
bool YOLOv8::cmp(Object &obj1, Object &obj2){
    return obj1.prob > obj2.prob;
};
float YOLOv8::iou_of(const Object &obj1, const Object &obj2)
{
    float x1_lu = obj1.rect.x;
    float y1_lu = obj1.rect.y;
    float x1_rb = x1_lu + obj1.rect.width;
    float y1_rb = y1_lu + obj1.rect.height;
    float x2_lu = obj2.rect.x;
    float y2_lu = obj2.rect.y;
    float x2_rb = x2_lu + obj2.rect.width;
    float y2_rb = y2_lu + obj2.rect.height;
    //交集左上角坐标i_x1, i_y1
    float i_x1 = std::max(x1_lu, x2_lu);
    float i_y1 = std::max(y1_lu, y2_lu);
    //交集右下角坐标i_x2, i_y2
    float i_x2 = std::min(x1_rb, x2_rb);
    float i_y2 = std::min(y1_rb, y2_rb);
    //交集框宽高
    float i_w = i_x2 - i_x1;
    float i_h = i_y2 - i_y1;
    //并集左上角坐标
    float o_x1 = std::min(x1_lu, x2_lu);
    float o_y1 = std::min(y1_lu, y2_lu);
    //并集右下角坐标
    float o_x2 = std::max(x1_rb, x2_rb);
    float o_y2 = std::max(y1_rb, y2_rb);
    //并集宽高
    float o_w = o_x2 - o_x1;
    float o_h = o_y2 - o_y1;

    return (i_w*i_h) / (o_w*o_h);
};
std::vector<int> YOLOv8::hardNMS(std::vector<Object> &input, std::vector<Object> &output, float iou_threshold, unsigned int topk)
{  //Object只有confidence和label
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    std::vector<int> indices;

    if (input.empty())
        return indices;
    std::vector<Object> res;
    //先对bboxs按照conf进行排序
    std::sort(input.begin(), input.end(),
              [](const Object &a, const Object &b)
              { return a.prob > b.prob; });   //[]表示C++中的lambda函数

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {   //按照conf依次遍历bbox
        if (merged[i])
            continue;
        //如果已经被剔除，continue
        Object buf;
        buf = input[i];
        merged[i] = 1; //剔除当前bbox

        //由于后面的置信度低，只需要考虑当前bbox后面的即可
        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j])
                continue;

            float iou = static_cast<float>(iou_of(input[j], input[i]));
            //计算iou
            if (iou > iou_threshold)
            { //超过阈值认为重合，剔除第j个bbox，
                merged[j] = 1;
            }
        }
        indices.push_back(i);
        res.push_back(buf); //将最高conf的bbox填入结果

        // keep top k
        //获取前k个输出，这个应该是针对密集输出的情况，此时input已经做了conf剔除
        count += 1;
        if (count >= topk)
            break;
    }
    output.swap(res);

    return indices;
}

bool YOLOv8::Detect()
{   


    cv::Mat res;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> obj,objs;
    cv::namedWindow("Real Time Detect", cv::WINDOW_AUTOSIZE);

    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }
    res = mRGB.clone();

    this->copy_from_Mat(mRGB, size);
    auto start = std::chrono::system_clock::now();

    this->infer();
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    // 将时间差保留两位小数
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << tc;
    std::string tc_str = "Tensort Infer Time: " + ss.str() + " ms";

    cv::putText(res, tc_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    this->postprocess(obj);
    hardNMS(obj, objs, 0.6, 10);

     this->draw_objects( res, objs, mClassnames, COLORS);
    if(!objs.empty())
    {
        for (auto& obj : objs) {
            // sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);


            int x = (int)obj.rect.x;
            int y = (int)obj.rect.y + 1;
            int classID= (int)obj.label;
            //add by jhw
            cv::Rect2i DetectArea(x, y, obj.rect.width, obj.rect.height );


            // 向对应的向量中添加元素
            this->mmDetectMap[this->mClassnames[classID]].push_back(DetectArea);



            if (count(this->mvDynamicNames.begin(), this->mvDynamicNames.end(), this->mClassnames[obj.label]))
            {

                cv::Rect2i DynamicArea(x, y, obj.rect.width, obj.rect.height );
                this->mvDynamicArea.push_back(DynamicArea);
            }
            if (this->mvDynamicArea.empty())
            {
                cv::Rect2i tDynamicArea(1, 1, 1, 1);
                this->mvDynamicArea.push_back(tDynamicArea);
            }


        }
    }


    return true;
}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;
    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
    
}

void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int*  num_dets = static_cast<int*>(this->host_ptrs[0]);
    auto* boxes    = static_cast<float*>(this->host_ptrs[1]);
    auto* scores   = static_cast<float*>(this->host_ptrs[2]);
    int*  labels   = static_cast<int*>(this->host_ptrs[3]);
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;


    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Object obj;
        obj.rect.x      = x0;
        obj.rect.y      = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob        = *(scores + i);
        obj.label       = *(labels + i);
        objs.push_back(obj);
    }

}

void YOLOv8::draw_objects(
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{

     for (auto& obj : objs) {
         cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
         cv::rectangle(res, obj.rect, color, 2);

         char text[256];
         //CLASS_NAMES[obj.label].c_str() 类别的string
          sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

 
          int      baseLine   = 0;
          cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

         int x = (int)obj.rect.x;
         int y = (int)obj.rect.y + 1;

         if (y > res.rows)
             y = res.rows;

         cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

         cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
     }
    cv::imshow("Real Time Detect", res);
    cv::waitKey(1);
}