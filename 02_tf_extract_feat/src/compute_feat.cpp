#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace tensorflow;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


tensorflow::Tensor readTensor(string filename, int batchSize)
{
    const int imgSize = 640;
    tensorflow::Tensor input_tensor(DT_FLOAT, tensorflow::TensorShape({batchSize, imgSize, imgSize, 1}));
    
    cv::Mat img = cv::imread(filename, 1);
    cv::resize(img, img, cv::Size(imgSize, imgSize), 0, 0, cv::INTER_NEAREST); // resize

    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    
    for (int i = 0; i < batchSize; i++) {
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(x,y));
                /*input_tensor_mapped(0,y,x,0) = ((float)color[2] - 127) / 128.0;
                 input_tensor_mapped(0,y,x,1) = ((float)color[1] - 127) / 128.0;
                 input_tensor_mapped(0,y,x,2) = ((float)color[0] - 127) / 128.0;*/
            
                /*input_tensor_mapped(i,y,x,0) = ((float)color[2] - 0) / 1.0;
                input_tensor_mapped(i,y,x,1) = ((float)color[1] - 0) / 1.0;
                input_tensor_mapped(i,y,x,2) = ((float)color[0] - 0) / 1.0;*/
                
                input_tensor_mapped(i,y,x,0) = (float)color[2];
                input_tensor_mapped(i,y,x,1) = (float)color[1];
                input_tensor_mapped(i,y,x,2) = (float)color[0];
            }
        }
    }
    
    return input_tensor;
}

int main(int argc, char* argv[]) {
    
    int batchSize = 4;
    string graph_path = "pb.model";
    
    GraphDef graph_def;
    // 读取模型文件
    if (!ReadBinaryProto(Env::Default(), graph_path, &graph_def).ok()) {
        cout << "Read model .pb failed"<<endl;
        return -1;
    }
    
    // 新建session
    unique_ptr<Session> session;
    SessionOptions sess_opt;
    //sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
    (&session)->reset(NewSession(sess_opt));
    if (!session->Create(graph_def).ok()) {
        cout<<"Create graph failed"<<endl;
        return -1;
    }
    
    string image_path("test.png");
    
    tensorflow::Tensor inputTmp = readTensor(image_path, batchSize);
    
    std::vector<tensorflow::Tensor> outputs;
    string inputName = "input";
    string outputName = "head/out_emb"; // graph中的输入节点和输出节点，需要预先知道
    
    std::vector<std::pair<std::string, tensorflow::Tensor>> imgs;
    imgs.push_back(std::pair<std::string, Tensor>(inputName, inputTmp));
    
    Status status = session->Run({imgs}, {outputName}, {}, &outputs); //Run, 得到运行结果，存到outputs中
    if (!status.ok()) {
        cout << "Running model failed"<<endl;
        cout << status.ToString() << endl;
        return -1;
    }
    
    // 得到模型运行结果
    Tensor t = outputs[0];
    auto tmap = t.tensor<float, 2>();
    int output_dim = (int)t.shape().dim_size(1);
    
    // 打印特征
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < output_dim; j++)
        {
            std::cout << tmap(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
