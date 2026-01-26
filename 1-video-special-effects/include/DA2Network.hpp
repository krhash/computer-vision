/*
  Bruce A. Maxwell
  January 2025
  Depth Anything V2 Network wrapper using ONNX Runtime with CUDA GPU support.
*/
#ifndef DA2NETWORK_HPP
#define DA2NETWORK_HPP

#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class DA2Network {
public:
  DA2Network(const char *network_path) {
    std::strncpy(network_path_, network_path, 255);
    std::strncpy(input_names_, "pixel_values", 255);
    std::strncpy(output_names_, "predicted_depth", 255);
    initSession(network_path);
  }

  DA2Network(const char *network_path, const char *input_layer_name, const char *output_layer_name) {
    std::strncpy(network_path_, network_path, 255);
    std::strncpy(input_names_, input_layer_name, 255);
    std::strncpy(output_names_, output_layer_name, 255);
    initSession(network_path);
  }

  ~DA2Network() {
    if(this->input_data_ != nullptr) { 
      delete[] this->input_data_;
      this->input_data_ = nullptr;
    }
    if(this->session_ != nullptr) {
      delete this->session_;
      this->session_ = nullptr;
    }
  }

  int in_height() { return this->height_; }
  int in_width() { return this->width_; }
  int out_height() { return this->out_height_; }
  int out_width() { return this->out_width_; }

  int set_input(const cv::Mat &src, const float scale_factor = 1.0f) {
    if(src.empty()) return -1;
    
    cv::Mat tmp;
    if(scale_factor != 1.0f) {
      cv::resize(src, tmp, cv::Size(), scale_factor, scale_factor);
    } else {
      tmp = src;
    }

    // Allocate memory if size changed
    if(tmp.rows != this->height_ || tmp.cols != this->width_) {
      this->height_ = tmp.rows;
      this->width_ = tmp.cols;

      if(this->input_data_ != nullptr) {
        delete[] this->input_data_;
      }
      
      this->input_data_ = new float[this->height_ * this->width_ * 3];
      this->input_shape_[2] = this->height_;
      this->input_shape_[3] = this->width_;
    }

    // Copy data in planar format with ImageNet normalization
    const int image_size = this->height_ * this->width_;
    for(int i = 0; i < tmp.rows; i++) {
      const cv::Vec3b *ptr = tmp.ptr<cv::Vec3b>(i);
      float *fptrR = &(this->input_data_[i * this->width_]);
      float *fptrG = &(this->input_data_[image_size + i * this->width_]);
      float *fptrB = &(this->input_data_[image_size * 2 + i * this->width_]);
      for(int j = 0; j < tmp.cols; j++) {
        fptrR[j] = ((ptr[j][2] / 255.0f) - 0.485f) / 0.229f;
        fptrG[j] = ((ptr[j][1] / 255.0f) - 0.456f) / 0.224f;
        fptrB[j] = ((ptr[j][0] / 255.0f) - 0.406f) / 0.225f;
      }
    }
    
    return 0;
  }

  int run_network(cv::Mat &dst, const cv::Size &output_size) {
    if(this->session_ == nullptr || this->height_ == 0 || this->input_data_ == nullptr) {
      return -1;
    }

    try {
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        this->input_data_,
        this->height_ * this->width_ * 3,
        this->input_shape_.data(),
        this->input_shape_.size()
      );

      const char* input_names[] = { input_names_ };
      const char* output_names[] = { output_names_ };
      
      auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
      );

      // Get output dimensions
      auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
      
      if(shape.size() == 3) {
        this->out_height_ = static_cast<int>(shape[1]);
        this->out_width_ = static_cast<int>(shape[2]);
      } else if(shape.size() == 2) {
        this->out_height_ = static_cast<int>(shape[0]);
        this->out_width_ = static_cast<int>(shape[1]);
      } else if(shape.size() == 4) {
        this->out_height_ = static_cast<int>(shape[2]);
        this->out_width_ = static_cast<int>(shape[3]);
      } else {
        return -1;
      }

      const float *tensorData = output_tensors[0].GetTensorData<float>();
      if(tensorData == nullptr) return -1;

      // Find min/max for normalization
      float max_val = -1e+6f, min_val = 1e+6f;
      int total_pixels = out_height_ * out_width_;
      
      for(int i = 0; i < total_pixels; i++) {
        if(tensorData[i] < min_val) min_val = tensorData[i];
        if(tensorData[i] > max_val) max_val = tensorData[i];
      }

      // Normalize to [0, 255]
      cv::Mat tmp(out_height_, out_width_, CV_8UC1);
      float range = (max_val - min_val > 1e-6f) ? (max_val - min_val) : 1.0f;
      
      for(int i = 0, k = 0; i < out_height_; i++) {
        uchar *ptr = tmp.ptr<uchar>(i);
        for(int j = 0; j < out_width_; j++, k++) {
          float value = 255.0f * (tensorData[k] - min_val) / range;
          ptr[j] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, value)));
        }
      }
      
      cv::resize(tmp, dst, output_size);
      return 0;
      
    } catch (...) {
      return -1;
    }
  }

private:
  void initSession(const char *network_path) {
    try {
      Ort::SessionOptions session_options;
      
      // CUDA GPU options
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = 0;
      cuda_options.arena_extend_strategy = 0;
      cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      cuda_options.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(cuda_options);
      
      // Optimization settings
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      #ifdef _WIN32
        std::string path_str(network_path);
        std::wstring wpath(path_str.begin(), path_str.end());
        this->session_ = new Ort::Session(env_, wpath.c_str(), session_options);
      #else
        this->session_ = new Ort::Session(env_, network_path, session_options);
      #endif
      
      std::cout << "DA2Network: Model loaded with CUDA GPU acceleration" << std::endl;
      
    } catch (const Ort::Exception& e) {
      std::cerr << "DA2Network error: " << e.what() << std::endl;
      this->session_ = nullptr;
      throw;
    }
  }

  int height_ = 0, width_ = 0;
  int out_height_ = 0, out_width_ = 0;
  char network_path_[256], input_names_[256], output_names_[256];
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "DA2Network"};
  Ort::Session *session_ = nullptr;
  float *input_data_ = nullptr;
  std::array<int64_t, 4> input_shape_{1, 3, 0, 0};
};

#endif // DA2NETWORK_HPP