#pragma once

#include <opencv2/opencv.hpp>

#ifndef WITHOUT_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <libsgm.h>
#include <opencv2/cudafeatures2d.hpp>
#else
namespace cv {
namespace cuda {
#ifndef HAVE_OPENCV_CUDAIMGPROC
//typedef cv::Mat GpuMat;
#endif
};
};
#endif

#ifdef WITH_VWORKS
#include <NVX/nvx.h>
#include <NVX/nvx_opencv_interop.hpp>
#ifdef OVX
extern ovxio::ContextGuard context;
#else 
extern vx_context context;
#endif
#endif


typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> EigenPose;
typedef std::vector<cv::Mat> CvImages;
typedef std::vector<cv::cuda::GpuMat> CvCudaImages;
