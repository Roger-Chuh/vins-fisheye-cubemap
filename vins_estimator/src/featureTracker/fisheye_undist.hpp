#pragma once

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include "cv_bridge/cv_bridge.h"
#include "../utility/opencv_cuda.h"
#include "../utility/tic_toc.h"

#define DEG_TO_RAD (M_PI / 180.0)
// #define REMAP_FUNC cv::INTER_LINEAR
#define REMAP_FUNC cv::INTER_CUBIC
// #define REMAP_FUNC cv::INTER_NEAREST
class FisheyeUndist {

    camodocal::CameraPtr cam;

    //std::vector<cv::cuda::GpuMat> undistMapsGPUX;
    //std::vector<cv::cuda::GpuMat> undistMapsGPUY;
    std::vector<cv::Mat> undistMapsGPUX;
    std::vector<cv::Mat> undistMapsGPUY;
public:
    /// pair存放u和v的remap mat
    std::vector<std::pair<cv::Mat, cv::Mat>> undistMaps;

    cv::Mat fisheye2cam_pt;
    cv::Mat fisheye2cam_id;

    cv::Mat imgL;
    cv::Mat imgR;
    camodocal::CameraPtr cam_top;
    camodocal::CameraPtr cam_side;
    double f_side = 0;
    double f_center = 0;
    double rot_angle = 0;
    double cx_side = 0, cy_side = 0;
    int imgWidth = 0;
    double fov = 0; //in degree
    Eigen::Vector3d cameraRotation;
    bool enable_cuda = false;
    int cam_id = 0;

    double raw_width;
    double raw_height;

    int sideImgHeight = 0;

    std::vector<Eigen::Quaterniond> t;

    FisheyeUndist(const std::string & camera_config_file, int _id, double _fov, bool _enable_cuda = true, int imgWidth = 600):
    imgWidth(imgWidth), fov(_fov), cameraRotation(0, 0, 0), enable_cuda(_enable_cuda), cam_id(_id) {
        cam = camodocal::CameraFactory::instance()
            ->generateCameraFromYamlFile(camera_config_file);
        raw_width = cam->imageWidth();
        raw_height = cam->imageHeight();
        fisheye2cam_pt = cv::Mat::zeros(raw_width, raw_height, CV_32FC2);
        fisheye2cam_id = cv::Mat::ones(raw_width, raw_height, CV_8UC1);
        fisheye2cam_id = fisheye2cam_id * 255;
        undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
        // ROS_INFO("undismap size %ld", undistMaps.size());
        if (enable_cuda) {
            for (auto mat : undistMaps) {
                cv::Mat maps[2];
                cv::split(mat.first, maps);
               // undistMapsGPUX.push_back(cv::cuda::GpuMat(maps[0]));
                //undistMapsGPUY.push_back(cv::cuda::GpuMat(maps[1]));
            }
        }
    }

    cv::cuda::GpuMat undist_id_cuda(cv::Mat image, int _id) {
#ifndef WITHOUT_CUDA
    // 0 TOP or DOWN
    // 1 left 2 front 3 right 4 back

        cv::cuda::GpuMat img_cuda(image);
        cv::cuda::GpuMat output;
        cv::cuda::remap(img_cuda, output, undistMapsGPUX[_id], undistMapsGPUY[_id], REMAP_FUNC);
        return output;
#endif
    }
    
    cv::cuda::GpuMat img_cuda;
    std::vector<cv::Mat> undist_all_cuda_cpu(const cv::Mat & image, bool use_rgb = false, std::vector<bool> mask = std::vector<bool>(0)) {
#ifndef WITHOUT_CUDA
        TicToc up;
        bool has_mask = mask.size() == undistMaps.size();
        if (use_rgb) {
            img_cuda.upload(image);
        } else {
            cv::Mat _tmp;
            cv::cvtColor(image, _tmp, cv::COLOR_BGR2GRAY);
            img_cuda.upload(_tmp);
        }

        std::cout << "Upload cost " << up.toc() << std::endl;


        std::vector<cv::Mat> ret;
        for (unsigned int i = 0; i < undistMaps.size(); i++) {
            cv::Mat tmp;
            if (!has_mask || (has_mask && mask[i]) ) {
                cv::cuda::GpuMat output;
                TicToc remap;
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], REMAP_FUNC);
                std::cout << "Remap cost " << remap.toc() << std::endl;
                TicToc down;
                output.download(tmp);
                std::cout << "Download cost " << down.toc() << std::endl;
            }
            ret.push_back(tmp);
        }
        return ret;
#endif
    }

    std::vector<cv::cuda::GpuMat> undist_all_cuda(const cv::Mat & image, bool use_rgb = false, std::vector<bool> mask = std::vector<bool>(0)) {
#ifndef WITHOUT_CUDA
        cv::cuda::GpuMat img_cuda;
        bool has_mask = mask.size() == undistMaps.size();
        if (use_rgb) {
            img_cuda.upload(image);
        } else {
            cv::Mat _tmp;
            cv::cvtColor(image, _tmp, cv::COLOR_BGR2GRAY);
            img_cuda.upload(_tmp);
        }

        std::vector<cv::cuda::GpuMat> ret;
        for (unsigned int i = 0; i < undistMaps.size(); i++) {
            cv::cuda::GpuMat output;
            if (!has_mask || (has_mask && mask[i]) ) {
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], REMAP_FUNC);
            }
            ret.push_back(output);
        }
        return ret;
#endif
    }

    std::vector<cv::Mat> undist_all(const cv::Mat & image, bool use_rgb = false, bool enable_top = true, bool enable_rear = true) {
        std::vector<cv::Mat> ret;

        ret.resize(undistMaps.size());
        bool disable[5] = {0};
        disable[0] = !enable_top;
        disable[5] = !enable_rear;
        if (use_rgb) {
#pragma omp parallel for num_threads(5)
            for (unsigned int i = 0; i < 5; i++) {
                if (!disable[i]) {
                    cv::remap(image, ret[i], undistMaps[i].first, undistMaps[i].second, REMAP_FUNC);
                }
            }
            return ret;

        } else {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
#pragma omp parallel for  num_threads(5)
            for (unsigned int i = 0; i < 5; i++) {
                if (!disable[i]) {
                }
            }
            return ret;
        }

        return ret;
    }


    void stereo_flatten(const cv::Mat & image1, const cv::Mat & image2, FisheyeUndist * undist2, std::vector<cv::Mat> & lefts, std::vector<cv::Mat> & rights, bool use_rgb = false, 
        bool enable_up_top = true, bool enable_up_rear = true,
        bool enable_down_top = true, bool enable_down_rear = true) {

        auto method = REMAP_FUNC;
        lefts.resize(5);
        rights.resize(5);
        bool disable[10] = {0};
        disable[0] = !enable_up_top;
        disable[4] = !enable_up_rear;

        disable[5] = !enable_down_top;
        disable[9] = !enable_down_rear;
std::cout<<"disable 0 4 5 9: "<< disable[0]<<" "<< disable[4]<<" "<< disable[5]<<" "<< disable[9]<<std::endl;
        if (use_rgb) {
#pragma omp parallel for num_threads(10)
            for (unsigned int i = 0; i < 10; i++) {
                if (!disable[i]) {
                    if (i > 4) {
                        cv::remap(image2, rights[i%5], undist2->undistMaps[i%5].first, undist2->undistMaps[i%5].second, method);
                    } else {
                        cv::remap(image1, lefts[i], undistMaps[i%5].first, undistMaps[i%5].second, method);
                    }
                }
            }
        } else {
            cv::Mat gray1, gray2;
            if (image1.channels() > 1) {
                cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
                cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray1 = image1;
                gray2 = image2;
//                cv::imshow("up_top_img ", gray1 );
  //              cv::waitKey(2);
    //            imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/vio.png",  gray1);

            }

       //     cv::imshow("gray_image", gray1 );
          //  cv::waitKey(2);
        //    imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/vio.png",  gray1);

            //std::cout<<"gray1 size:"<<gray1.size<<std::endl;
            //std::cout<<"gray2 size:"<<gray2.size<<std::endl;

            //std::cout<<"undist2->undistMaps[i%5].first:\n"<<undist2->undistMaps[0%5].first<<std::endl;


#pragma omp parallel for num_threads(10)
            for (unsigned int i = 0; i < 10; i++) {
                if (!disable[i]) {
                    if (i > 4) {
                        cv::remap(gray2, rights[i%5], undist2->undistMaps[i%5].first, undist2->undistMaps[i%5].second, method);
                    } else {
                        cv::remap(gray1, lefts[i], undistMaps[i%5].first, undistMaps[i%5].second, method);
                    }
                }
            }
        }
    }


    std::vector<std::pair<cv::Mat, cv::Mat>> generateAllUndistMap(camodocal::CameraPtr p_cam,
                                          Eigen::Vector3d rotation,
                                          const unsigned &imgWidth,
                                          const double &fov //degree
    ) {
        // ROS_INFO("Generating undistortion maps:");
        double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
        if (sideVerticalFOV < 0)
            sideVerticalFOV = 0;
        double centerFOV = fov * DEG_TO_RAD - sideVerticalFOV * 2;
        rot_angle = M_PI/2;
        if(USE_NEW)
        {
            centerFOV = CENTER_FOV * DEG_TO_RAD;
            sideVerticalFOV = (fov - CENTER_FOV) * DEG_TO_RAD /2;
            rot_angle = (centerFOV + sideVerticalFOV)/2;  // M_PI/2;//

        }
        ROS_INFO("Build for camera %d", cam_id);
        ROS_INFO("Center FOV: %f_center", centerFOV);

        // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
        f_center = (double)imgWidth / 2 / tan(centerFOV / 2);
        if (!USE_NEW)
        {
            f_side = (double)imgWidth / 2;
        }
        else
        {
            f_side = f_center;
        }


        // sideImgHeight = sideVerticalFOV / centerFOV * imgWidth;
        sideImgHeight = 2 * f_side * tan(sideVerticalFOV/2);

        ROS_INFO("Side image height: %d", sideImgHeight);
        std::vector<std::pair<cv::Mat, cv:: Mat>> maps;
        maps.reserve(5);

        // test points
        Eigen::Vector3d testPoints[] = {
            Eigen::Vector3d(0, 0, 1),
            Eigen::Vector3d(1, 0, 1),
            Eigen::Vector3d(0, 1, 1),
            Eigen::Vector3d(1, 1, 1),
        };
        for (unsigned int i = 0; i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++)
        {
            Eigen::Vector2d temp;
            p_cam->spaceToPlane(testPoints[i], temp);
            // ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to (%.2f,%.2f)", i,
                    // testPoints[i][0], testPoints[i][1], testPoints[i][2],
                    // temp[0], temp[1]);
        }
        
        t.resize(5);
        t[0] = Eigen::Quaterniond::Identity();

        // ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center, f_side);

        cam_top = camodocal::PinholeCameraPtr( new camodocal::PinholeCamera("top",
                  imgWidth, imgWidth,0, 0, 0, 0,
                  f_center, f_center, imgWidth/2, imgWidth/2));
         
        cx_side = imgWidth/2;
        cy_side = sideImgHeight/2;
        cam_side = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera("side",
                  imgWidth, sideImgHeight,0, 0, 0, 0,
                  f_side, f_side, imgWidth/2, sideImgHeight/2));
/// 因为这是第0号针孔相机，不需要做额外旋转，所以直接调genOneUndistMap函数，然后push进去
        maps.push_back(genOneUndistMap(0, p_cam, t[0], imgWidth, imgWidth, f_center));

        Eigen::Quaterniond t0 = t[0];
        if (cam_id == 1) {
            std::cout << "Is camera 1 will invert T" << std::endl;
            // TODO
            if (!USE_NEW) {
                t0 = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
            }
            else
            {
                t0 = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
            }
        };

        if (sideImgHeight > 0)
        {/// left front right rear
        if(!USE_NEW) {
            //facing y
            t[1] = t0 * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(1, p_cam, t[1], imgWidth, sideImgHeight, f_side));
            //turn right/left?
            t[2] = t[1] * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(2, p_cam, t[2], imgWidth, sideImgHeight, f_side));
            t[3] = t[2] * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(3, p_cam, t[3], imgWidth, sideImgHeight, f_side));
            t[4] = t[3] * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(4, p_cam, t[4], imgWidth, sideImgHeight, f_side));
            }
        else {
            //facing y
            t[1] = t0 * Eigen::AngleAxis<double>(-rot_angle , Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(1, p_cam, t[1], imgWidth, sideImgHeight, f_side));
            //turn right/left?
            t[2] = t[1] * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(2, p_cam, t[2], imgWidth, sideImgHeight, f_side));
            t[3] = t[2] * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(3, p_cam, t[3], imgWidth, sideImgHeight, f_side));
            t[4] = t[3] * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(4, p_cam, t[4], imgWidth, sideImgHeight, f_side));
            }
        }
        return maps;
    }


    //std::pair<int, cv::Point2f> project_point_to_vcam_id2(Eigen::Vector3d pts_cam) {
    Eigen::Vector2d project_point_to_vcam_id2(Eigen::Vector3d pts_cam) {
        Eigen::Vector2d imgPoint;
        /// 找到与3d球坐标对应的2d包含畸变的鱼眼图像2d坐标
        cam->spaceToPlane(pts_cam, imgPoint);
        return imgPoint;
    }

    std::pair<int, cv::Point2f> project_point_to_vcam_id(Eigen::Vector3d pts_cam) {
        //First project the point to fisheye image plane
        /// 这个归一化 [球坐标] 应该落在[ 前、后、左、右、中] 哪一部分针孔坐标系上, 并给出在针孔相机下的2d坐标
        Eigen::Vector2d imgPoint;
        /// 找到与3d球坐标对应的2d包含畸变的鱼眼图像2d坐标
        cam->spaceToPlane(pts_cam, imgPoint);
        cv::Point2f pt = fisheye2cam_pt.at<cv::Vec2f>(cv::Point(imgPoint.x(), imgPoint.y()));
        int id = 255;
        if ( !imgPoint.hasNaN() && 
                imgPoint.x() >= 0 && imgPoint.x() < raw_width &&
                imgPoint.y() >= 0 && imgPoint.y() < raw_height
            ) {
                fisheye2cam_id.at<uint8_t>(cv::Point(imgPoint.x(), imgPoint.y()));
            }
        if (id != 255) {
            // std::cout << "\n\nPT" << pts_cam << " IMG " << imgPoint << " ID" << id << " PT " << pt << std::endl;
            return std::make_pair(id, pt);
        } else {
            // std::cout << "Map not found, try project" << std::endl;
        }

        for (int i = 0; i < 5; i ++) {
            /// 把球面3d坐标对应的反投影射线根据t0,t1,t2,t3,t4做旋转，旋转到5个针孔相机坐标下（旋转后仍是3d的vector）
            auto pts_vcam = t[i].inverse()*pts_cam;
            int width = imgWidth;
            int height = sideImgHeight;
            if (pts_vcam.z() < 0) {
                continue;
            }

            /// 做这个区分应该是cam_top和cam_side的内参并不一样（虽然我在用的时候是强制cam_top和cam_side内参一致的）
            if (i == 0) {
                /// 0号区域强制是个正方形，不能改
                height = imgWidth;
                cam_top->spaceToPlane(pts_vcam, imgPoint);
            } else {
                /// 这里是为了区分cam_top和cam_side内参不一样，我在用的时候是强制cam_top和cam_side内参一样的
                cam_side->spaceToPlane(pts_vcam, imgPoint);
            }

/// imgPoint表示在各个（5个）针孔相机下的像素坐标
            if (!imgPoint.hasNaN() && 
                imgPoint.x() >= 0 && imgPoint.x() < width &&
                imgPoint.y() >= 0 && imgPoint.y() < height
            ) {
                id = i;
                pt.x = imgPoint.x();
                pt.y = imgPoint.y();
                /// 2d像素点落在图像内，表示找到的改球坐标所属的针孔相机id，把for循环break调，没必要在做for循环了，
                /// 因为1个3d球坐标只可能落在一个针孔相机内，不存在一对多的情况
                break;
            }
        }

        
        if (id == 255) {
            // std::cout << "PT " << pts_cam << "not found in image" << std::endl;
            return std::make_pair(-1, cv::Point2f(0, 0));
        } else {
            // std::cout << "\n\nPT" << pts_cam << "IMG " << imgPoint << "ID" << id << "PT" << pt << std::endl;
            /// 前面的id表示预测的特征点落在0，1，2，3，4哪个针孔相机内
            return std::make_pair(id, pt);
        }

    }

    std::pair<cv::Mat, cv::Mat> genOneUndistMap(
        int _id, 
        camodocal::CameraPtr p_cam,
        Eigen::Quaterniond rotation,
        const unsigned &imgWidth,
        const unsigned &imgHeight,
        const double &f_center) {
                cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
        ROS_DEBUG("Generating map of size (%d,%d)", map.size[0], map.size[1]);
        ROS_DEBUG("Perspective facing (%.2f,%.2f,%.2f)",
                (rotation * Eigen::Vector3d(0, 0, 1))[0],
                (rotation * Eigen::Vector3d(0, 0, 1))[1],
                (rotation * Eigen::Vector3d(0, 0, 1))[2]);
        for (unsigned int x = 0; x < imgWidth; x++)
            for (unsigned int y = 0; y < imgHeight; y++)
            {
                Eigen::Vector3d objPoint =
                    rotation *
                    Eigen::Vector3d(
                        ((double)x - (double)imgWidth / 2),
                        ((double)y - (double)imgHeight / 2),
                        f_center);
                Eigen::Vector2d imgPoint;
                p_cam->spaceToPlane(objPoint, imgPoint);

                map.at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
                if(!isnan(imgPoint.x()) && !isnan(imgPoint.y()) && 
                    imgPoint.x() >=0 && imgPoint.x() <= raw_width &&
                    imgPoint.y() >=0 && imgPoint.y() <= raw_height
                ) {
                    auto & pt = fisheye2cam_pt.at<cv::Vec2f>(cv::Point(imgPoint.x(), imgPoint.y()));
                    fisheye2cam_id.at<uint8_t>(cv::Point(imgPoint.x(), imgPoint.y())) = _id;
                    pt[0] = x;
                    pt[1] = y;
                }
                
            }

        ROS_DEBUG("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
                map.at<cv::Vec2f>(cv::Point(0, 0))[0],
                map.at<cv::Vec2f>(cv::Point(0, 0))[1],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

        Eigen::Vector3d objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)0 - (double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        // std::cout << objPoint << std::endl;

        objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        // std::cout << objPoint << std::endl;
        cv::Mat map1, map2;
        cv::convertMaps(map, cv::Mat(), map1, map2, CV_16SC2);
        return std::make_pair(map, cv::Mat());
        // return std::make_pair(map1, map2);
    }

};

