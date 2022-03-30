/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "../utility/opencv_cuda.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

#ifdef WITH_VWORKS
#include "vworks_feature_tracker.hpp"
#endif


#define PYR_LEVEL_ 3
//#define PYR_LEVEL 5
//#define PYR_LEVEL 1

#define WIN_SIZE_ cv::Size(21, 21)
//#define WIN_SIZE cv::Size(5, 5)
//#define WIN_SIZE cv::Size(11, 11)


using namespace std;
using namespace camodocal;
using namespace Eigen;

typedef Eigen::Matrix<double,5,1> Vector5d;
// typedef Eigen::Vector<double,5> Vector5d;

typedef Eigen::Matrix<double, 8, 1> TrackFeatureNoId;
typedef pair<int, TrackFeatureNoId> TrackFeature;
typedef vector<TrackFeature> FeatureFramenoId;
typedef map<int, FeatureFramenoId> FeatureFrame;

typedef Eigen::Matrix<double, 8, 1> TrackLineFeatureNoId;
typedef pair<int, TrackLineFeatureNoId> TrackLineFeature;
typedef vector<TrackLineFeature> LineFeatureFramenoId;
typedef map<int, LineFeatureFramenoId> LineFeatureFrame1;
#if 0
typedef map<int, vector<pair<int, Vector4d>>> LineFeatureFrame;
#else
typedef map<int, vector<pair<int, Vector5d>>> LineFeatureFrame;
#endif

//LineFeatureFrame line_temp;
//line_temp[0].emplace_back(0, Vector4d(0, 0, 0, 0));
//lines[feature_id].emplace_back(camera_id, Vector4d(x_startpoint, y_startpoint, x_endpoint, y_endpoint));
//#define lineFeatureFrameEmpty   line_temp

struct PointLineFeature
{
    LineFeatureFrame lineFeatureFrame;
    FeatureFrame featureFrame;
    //vector<Eigen::MatrixXd> detectionsL;
    //vector<Eigen::MatrixXd> detectionsR;

    //vector<vector<Eigen::MatrixXd>> detectionsL;
    //vector<vector<Eigen::MatrixXd>> detectionsR;

};

struct CamFaces
{
    // LineFeatureFrame lineFeatureFrame;
    // FeatureFrame featureFrame;
    Eigen::Quaterniond t01; /// t01, transform from cam1 to cam0, tcx
    Eigen::Quaterniond t02; /// t02, transform from cam2 to cam0, tcx
    Eigen::Quaterniond t03; /// t03, transform from cam3 to cam0, tcx
    Eigen::Quaterniond t04; /// t04, transform from cam4 to cam0, tcx

    //vector<Eigen::MatrixXd> detectionsL;
    //vector<Eigen::MatrixXd> detectionsR;

    //vector<vector<Eigen::MatrixXd>> detectionsL;
    //vector<vector<Eigen::MatrixXd>> detectionsR;

};

class Estimator;
class FisheyeUndist;
class LineFeatureTracker;
namespace FeatureTracker {


class BaseFeatureTracker {
public:
    double focal_x = 0;
    double focal_y = 0;
    double center_x = 0;
    double center_y = 0;
    int cen_width = 0;
    int cen_height = 0;

    // CamFaces cam_faces;
    vector<Eigen::Quaterniond> cam_faces;
/*
    Eigen::Quaterniond t01; /// t01, transform from cam1 to cam0
    Eigen::Quaterniond t02; /// t02, transform from cam2 to cam0
    Eigen::Quaterniond t03; /// t03, transform from cam3 to cam0
    Eigen::Quaterniond t04; /// t04, transform from cam4 to cam0
    */

    BaseFeatureTracker(Estimator * _estimator):
        estimator(_estimator)
    {
        // roger line
        allfeature_cnt = 0;
        frame_cnt = 0;
        sum_time = 0.0;

        width = WIDTH;
        height = ROW;
    }
    
    virtual void setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) = 0;
    //virtual void setPrediction2(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) = 0;
    virtual bool setPrediction2(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPt_cam1 =  Eigen::Vector3d()) = 0;
    virtual Eigen::Vector2d setPrediction3(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPt_cam1 =  Eigen::Vector3d()) = 0;

     //virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, cv::InputArray _img1 = cv::noArray()) = 0;
    //virtual FeatureFrame trackImage(double _cur_time, cv::InputArray _img, cv::InputArray _img1 = cv::noArray(), const LineFeatureFrame &lineFeatureFrame = map<int, vector<pair<int, Vector4d>>>()) = 0;
    virtual PointLineFeature trackImage(double _cur_time, cv::InputArray _img, cv::InputArray _img1 = cv::noArray()) = 0;

    void setFeatureStatus(int feature_id, int status) {
        this->pts_status[feature_id] = status;
        if (status < 0) {
            removed_pts.insert(feature_id);
        }
    }

    virtual void readIntrinsicParameter(const vector<string> &calib_file) = 0;


    void set_orig_imgs(const cv::Mat img_u, const cv::Mat img_d){
        this->img_orig_up = img_u;
        this->img_orig_down = img_d;
    };

    //vector<camodocal::CameraPtr> m_camera;
protected:
    cv::Mat img_orig_up, img_orig_down;
    bool hasPrediction = false;
    int n_id = 0;

    double cur_time;
    double prev_time;
    int height, width;


    /// roger line
    int frame_cnt;
    int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值
    //int frame_cnt;
    vector<int> ids;                     // 每个特征点的id
    vector<int> linetrack_cnt;           // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
    //int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值

    double sum_time;
    double mean_time;
/// roger line






    // cv::Mat img_orig_up, img_orig_down;

    Estimator * estimator = nullptr;
    
    void setup_feature_frame(FeatureFrame & ff, vector<int> ids, vector<cv::Point2f> cur_pts, vector<cv::Point3f> cur_un_pts, vector<cv::Point3f> cur_pts_vel, int camera_id);
    virtual FeatureFrame setup_feature_frame() = 0;

    void drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, map<int, cv::Point2f> prev_pts, map<int, cv::Point2f> predictions = map<int, cv::Point2f>());

    map<int, int> pts_status;
    set<int> removed_pts;

    vector<camodocal::CameraPtr> m_camera;

    bool stereo_cam = false;
};

    // cv::Mat img_orig_up, img_orig_down;

map<int, cv::Point2f> pts_map(vector<int> ids, vector<cv::Point2f> cur_pts);
map<int, cv::Point3f> pts_map(vector<int> ids, vector<cv::Point3f> cur_pts);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
double distance(cv::Point2f &pt1, cv::Point2f &pt2);

#ifndef WITHOUT_CUDA
vector<cv::Point2f> opticalflow_track(cv::Mat & cur_img,
                    std::vector<cv::Mat> & prev_pyr, vector<cv::Point2f> & prev_pts,
                    vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts,
                    bool is_lr_track, std::map<int, cv::Point2f> prediction_points = std::map<int, cv::Point2f>());

std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::Mat& prevImg, int maxLevel_ = 3);
void detectPoints(const cv::Mat & img, vector<cv::Point2f> & n_pts,
        vector<cv::Point2f> & cur_pts, int require_pts);
#endif

vector<cv::Point2f> get_predict_pts(vector<int> id, const vector<cv::Point2f> & cur_pt, const std::map<int, cv::Point2f> & predict);
    
vector<cv::Point2f> opticalflow_track(vector<cv::Mat> * cur_pyr, 
                    vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                    vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts, std::map<int, cv::Point2f> prediction_points = std::map<int, cv::Point2f>());

vector<cv::Point2f> opticalflow_track(cv::Mat & cur_img, vector<cv::Mat> * cur_pyr, 
                    cv::Mat & prev_img, vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                    vector<int> & ids, vector<int> & track_cnt, std::set<int> removed_pts, std::map<int, cv::Point2f> prediction_points = std::map<int, cv::Point2f>());

std::vector<cv::Point2f> detect_orb_by_region(cv::InputArray _img, cv::InputArray _mask, int features, int cols = 4, int rows = 4);
void detectPoints(cv::InputArray img, cv::InputArray mask, vector<cv::Point2f> & n_pts, vector<cv::Point2f> & cur_pts, int require_pts, vector<FisheyeUndist> &fisheys_undists);


bool inBorder(const cv::Point2f &pt, cv::Size shape);

};

