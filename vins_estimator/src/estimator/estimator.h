/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../utility/opencv_cuda.h"

#include "../factor/line_parameterization.h"
#include "../factor/line_projection_factor.h"
class DepthCamManager;


struct RetriveData
{
    /* data */
    int old_index;
    int cur_index;
    double header;
    Vector3d P_old;
    Matrix3d R_old;
    vector<cv::Point2f> measurements;
    vector<int> features_ids;
    bool relocalized;
    bool relative_pose;
    Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    double loop_pose[7];
};

class Estimator
{
  public:

double travelled_distance = 0;
Eigen::Vector3d last_pose;
    Estimator();

    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const FeatureFrame &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    bool is_next_odometry_frame();
    //void inputFisheyeImage(double t, const CvCudaImages & up_imgs, const CvCudaImages & down_imgs, bool is_blank_init = false);
    void inputFisheyeImage(double t, const CvImages & fisheye_imgs_up, const CvImages & fisheye_imgs_down, const cv::Mat  up_raw, const cv::Mat  down_raw);
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const FeatureFrame &image, const LineFeatureFrame &lineFeatureImage,const double header);
    void processMeasurements();

    void processDepthGeneration();

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization(vector<Eigen::Quaterniond> cam_faces);
    void vector2double(vector<Eigen::Quaterniond> cam_faces);
    void double2vector(vector<Eigen::Quaterniond> cam_faces);
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    /// roger line
    void optimizationwithLine();
    void onlyLineOpt(vector<Eigen::Quaterniond> cam_faces);
    void LineBA();
    void LineBAincamera();
    void double2vector2(vector<Eigen::Quaterniond> cam_faces);

    bool intr_assigned = false;
    cv::Mat K1_s, K2_s, R_s, T_s, R1_s, R2_s, P1_s, P2_s, Q_s;
    cv::Mat map11_s, map12_s, map21_s, map22_s;
    Vector3d baseline;

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mBuf;
    std::mutex odomBuf;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<pair<double,FeatureFrame >> featureBuf;
    queue<pair<double,LineFeatureFrame >> lineFeatureBuf;
    queue<pair<double,pair<vector<cv::Mat>,vector<cv::Mat>>>> windowed_images;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;
    std::thread depthThread;

    FeatureTracker::BaseFeatureTracker * featureTracker = nullptr;
    LineFeatureTracker * lineFeatureTracker = nullptr;
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];
    cv::Mat currentImg;
    //vector<pair<double, cv::Mat>> windowed_imgs;
    pair<double, pair<CvImages, CvImages>> windowed_imgs[(WINDOW_SIZE + 1)];
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)] = {0};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;
    float sum_t_feature;
    int begin_time_count;

    double frame_cnt_ = 0;
    double sum_solver_time_ = 0.0;
    double mean_solver_time_ = 0.0;
    double sum_marg_time_ = 0.0;
    double mean_marg_time_=0.0;


    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector4d> line_cloud;
    vector<Vector4d> margin_line_cloud;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

double baseline_;
    double para_LineFeature[NUM_OF_F][SIZE_LINE];
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    std::vector<int> param_feature_id;
    std::map<int, int> param_feature_id_to_index;
    double para_Ex_Pose[2][SIZE_POSE];
  //  double para_Ex_Pose22[4][SIZE_POSE];
    double para_Ex_Pose22[5][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    RetriveData retrive_pose_data, front_pose;
    vector<RetriveData> retrive_data_vector;

    int loop_window_index;
    bool relocalize;
    Vector3d relocalize_t;
    Matrix3d relocalize_r;


    MarginalizationInfo *last_marginalization_info = nullptr;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration = nullptr;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;
    bool fast_prop_inited;

    bool initFirstPoseFlag;

    DepthCamManager * depth_cam_manager = nullptr;

    queue<double> fisheye_imgs_stampBuf;

    queue<std::vector<cv::cuda::GpuMat>> fisheye_imgs_upBuf_cuda;
    queue<std::vector<cv::cuda::GpuMat>> fisheye_imgs_downBuf_cuda;

    queue<std::vector<cv::Mat>> fisheye_imgs_upBuf;
    queue<std::vector<cv::Mat>> fisheye_imgs_downBuf;
    queue<std::pair<double, EigenPose>> odometry_buf;

};
