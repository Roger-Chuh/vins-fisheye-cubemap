/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

extern double FOCAL_LENGTH;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 2000; // 1000;
const int LINE_MIN_OBS = 4; // 2; // 4;       //  5;  // 4;   //  3;   // 2;   // 4; // 8; // 7;  // 5;   // 2; //5; //2; // 5;
extern double triangulate_max_err;
#define UNIT_SPHERE_ERROR


extern double BASE_LINE;

extern double INIT_DEPTH;
extern double THRES_OUTLIER;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;
extern int USE_VXWORKS;
extern std::string configPath;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern std::string depth_config;
extern double TD;
extern double depth_estimate_baseline;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, WIDTH;
extern int SHOW_WIDTH;
extern int NUM_OF_CAM;
extern int USE_NEW;
extern int MIN_TRACE_TO_MARG;
extern double FISHEYE_FOV_ACTUAL;
extern int SHOW_LINE;
extern int SHOW_DISP;
extern int EQUALIZE;
extern int USE_LINE;
extern int USE_MULTI_LINE_TRIANGULATION;
extern double LINE_ANGLE_THRES;
extern double LINE_PIXEL_THRES;
extern int CUBE_MAP;
extern int PRINT_LOG;
extern double LONG_TRACK_RATIO;
extern int KEYFRAME_LONGTRACK_THRES;

extern int LK_PYR_LEVEL;
extern int LK_WIN_SIZE;

extern int STEREO;
extern int FISHEYE;
extern int RGB_DEPTH_CLOUD;
extern int ENABLE_DEPTH;
extern int ENABLE_PERF_OUTPUT;
extern double FISHEYE_FOV;
extern double CENTER_FOV;

extern int enable_up_top;
extern int enable_down_top;
extern int enable_up_side;
extern int enable_down_side;
extern int enable_rear_side;

extern int USE_IMU;
extern int USE_GPU;
extern int PUB_RECTIFY;
extern int USE_ORB;
extern Eigen::Matrix3d rectify_R_left;
extern Eigen::Matrix3d rectify_R_right;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string COMP_IMAGE0_TOPIC, COMP_IMAGE1_TOPIC;

extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int TOP_PTS_CNT;
extern int SIDE_PTS_CNT;
extern int MAX_SOLVE_CNT;

extern int MIN_DIST;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern int SHOW_FEATURE_ID;

extern double IMU_FREQ;
extern double IMAGE_FREQ;

extern int WARN_IMU_DURATION;
extern int PUB_FLATTEN;
extern int FLATTEN_COLOR;
extern int IS_COMP_IMAGES;
extern int PUB_FLATTEN_FREQ;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    SIZE_LINE = 4
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
