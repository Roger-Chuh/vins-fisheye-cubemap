/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double THRES_OUTLIER;
double triangulate_max_err; // = 0.5;

double BASELINE;

double IMU_FREQ;
double IMAGE_FREQ;
double FOCAL_LENGTH = 460;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, WIDTH;
int SHOW_WIDTH;
double TD;
int NUM_OF_CAM;
int USE_NEW;
int MIN_TRACE_TO_MARG;
double FISHEYE_FOV_ACTUAL;
int SHOW_DISP;
int SHOW_LINE;
double BASE_LINE;
int EQUALIZE;
int USE_LINE;
int USE_MULTI_LINE_TRIANGULATION;
double LINE_ANGLE_THRES;
double LINE_PIXEL_THRES;
int CUBE_MAP;
double LONG_TRACK_RATIO;
int PRINT_LOG;
int KEYFRAME_LONGTRACK_THRES;

int LK_PYR_LEVEL; //: lk_pyr_level: 3
int LK_WIN_SIZE; //lk_win_size: 21

int STEREO;
int FISHEYE;
double FISHEYE_FOV;
double CENTER_FOV;
double ROT_ANGLE;
int enable_up_top;
int enable_down_top;
int enable_up_side;
int enable_down_side;
int enable_rear_side;

int USE_VXWORKS;
double depth_estimate_baseline;

int USE_IMU;
int USE_GPU;
int PUB_RECTIFY;
int USE_ORB;
Eigen::Matrix3d rectify_R_left;
Eigen::Matrix3d rectify_R_right;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string COMP_IMAGE0_TOPIC, COMP_IMAGE1_TOPIC;
int IS_COMP_IMAGES;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
std::string depth_config;
int MAX_CNT;
int TOP_PTS_CNT;
int SIDE_PTS_CNT;
int MAX_SOLVE_CNT;
int RGB_DEPTH_CLOUD;
int ENABLE_DEPTH;
int ENABLE_PERF_OUTPUT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
int SHOW_FEATURE_ID;

int WARN_IMU_DURATION;
int PUB_FLATTEN;
int FLATTEN_COLOR;
int PUB_FLATTEN_FREQ;

std::string configPath;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    /*
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);*/

    cv::FileStorage fsSettings;
    try {
        fsSettings.open(config_file.c_str(), cv::FileStorage::READ);
    } catch(cv::Exception ex) {
        std::cerr << "ERROR:" << ex.what() << " Can't open config file" << std::endl;
    }
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;

    fsSettings["compressed_image0_topic"] >> COMP_IMAGE0_TOPIC;
    fsSettings["compressed_image1_topic"] >> COMP_IMAGE1_TOPIC;
    IS_COMP_IMAGES = fsSettings["is_compressed_images"];
    MAX_CNT = fsSettings["max_cnt"];
    TOP_PTS_CNT = fsSettings["top_cnt"];
    SIDE_PTS_CNT = fsSettings["side_cnt"];
    MAX_SOLVE_CNT = fsSettings["max_solve_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    USE_ORB = fsSettings["use_orb"];

    SHOW_TRACK = fsSettings["show_track"];
    SHOW_FEATURE_ID = fsSettings["show_track_id"];
    FLOW_BACK = fsSettings["flow_back"];
    RGB_DEPTH_CLOUD = fsSettings["rgb_depth_cloud"];
    ENABLE_DEPTH = fsSettings["enable_depth"];
    THRES_OUTLIER = fsSettings["thres_outlier"];
    triangulate_max_err = fsSettings["tri_max_err"];
    USE_GPU = 0; //fsSettings["use_gpu"];
#ifdef WITHOUT_CUDA
        if (USE_GPU) {
            std::cerr << "Compile with WITHOUT_CUDA mode, use_gpu is not supported!!!" << std::endl;
            exit(-1);
        }
#endif
    FISHEYE = fsSettings["is_fisheye"];
    FISHEYE_FOV = fsSettings["fisheye_fov"];
    CENTER_FOV = fsSettings["center_fov"];
    USE_VXWORKS = fsSettings["use_vxworks"];
    enable_up_top = fsSettings["enable_up_top"];
    enable_up_side = fsSettings["enable_up_side"];
    enable_down_top = fsSettings["enable_down_top"];
    enable_down_side = fsSettings["enable_down_side"];
    enable_rear_side = fsSettings["enable_rear_side"];
    depth_estimate_baseline = fsSettings["depth_estimate_baseline"];
    ENABLE_PERF_OUTPUT = fsSettings["enable_perf_output"];

    IMU_FREQ = fsSettings["imu_freq"];
    IMAGE_FREQ = fsSettings["image_freq"];
    WARN_IMU_DURATION = fsSettings["warn_imu_duration"];
    PUB_FLATTEN = fsSettings["pub_flatten"];
    FLATTEN_COLOR = fsSettings["flatten_color"];
    USE_IMU = fsSettings["imu"];
    PUB_FLATTEN_FREQ = fsSettings["pub_flatten_freq"];
    if (PUB_FLATTEN_FREQ == 0) {
        PUB_FLATTEN_FREQ = 10;
    }

    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    BASE_LINE = -1;


    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    RIC.resize(2);
    TIC.resize(2);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC[0] = Eigen::Matrix3d::Identity();
        TIC[0] = Eigen::Vector3d::Zero();
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC[0] = T.block<3, 3>(0, 0);
        TIC[0] = T.block<3, 1>(0, 3);
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];

    printf("camera number %d\n", NUM_OF_CAM);

    USE_NEW = fsSettings["use_new"];
    MIN_TRACE_TO_MARG = fsSettings["min_trace_to_marg"];
    FISHEYE_FOV_ACTUAL = fsSettings["fisheye_fov_actual"];
    SHOW_LINE = fsSettings["show_line"];
    SHOW_DISP = fsSettings["show_disp"];
    EQUALIZE = fsSettings["equalize"];
    USE_LINE = fsSettings["use_line"];
    USE_MULTI_LINE_TRIANGULATION = fsSettings["use_multi_line_triangulation"];
    LINE_ANGLE_THRES = fsSettings["line_angle_thres"];
    LINE_PIXEL_THRES = fsSettings["line_pixel_thres"];

    CUBE_MAP = fsSettings["cube_map"];
    PRINT_LOG = fsSettings["print_log"];
    LONG_TRACK_RATIO = fsSettings["long_track_ratio"];
    KEYFRAME_LONGTRACK_THRES = fsSettings["keyframe_longtrack_thres"];

    LK_PYR_LEVEL = fsSettings["lk_pyr_level"];
    LK_WIN_SIZE = fsSettings["lk_win_size"];


    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    configPath = config_file.substr(0, pn);


    depth_config = configPath + "/" +((std::string) fsSettings["depth_config"]);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.resize(2);

    CAM_NAMES[0] = cam0Path;

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        
        CAM_NAMES[1] = cam1Path;

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC[1] = T.block<3, 3>(0, 0);
        TIC[1] = T.block<3, 1>(0, 3);
        fsSettings["publish_rectify"] >> PUB_RECTIFY;
    }

    INIT_DEPTH = 5.0; // 2.0; //5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    WIDTH = fsSettings["image_width"];
    SHOW_WIDTH = fsSettings["show_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, WIDTH);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }
    if(PUB_RECTIFY)
    {
        cv::Mat rectify_left;
        cv::Mat rectify_right;
        fsSettings["cam0_rectify"] >> rectify_left;
        fsSettings["cam1_rectify"] >> rectify_right;
        cv::cv2eigen(rectify_left, rectify_R_left);
        cv::cv2eigen(rectify_right, rectify_R_right);

    }

    //fsSettings.release();
}
