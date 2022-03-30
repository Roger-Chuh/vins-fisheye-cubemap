/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

#include <eigen3/Eigen/Dense>

using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "../utility/tic_toc.h"
#include "../featureTracker/feature_tracker.h"


#include "parameters.h"
#include "../utility/line_geometry.h"
//#define KEYFRAME_LONGTRACK_THRES 20
#define KEYFRAME_LONGTRACK_THRES_ 20


class FeaturePerFrame
{
  public:
    FeaturePerFrame(const TrackFeatureNoId &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        velocity.z() = _point(7); 
        cur_td = td;
        is_stereo = false;
    }
    void rightObservation(const TrackFeatureNoId &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        velocityRight.z() = _point(7); 
        is_stereo = true;
    }
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector3d velocity, velocityRight;
    bool is_stereo;
    int camera = 0;
};

class FeaturePerId
{
  public:
    const int feature_id = -1;
    int start_frame = -1;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num = 0;
    double estimated_depth = -1;
    bool depth_inited = false;
    bool need_triangulation = true;
    bool is_stereo = false;
    int solve_flag = 0; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    bool good_for_solving = false;
    int main_cam = 0;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    FeaturePerId() {

    }

    int endFrame();
};
class lineFeaturePerFrame
{
public:

    //lineFeaturePerFrame(const Vector4d &line)
   // {
   //     lineobs = line;
   // }


    lineFeaturePerFrame(const Vector5d &line)
    {
        lineobs = line;
    }


    //lineFeaturePerFrame(const Vector8d &line)
   // {
   //     lineobs = line.head<4>();
   //     lineobs_R = line.tail<4>();
   // }

    lineFeaturePerFrame(const Vector10d &line)
    {
        lineobs = line.head<5>();
        lineobs_R = line.tail<5>();
    }


    // Vector4d lineobs;   // 每一帧上的观测
    // Vector4d lineobs_R;
    Vector5d lineobs;   // 每一帧上的观测
    Vector5d lineobs_R;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};
class lineFeaturePerId
{
public:
    const int feature_id;
    int start_frame;

    //  feature_per_frame 是个向量容器，存着这个特征在每一帧上的观测量。
    //                    如：feature_per_frame[0]，存的是ft在start_frame上的观测值; feature_per_frame[1]存的是start_frame+1上的观测
    vector<lineFeaturePerFrame> linefeature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    bool is_triangulation;
    Vector6d line_plucker;

    Vector4d obs_init;
    Vector4d obs_j;
    Vector6d line_plk_init; // used to debug
    Vector3d ptw1;  // used to debug
    Vector3d ptw2;  // used to debug
    Eigen::Vector3d tj_;   // tij
    Eigen::Matrix3d Rj_;
    Eigen::Vector3d ti_;   // tij
    Eigen::Matrix3d Ri_;
    int removed_cnt;
    int all_obs_cnt;    // 总共观测多少次了？

    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    lineFeaturePerId(int _feature_id, int _start_frame)
            : feature_id(_feature_id), start_frame(_start_frame),
              used_num(0), solve_flag(0),is_triangulation(false)
    {
        removed_cnt = 0;
        all_obs_cnt = 1;
    }

    int endFrame();
};
class FeatureManager
{
  public:
    FeatureTracker::BaseFeatureTracker * ft = nullptr;
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    //bool addFeatureCheckParallax(int frame_count, const FeatureFrame &image, const LineFeatureFrame &lineFeatureImage, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(std::map<int, double> deps);
    void removeFailures();
    void clearDepth();
    std::map<int, double> getDepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void triangulatePoint3DPts(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector3d &point0, Eigen::Vector3d &point1, Eigen::Vector3d &point_3d);
    double triangulatePoint3DPts(vector<Eigen::Matrix<double, 3, 4>> &Poses, vector<Eigen::Vector3d> &points, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P, vector<Eigen::Quaterniond> cam_faces);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);
    map<int, FeaturePerId> feature;
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;

    set<int> outlier_features;



    /// roger line
    int getLineFeatureCount();
    list<lineFeaturePerId> linefeature;
    MatrixXd getLineOrthVector(Vector3d Ps[], Vector3d tic[], Matrix3d ric[],vector<Eigen::Quaterniond> cam_faces);
    void setLineOrth(MatrixXd x, Vector3d Ps[], Matrix3d Rs[],Vector3d tic[], Matrix3d ric[],vector<Eigen::Quaterniond> cam_faces);
    MatrixXd getLineOrthVectorInCamera();
    void setLineOrthInCamera(MatrixXd x);

    double reprojection_error( Vector4d obs, Matrix3d Rwc, Vector3d twc, Vector6d line_w ,double &dlt_theta);
    void removeLineOutlier(Vector3d Ps[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces);
    void removeLineOutlier();
    bool addFeatureCheckParallax(int frame_count, const FeatureFrame &image, const LineFeatureFrame &lineFeatureImage, double td);
    void triangulateLine(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    // void triangulateLine2(Vector3d Ps[], Matrix3d Rs_[], Vector3d tic[], Matrix3d ric[], CamFaces cam_faces);
    void triangulateLine2(Vector3d Ps[], Matrix3d Rs_[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces);

    void triangulateLine(double baseline);  // stereo line
    Vector6d triangulateLine_svd(std::map<int, Eigen::Matrix3d> &Rwc_list, std::map<int, Eigen::Vector3d> &twc_list, std::map<int, Eigen::Vector4d> &obs_list, std::map<int, double> &cos_theta_list);


//    std::map<int, Eigen::Vector3d> pi_norm_list;





private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[2];
};

#endif