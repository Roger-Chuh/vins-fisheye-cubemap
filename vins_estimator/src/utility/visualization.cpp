/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include <vins/VIOKeyframe.h>
#include <sensor_msgs/PointCloud.h>
#include <vins/FlattenImages.h>
#include "cv_bridge/cv_bridge.h"
#include "../utility/ros_utility.h"



//#include "../fisheyeNode.hpp"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_right;
ros::Publisher pub_rectify_pose_left;
ros::Publisher pub_rectify_pose_right;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;
ros::Publisher pub_flatten_images;
ros::Publisher pub_center_images;
ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;
ros::Publisher pub_viokeyframe;
ros::Publisher pub_viononkeyframe;
ros::Publisher pub_bias;


// ros::Publisher pub_point_cloud, pub_margin_cloud,
ros::Publisher pub_lines, pub_marg_lines;
ros::Publisher pub_center_depth;


CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<geometry_msgs::PoseStamped>("camera_pose", 1000);
    pub_camera_pose_right = n.advertise<geometry_msgs::PoseStamped>("camera_pose_right", 1000);
    pub_rectify_pose_left = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_left", 1000);
    pub_rectify_pose_right = n.advertise<geometry_msgs::PoseStamped>("rectify_pose_right", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_viokeyframe = n.advertise<vins::VIOKeyframe>("viokeyframe", 1000);
    pub_viononkeyframe = n.advertise<vins::VIOKeyframe>("viononkeyframe", 1000);
    //pub_flatten_images = n.advertise<vins::FlattenImages>("flatten_images", 1000);
    pub_center_images = n.advertise<sensor_msgs::Image>("center_images", 1000);
    pub_bias = n.advertise<sensor_msgs::Imu>("imu_bias", 1000);

    //pub_lines =
    pub_lines = n.advertise<visualization_msgs::Marker>("lines_cloud", 1000);
    pub_marg_lines = n.advertise<visualization_msgs::Marker>("history_lines_cloud", 1000);

    //pub_depth
    pub_center_depth = n.advertise<sensor_msgs::Image>("center_depth", 1000);


    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}


void pubIMUBias(const Eigen::Vector3d &Ba, const Eigen::Vector3d Bg, const std_msgs::Header &header) {
    sensor_msgs::Imu bias;
    bias.header = header;
    bias.linear_acceleration.x = Ba.x();
    bias.linear_acceleration.y = Ba.y();
    bias.linear_acceleration.z = Ba.z();

    bias.angular_velocity.x = Ba.x();
    bias.angular_velocity.y = Ba.y();
    bias.angular_velocity.z = Ba.z();

    pub_bias.publish(bias);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "odometry";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);

}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{

    
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "odometry";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.toSec() * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
        foutC.close();
        Eigen::Vector3d tmp_T = estimator.Ps[WINDOW_SIZE];
#if 0
        Eigen::Vector3d tmp_T0 = estimator.Ps[0];
        Eigen::Vector3d tmp_T1 = estimator.Ps[1];
        Eigen::Vector3d delta_T = tmp_T0 - tmp_T1;
        // estimator.travelled_distance = estimator.travelled_distance + sqrt(delta_T(0)*delta_T(0) + delta_T(1)*delta_T(1) + delta_T(2)*delta_T(2));
#endif
        printf("######## [time: %f], [kf: %d], [t: %5.3f %5.3f %5.3f], [q: %4.2f %4.2f %4.2f %4.2f], [td: %3.6fms], [travelled_distance: %0.3fm]\n", header.stamp.toSec(), !estimator.marginalization_flag,
            tmp_T.x(), tmp_T.y(), tmp_T.z(),
            tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z(), estimator.td*1000, estimator.travelled_distance);

        vins::VIOKeyframe vkf;
        vkf.header = header;
        int i = WINDOW_SIZE;
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);
        Vector3d P_r = P + R * estimator.tic[0];
        Quaterniond R_r = Quaterniond(R * estimator.ric[0]);
        vkf.pose_cam.position.x = P_r.x();
        vkf.pose_cam.position.y = P_r.y();
        vkf.pose_cam.position.z = P_r.z();
        vkf.pose_cam.orientation.x = R_r.x();
        vkf.pose_cam.orientation.y = R_r.y();
        vkf.pose_cam.orientation.z = R_r.z();
        vkf.pose_cam.orientation.w = R_r.w();

        vkf.camera_extrisinc.position.x = estimator.tic[0].x();
        vkf.camera_extrisinc.position.y = estimator.tic[0].y();
        vkf.camera_extrisinc.position.z = estimator.tic[0].z();

        Quaterniond ric = Quaterniond(estimator.ric[0]);
        ric.normalize();

        vkf.camera_extrisinc.orientation.x = ric.x();
        vkf.camera_extrisinc.orientation.y = ric.y();
        vkf.camera_extrisinc.orientation.z = ric.z();
        vkf.camera_extrisinc.orientation.w = ric.w();

        vkf.pose_drone = odometry.pose.pose;
        
        vkf.header.stamp = odometry.header.stamp;

        for (auto &_it : estimator.f_manager.feature)
        {
            auto & it_per_id = _it.second;
            int frame_size = it_per_id.feature_per_frame.size();
            // ROS_INFO("START FRAME %d FRAME_SIZE %d WIN SIZE %d solve flag %d", it_per_id.start_frame, frame_size, WINDOW_SIZE, it_per_id.solve_flag);


            bool isFrontImage = estimator.featureTracker->setPrediction2(it_per_id.feature_per_frame[0].point, it_per_id.feature_per_frame[0].point);
            isFrontImage = true;
            //if(it_per_id.start_frame < WINDOW_SIZE && it_per_id.start_frame + frame_size >= WINDOW_SIZE&& it_per_id.solve_flag == 1 && isFrontImage)
            //if(it_per_id.start_frame < WINDOW_SIZE && it_per_id.start_frame + frame_size >= WINDOW_SIZE&& it_per_id.solve_flag < 2 && isFrontImage)
            if(it_per_id.start_frame < WINDOW_SIZE && it_per_id.start_frame + frame_size >= WINDOW_SIZE&& it_per_id.solve_flag < 2 && isFrontImage)
            {
                geometry_msgs::Point32 fp2d_uv;
                geometry_msgs::Point32 fp2d_norm;
                int imu_j = frame_size - 1;

                fp2d_uv.x = it_per_id.feature_per_frame[imu_j].uv.x();
                fp2d_uv.y = it_per_id.feature_per_frame[imu_j].uv.y();
                fp2d_uv.z = 0;

                fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x();
                fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y();
                //fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x() / it_per_id.feature_per_frame[imu_j].point.z();
                //fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y() / it_per_id.feature_per_frame[imu_j].point.z();
                fp2d_norm.z = 0;

                vkf.feature_points_id.push_back(it_per_id.feature_id);
                vkf.feature_points_2d_uv.push_back(fp2d_uv);
                vkf.feature_points_2d_norm.push_back(fp2d_norm);

                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_j] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam])
                                    + estimator.Ps[imu_j];

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);

                vkf.feature_points_3d.push_back(p);
                vkf.feature_points_flag.push_back(it_per_id.solve_flag);
            }

        }
        pub_viononkeyframe.publish(vkf);
    }
    
   
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        geometry_msgs::PoseStamped odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.position.x = P.x();
        odometry.pose.position.y = P.y();
        odometry.pose.position.z = P.z();
        odometry.pose.orientation.x = R.x();
        odometry.pose.orientation.y = R.y();
        odometry.pose.orientation.z = R.z();
        odometry.pose.orientation.w = R.w();

        if(STEREO)
        {
            Vector3d P_r = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1]);

            geometry_msgs::PoseStamped odometry_r;
            odometry_r.header = header;
            odometry_r.header.frame_id = "world";
            odometry_r.pose.position.x = P_r.x();
            odometry_r.pose.position.y = P_r.y();
            odometry_r.pose.position.z = P_r.z();
            odometry_r.pose.orientation.x = R_r.x();
            odometry_r.pose.orientation.y = R_r.y();
            odometry_r.pose.orientation.z = R_r.z();
            odometry_r.pose.orientation.w = R_r.w();
            pub_camera_pose_right.publish(odometry_r);
            if(PUB_RECTIFY)
            {
                Vector3d R_P_l = P;
                Vector3d R_P_r = P_r;
                Quaterniond R_R_l = Quaterniond(estimator.Rs[i] * estimator.ric[0] * rectify_R_left.inverse());
                Quaterniond R_R_r = Quaterniond(estimator.Rs[i] * estimator.ric[1] * rectify_R_right.inverse());
                geometry_msgs::PoseStamped R_pose_l, R_pose_r;
                R_pose_l.header = header;
                R_pose_r.header = header;
                R_pose_l.header.frame_id = "world";
                R_pose_r.header.frame_id = "world";
                R_pose_l.pose.position.x = R_P_l.x();
                R_pose_l.pose.position.y = R_P_l.y();
                R_pose_l.pose.position.z = R_P_l.z();
                R_pose_l.pose.orientation.x = R_R_l.x();
                R_pose_l.pose.orientation.y = R_R_l.y();
                R_pose_l.pose.orientation.z = R_R_l.z();
                R_pose_l.pose.orientation.w = R_R_l.w();

                R_pose_r.pose.position.x = R_P_r.x();
                R_pose_r.pose.position.y = R_P_r.y();
                R_pose_r.pose.position.z = R_P_r.z();
                R_pose_r.pose.orientation.x = R_R_r.x();
                R_pose_r.pose.orientation.y = R_R_r.y();
                R_pose_r.pose.orientation.z = R_R_r.z();
                R_pose_r.pose.orientation.w = R_R_r.w();

                pub_rectify_pose_left.publish(R_pose_l);
                pub_rectify_pose_right.publish(R_pose_r);

            }
        }

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto _it : estimator.f_manager.feature)
    {
        auto it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &_it : estimator.f_manager.feature)
    {
        auto & it_per_id = _it.second;
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

//static cv::Mat ComputeDispartiyMap(cv::Mat & left, cv::Mat & right, int &intr_assigned, Matrix3d &intrL, Matrix3d &intrR, Matrix3d &Rot, Vector3d &Trans,
//cv::Mat &K1,cv::Mat & K2,cv::Mat & R,cv::Mat & T, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &Q,
//cv::Mat &_map11, cv::Mat &_map12, cv::Mat &_map21, cv::Mat &_map22) {
    //static bool ComputeDispartiyMap(bool assign, cv::Mat &disparity, cv::Mat & left, cv::Mat & right, Matrix3d &intrL, Matrix3d &intrR, Matrix3d &Rot, Vector3d &Trans,
      //                                 cv::Mat &K1,cv::Mat & K2,cv::Mat & R,cv::Mat & T, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &Q,
        //                               cv::Mat &_map11, cv::Mat &_map12, cv::Mat &_map21, cv::Mat &_map22) {


static void writeMatToFile(cv::Mat& m, string filename)
{
    std::ofstream fout(filename);

    if (!fout)
    {
        std::cout << "File Not Opened" << std::endl;
        return;
    }

    for (int i = 0; i<m.rows; i++)
    {
        for (int j = 0; j<m.cols; j++)
        {
            fout << m.at<float>(i, j) << "\t";
        }
        fout << std::endl;
    }

    fout.close();
}

static cv::Mat ComputeDispartiyMap(cv::Mat & left, cv::Mat & right,const Estimator & estimator) {
    // stereoRectify(InputArray cameraMatrix1, InputArray distCoeffs1,
    // InputArray cameraMatrix2, InputArray distCoeffs2,
    //Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2,
    //OutputArray Q,
    //  int flags=CALIB_ZERO_DISPARITY, double alpha=-1,
    // Size newImageSize=Size(), Rect* validPixROI1=0, Rect* validPixROI2=0 )¶
#if 0
    TicToc tic;
    if (first_init) {
        cv::Mat _Q;
        cv::Size imgSize = left.size();

        // std::cout << "ImgSize" << imgSize << "\nR" << R << "\nT" << T << std::endl;
        cv::stereoRectify(cameraMatrix, cv::Mat(), cameraMatrix, cv::Mat(), imgSize,
                          R, T, R1, R2, P1, P2, _Q, 0);
        std::cout << Q << std::endl;
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R1, P1, imgSize, CV_32FC1, _map11,
                                _map12);
        initUndistortRectifyMap(cameraMatrix, cv::Mat(), R2, P2, imgSize, CV_32FC1, _map21,
                                _map22);
        _Q.convertTo(Q, CV_32F);

        first_init = false;
    }


    cv::Mat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    cv::remap(left, leftRectify, _map11, _map12, cv::INTER_LINEAR);
    cv::remap(right, rightRectify, _map21, _map22, cv::INTER_LINEAR);

    auto sgbm = cv::StereoSGBM::create(params.min_disparity, params.num_disp, params.block_size,
                                       params.p1, params.p2, params.disp12Maxdiff, params.prefilterCap, params.uniquenessRatio, params.speckleWindowSize,
                                       params.speckleRange, params.mode);

    // sgbm->compute(right_rect, left_rect, disparity);
    sgbm->compute(leftRectify, rightRectify, disparity);
    ROS_INFO("CPU SGBM time cost %fms", tic.toc());
    if (show) {
        cv::Mat disparity_color, disp;
        disparity.convertTo(disp, CV_8U, 255. / params.num_disp/16);
        cv::applyColorMap(disp, disparity_color, cv::COLORMAP_RAINBOW);

        cv::Mat _show;

        cv::hconcat(leftRectify, rightRectify, _show);
        cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
        cv::hconcat(_show, disparity_color, _show);

        cv::imshow("RAW DISP", _show);
        cv::waitKey(2);
    }
    return disparity;
#else
    /*
    static Ptr<StereoSGBM> cv::StereoSGBM::create	(	int 	minDisparity = 0,
                                                          int 	numDisparities = 16,
                                                          int 	blockSize = 3,
                                                          int 	P1 = 0,
                                                          int 	P2 = 0,
                                                          int 	disp12MaxDiff = 0,
                                                          int 	preFilterCap = 0,
                                                          int 	uniquenessRatio = 0,
                                                          int 	speckleWindowSize = 0,
                                                          int 	speckleRange = 0,
                                                          int 	mode = StereoSGBM::MODE_SGBM
    )*/
    /*
    Parameters
    minDisparity	Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    numDisparities	Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
    blockSize	Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    P1	The first parameter controlling the disparity smoothness. See below.
    P2	The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively).
    disp12MaxDiff	Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    preFilterCap	Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
    uniquenessRatio	Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    speckleWindowSize	Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleRange	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    mode	Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .

            */
#if 0
    int minDisparity = 0;
    int numDisparities = 16; // 32;
    int blockSize = 5;
    int P1 = 8*1*blockSize*blockSize;
    int P2 = 32*1*blockSize*blockSize;
    int disp12MaxDiff = 2;
    int preFilterCap = 63;
    int uniquenessRatio = 7;
    int speckleWindowSize = 100;
    int speckleRange = 1;
    int mode = 0;
    auto sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                       P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize,
                                       speckleRange, mode);
    cv::Mat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    leftRectify = left;
    rightRectify = right;
    sgbm->compute(leftRectify, rightRectify, disparity);
    if (1) {
        cv::Mat disparity_color, disp;
        disparity.convertTo(disp, CV_8U, 255. / numDisparities/16);
        cv::applyColorMap(disp, disparity_color, cv::COLORMAP_RAINBOW);

        cv::Mat _show;

        cv::hconcat(leftRectify, rightRectify, _show);
        cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
        cv::hconcat(_show, disparity_color, _show);

        cv::imshow("RAW DISP", _show);
        cv::waitKey(2);
    }
    return disparity;
#else

    enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };
    //int numberOfDisparities = 16; 32; //16; //((left.cols / 8) + 15) & -16;
    int numberOfDisparities = ((left.cols / 8) + 15) & -16;
    // int numberOfDisparities = ((left.cols / 16) + 15) & -16;
    //int numberOfDisparities = ((left.cols / 12) + 15) & -16;
    printf("********* number of disparities: %d \n",numberOfDisparities);
    //cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
    //sgbm->setPreFilterCap(63);
    sgbm->setPreFilterCap(31);
    int SADWindowSize = 5; // 9;
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = left.channels();
    int min_disp = 1; //0.2; // 3; //1; //5;
    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(min_disp);
    sgbm->setNumDisparities(numberOfDisparities);
    //sgbm->setUniquenessRatio(10);
    sgbm->setUniquenessRatio(5);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);


    cv::Mat D1, D2;
    //cv::Mat K1, K2, D1, D2, R, T; // , M11, M22;
//    Eigen::VectorXd dist1, dist2;
//    dist1.resize(5);
//    dist2.resize(5);
//    dist1.setZero();
//    dist2.setZero();
//    cv::eigen2cv(dist1, D1);
//    cv::eigen2cv(dist2, D2);
    //D1.resize(5,1);
    //D2.resize(5,1);
    //D1 << 0.0, 0.0, 0.0, 0.0,0.0;
    //if (K1.at<double>(0,0) == 0) {
#if 0
    if (assign){
        Eigen::VectorXd dist1, dist2;
        dist1.resize(5);
        dist2.resize(5);
        dist1.setZero();
        dist2.setZero();
        cv::eigen2cv(dist1, D1);
        cv::eigen2cv(dist2, D2);

        cv::eigen2cv(intrL, K1);
        cv::eigen2cv(intrR, K2);
        cv::eigen2cv(Rot, R);
        cv::eigen2cv(Trans, T);
//        cv::eigen2cv(dist1, D1);
//        cv::eigen2cv(dist2, D2);

        cv::Size imgSize = left.size();
        //cv::Mat R1, R2, P1, P2, Q_;
        cv::Mat Q_;

        cv::Rect validRoi[2];
        stereoRectify(K1, D1, K2, D2, imgSize, R, T, R1, R2, P1, P2, Q_,
                      cv::CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);
        std::cout << Q_ << std::endl;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, _map11,
                                _map12);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, _map21,
                                _map22);
        Q_.convertTo(Q, CV_32F);
        std::cout << Q << std::endl;
        assign = !assign;
       // intr_assigned = 1; //true;
    }
#endif

    cv::Mat leftRectify, rightRectify, disparity(left.size(), CV_8U);
    // leftRectify = left;
    // rightRectify = right;
    std::cout<<"estimator.map11_s size: "<<estimator.map11_s.size<<std::endl;
#if 0
    cv::remap(left, leftRectify, estimator.map11_s, estimator.map12_s, cv::INTER_LINEAR);
    cv::remap(right, rightRectify, estimator.map21_s, estimator.map22_s, cv::INTER_LINEAR);
#else
    cv::remap(left, leftRectify, estimator.map11_s, estimator.map12_s, cv::INTER_CUBIC);
    cv::remap(right, rightRectify, estimator.map21_s, estimator.map22_s, cv::INTER_CUBIC);
#endif
    int alg = STEREO_SGBM;
    if (alg == STEREO_HH)
        sgbm->setMode(cv::StereoSGBM::MODE_HH);
    else if (alg == STEREO_SGBM)
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    else if (alg == STEREO_3WAY)
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->compute(leftRectify, rightRectify, disparity);
    // if (1) {
    if (SHOW_DISP) {
        cv::Mat disparity_color, disp;
        disparity.convertTo(disp, CV_8U, 255.0 / numberOfDisparities/16.0);
        //cv::applyColorMap(disp, disparity_color, cv::COLORMAP_JET);
        cv::applyColorMap(disp, disparity_color, cv::COLORMAP_RAINBOW);

        cv::Mat _show;

        cv::hconcat(leftRectify, rightRectify, _show);
        cv::cvtColor(_show, _show, cv::COLOR_GRAY2BGR);
        cv::hconcat(_show, disparity_color, _show);

        cv::imshow("RAW DISP", _show);
        cv::waitKey(2);
    }
    //return assign; //disparity;



    cv::Mat imgDisparity32F, depthMap;
    TicToc tic1;
    disparity.convertTo(imgDisparity32F, CV_32F, 1./16);


    depthMap = estimator.featureTracker->focal_x * estimator.baseline.norm() / imgDisparity32F;
    depthMap = 1000 * depthMap;
    cv::Mat depth16U(left.size(), CV_16UC1);
    depthMap.convertTo(depth16U, CV_16UC1);


    cv::threshold(imgDisparity32F, imgDisparity32F, min_disp, 1000, cv::THRESH_TOZERO);
    //ROS_INFO("Convert cost %fms", tic1.toc());
    //Vector3d
   // depthMap = estimator.featureTracker->focal_x * estimator.baseline.norm() / imgDisparity32F;
    // cv::threshold(depthMap, depthMap, 5, 1000, cv::THRESH_TOZERO_INV);

    cv::threshold(depthMap, depthMap, 10, 1000, cv::THRESH_TOZERO_INV);

    // cv::threshold(depthMap, depthMap, 30, 1000, cv::THRESH_TOZERO_INV);
    // cv::threshold(depthMap, depthMap, 3, 1000, cv::THRESH_TOZERO_INV);

    //TicToc tic;
    //cv::Mat XYZ = cv::Mat::zeros(imgDisparity32F.rows, imgDisparity32F.cols, CV_32FC3);   // Output point cloud
    //cv::reprojectImageTo3D(imgDisparity32F, XYZ, estimator.Q_s);    // cv::project
    //ROS_INFO("Reproject to 3d cost %fms", tic.toc());


    //return imgDisparity32F;
    //return depthMap;
    return depth16U;
#endif
#endif
}
void pubKeyframe(const Estimator &estimator)
{
    int frameInd = 2; // WINDOW_SIZE-1; // 2;

    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        vins::VIOKeyframe vkf;
        /// 妙啊，i-2其实就是次次新帧，这一帧也正是关键帧，这种写法也跟我之前对滑窗中帧属性的分布的理解一致
        int i = WINDOW_SIZE - frameInd;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - frameInd]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();


        //This is pose of left camera!!!!
        Vector3d P_r = P + R * estimator.tic[0];
        Quaterniond R_r = Quaterniond(R * estimator.ric[0]);
        R_r.normalize();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());
        vkf.pose_cam.position.x = P_r.x();
        vkf.pose_cam.position.y = P_r.y();
        vkf.pose_cam.position.z = P_r.z();
        vkf.pose_cam.orientation.x = R_r.x();
        vkf.pose_cam.orientation.y = R_r.y();
        vkf.pose_cam.orientation.z = R_r.z();
        vkf.pose_cam.orientation.w = R_r.w();

        vkf.camera_extrisinc.position.x = estimator.tic[0].x();
        vkf.camera_extrisinc.position.y = estimator.tic[0].y();
        vkf.camera_extrisinc.position.z = estimator.tic[0].z();

        Quaterniond ric = Quaterniond(estimator.ric[0]);
        ric.normalize();

        vkf.camera_extrisinc.orientation.x = ric.x();
        vkf.camera_extrisinc.orientation.y = ric.y();
        vkf.camera_extrisinc.orientation.z = ric.z();
        vkf.camera_extrisinc.orientation.w = ric.w();

        vkf.pose_drone = odometry.pose.pose;
        
        vkf.header.stamp = odometry.header.stamp;


        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - frameInd]);
        point_cloud.header.frame_id = "world";


///////////// {


if (USE_NEW){
    //estimator.featureTracker->m_camera;

#if 1
    //vins::FlattenImages images;
    vins::FlattenImages images_gray;
            sensor_msgs::Image image_center;
            //sensor_msgs::ImageConst;

    pair<double, pair<CvImages, CvImages>> time_img;
            time_img = estimator.windowed_imgs[WINDOW_SIZE - frameInd];
            double time_check = time_img.first;
            pair<CvImages,CvImages>  imgCenLR = time_img.second;
            CvImages imgCen = imgCenLR.first;
            CvImages imgCenR = imgCenLR.second;
            assert(time_check == estimator.Headers[WINDOW_SIZE - frameInd]);
    //fisheye_handler.setup_extrinsic(images_gray, estimator);

            //cv::Mat ComputeDispartiyMap(imgCen[0], imgCenR[0]);

    // img1_msg->header.stamp.toSec();

    /*
    estimator.fisheye_imgs_stampBuf;
    estimator.fisheye_imgs_upBuf; //.push(fisheye_imgs_up);
    estimator.fisheye_imgs_downBuf; //.push(fisheye_imgs_down);
    estimator.fisheye_imgs_stampBuf; //.push(t);
*/

#if 0
    Eigen::Quaterniond t_left = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
    static Eigen::Quaterniond t_down;
    if (!USE_NEW) {
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));
    }
    else
    {
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d(1, 0, 0)));
    }
    std::vector<Eigen::Quaterniond> t_dirs;
    t_dirs.push_back(Eigen::Quaterniond::Identity());
    t_dirs.push_back(t_left);
    for (unsigned int i = 0; i < 3; i ++) {
        t_dirs.push_back(t_dirs.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));
    }

    for (unsigned int i = 0; i < 4; i ++) {
        images_gray.extrinsic_up_cams.push_back(
                pose_from_PQ(estimator.tic[0], Eigen::Quaterniond(estimator.ric[0])*t_dirs[i])
        );
        //images.extrinsic_down_cams.push_back(
         //       pose_from_PQ(estimator.tic[1], Eigen::Quaterniond(estimator.ric[1])*t_down*t_dirs[i])
       // );
    }
#endif
    //images_gray.extrinsic_up_cams;

#if 0
    //images.header.stamp = stamp;
    images_gray.header.stamp = point_cloud.header.stamp;  /// odometry.header.stamp;
            image_center.header.stamp = point_cloud.header.stamp;
    //static int count = 0;
    //count ++;

   // CvCudaImages fisheye_up_imgs_cuda, fisheye_down_imgs_cuda;
   // CvCudaImages fisheye_up_imgs_cuda_gray, fisheye_down_imgs_cuda_gray;
#if 0
            if (USE_GPU) {
        fisheye_up_imgs.getGpuMatVector(fisheye_up_imgs_cuda);
        fisheye_down_imgs.getGpuMatVector(fisheye_down_imgs_cuda);
        fisheye_up_imgs_gray.getGpuMatVector(fisheye_up_imgs_cuda_gray);
        fisheye_down_imgs_gray.getGpuMatVector(fisheye_down_imgs_cuda_gray);
        // std::cout << "fisheye_up_imgs_cuda_gray size: " << fisheye_up_imgs_cuda_gray.size() << std::endl;

    }
#endif

             size_t _size = imgCen.size(); //fisheye_up_imgs_cuda.size();
            //int _size = imgCen.size();
//image_center.encoding = "mono8";
            cv_bridge::CvImage outImg_gray_center;
            outImg_gray_center.encoding = "mono8";
            outImg_gray_center.header = image_center.header;
            for (unsigned int i = 0; i < _size; i++) {
                //cv_bridge::CvImage outImg;
                cv_bridge::CvImage outImg_gray;

                //outImg.encoding = "8UC3";
                outImg_gray.encoding = "mono8";
                //outImg.header = images_gray.header;
                outImg_gray.header = images_gray.header;
                //TicToc to;
                outImg_gray.image = imgCen[i];

                images_gray.up_cams.push_back(*outImg_gray.toImageMsg());
                images_gray.down_cams.push_back(*outImg_gray.toImageMsg());
            }

            pub_flatten_images.publish(images_gray);
#else
cv::Mat imgLR;
            cv::hconcat(imgCen[0], imgCenR[0], imgLR);

            cv_bridge::CvImage cvi;

               //cvi.header.stamp = ros::Time(time_check);
            cvi.header.stamp =  ros::Time(estimator.Headers[WINDOW_SIZE - frameInd]);
               cvi.header.frame_id = "image";

#if 0
            cvi.encoding = "mono8";
#if 0
               cvi.image = imgCen[0];
#else
            cvi.image = imgCen[0]; //imgLR;
#endif
#else
        cvi.encoding = "rgb8";
        cv::Mat imgColor;
        cvtColor(imgCen[0], imgColor, CV_GRAY2RGB);
        cvi.image = imgColor;
#endif
               sensor_msgs::Image im;
               cvi.toImageMsg(im);
            pub_center_images.publish(im);

           // pub_center_images.publish(images_gray);
#endif

#else

cv::Mat imgCen = estimator.windowed_imgs[WINDOW_SIZE - frameInd];


#endif
//////////// }

        Matrix3d ric0 = estimator.ric[0];
        Matrix3d ric1 = estimator.ric[1];
        Vector3d tic0 = estimator.tic[0];
        Vector3d tic1 = estimator.tic[1];
        Matrix4d t0, t1, t01;
        t0.setIdentity();
        t1.setIdentity();
        t0.topLeftCorner<3,3>() = ric0;
        t1.topLeftCorner<3,3>() = ric1;
        t0.topRightCorner<3,1>() = tic0;
        t1.topRightCorner<3,1>() = tic1;
        t01 = t1.inverse() * t0;
        Matrix3d intrL, intrR;
        intrL.setIdentity();
        //intrR.setIdentity();
        intrL(0,0) = estimator.featureTracker->focal_x;
        intrL(1,1) = estimator.featureTracker->focal_y;
        intrL(0,2) = estimator.featureTracker->center_x;
        intrL(1,2) = estimator.featureTracker->center_y;
        intrR = intrL;
#if 0
        std::cout<<"ric0:\n"<<ric0<<std::endl;
        std::cout<<"tic0:\n"<<tic0<<std::endl;
        std::cout<<"t0:\n"<<t0<<std::endl;
        std::cout<<"ric1:\n"<<ric1<<std::endl;
        std::cout<<"tic1:\n"<<tic1<<std::endl;
        std::cout<<"t1:\n"<<t1<<std::endl;
        std::cout<<"t01:\n"<<t01<<std::endl;
#endif
// estimator

        //cv::Mat disparityMap = ComputeDispartiyMap(imgCen[0], imgCenR[0]);


//        cv::Mat & left, cv::Mat & right, bool &intr_assigned, Matrix3d &intrL, Matrix3d &intrR, Matrix3d &Rot, Vector3d &Trans,
//                cv::Mat &K1,cv::Mat & K2,cv::Mat & R,cv::Mat & T, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &Q,
//                cv::Mat &_map11, cv::Mat &_map12, cv::Mat &_map21, cv::Mat &_map22
#if 0
cv::Mat disparityMap(imgCen[0].size(), CV_8U);
        if (!estimator.intr_assigned) {
            bool assign = ComputeDispartiyMap(true, disparityMap, imgCen[0], imgCenR[0], estimator.intr_assigned, intrL,
                                              intrR, t01.topLeftCorner<3, 3>(), t01.topRightCorner<3, 1>(),
                                              estimator.K1_s, estimator.K2_s, estimator.R_s, estimator.T_s,
                                              estimator.R1_s, estimator.R2_s, estimator.P1_s, estimator.P2_s,
                                              estimator.Q_s,
                                              estimator.map11_s, estimator.map12_s, estimator.map21_s,
                                              estimator.map22_s);
            estimator.intr_assigned = assign;
        }
        else {

        }
#else
        //cv::Mat disparityMap = ComputeDispartiyMap(imgCen[0], imgCenR[0], estimator);

        cv::Mat depthMap = ComputeDispartiyMap(imgCen[0], imgCenR[0], estimator);

        std::string filename = "/home/roger/vins-fisheye2/src/VINS-Fisheye/data/depth.txt";

#if 0
        writeMatToFile(depthMap, filename);
#endif

//cv::imshow("depth", depthMap);
//cv::waitKey(2);


#if 0
        cv_bridge::CvImage depth_msg;
        depth_msg = cv_bridge::CvImage(std_msgs::Header(), CV_32FC1, depthMap).toImageMsg();
        depth_msg->header.seq = frame;
        depth_msg->header.stamp = ros::Time::now();
        depth_msg->header.frame_id = "camera_depth_frame";





        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.seq = depth_id_;
        //depth_id_++;



          img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO16, depthMap);
        sensor_msgs::Image img_msg;
        img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
        pub_depth_.publish(img_msg);
#endif

        cv_bridge::CvImage img_bridge;

#if 0
        img_bridge.header.stamp =  ros::Time(estimator.Headers[WINDOW_SIZE - frameInd]);
        img_bridge.header.frame_id = "image";
        img_bridge.encoding = "mono8"; //"16UC1"; //"32FC1";
        img_bridge.image = depthMap;
#else
        std_msgs::Header header;
        header.stamp =  ros::Time(estimator.Headers[WINDOW_SIZE - frameInd]); //ros::Time::now();
        header.frame_id = "image";
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_16UC1, depthMap);
        //img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, depthMap);
        //img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO16, depthMap);
#endif
        sensor_msgs::Image img_msg;
        img_bridge.toImageMsg(img_msg);
        pub_center_depth.publish(img_msg);




#endif
}

    //}

printf("entering bug...\n");
int validKeyFrameFeatNum = 0;
        for (auto &_it : estimator.f_manager.feature)
        {
            auto & it_per_id = _it.second;
            int frame_size = it_per_id.feature_per_frame.size();

           // estimator.featureTracker->setPrediction2(predictPts, predictPts1);

           bool isFrontImage = estimator.featureTracker->setPrediction2(it_per_id.feature_per_frame[0].point, it_per_id.feature_per_frame[0].point);
            isFrontImage = true;
           // if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag < 2 && isFrontImage)
            int imu_jj1 =  WINDOW_SIZE - frameInd - it_per_id.start_frame;
            bool isFrontImageCur;
#if 0
            if ((imu_jj1 >= 0) && (it_per_id.feature_per_frame.size() >= imu_jj1 + 1)) {
                isFrontImageCur = estimator.featureTracker->setPrediction2(it_per_id.feature_per_frame[imu_jj1].point,
                                                                           it_per_id.feature_per_frame[imu_jj1].point);
            }
#endif
            isFrontImageCur = true;
           if(it_per_id.start_frame < WINDOW_SIZE - frameInd && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - frameInd && it_per_id.solve_flag == 1 && isFrontImage && isFrontImageCur)
            {



                validKeyFrameFeatNum++;
                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[it_per_id.main_cam] * pts_i + estimator.tic[it_per_id.main_cam])
                                      + estimator.Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                vkf.feature_points_3d.push_back(p);

                // int imu_j = frame_size - 2;
                int imu_j =  WINDOW_SIZE - frameInd - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.z());


                //estimator.featureTracker->side_size_single
#if 1
                Eigen::Vector2d imgPoint, fisheye_uv;
                /// 找到与3d球坐标对应的2d包含畸变的鱼眼图像2d坐标
                //cam->spaceToPlane(pts_cam, imgPoint);
                fisheye_uv = estimator.featureTracker->setPrediction3(it_per_id.feature_per_frame[imu_j].point, it_per_id.feature_per_frame[imu_j].point);
               //p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
               //p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());

               p_2d.values.push_back(fisheye_uv.x());
               p_2d.values.push_back(fisheye_uv.y());
#else
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
#endif
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);

                geometry_msgs::Point32 fp2d_uv;
                geometry_msgs::Point32 fp2d_norm;
                fp2d_uv.x = it_per_id.feature_per_frame[imu_j].uv.x();
                fp2d_uv.y = it_per_id.feature_per_frame[imu_j].uv.y();
                fp2d_uv.z = 0;

                fp2d_norm.x = it_per_id.feature_per_frame[imu_j].point.x();
                fp2d_norm.y = it_per_id.feature_per_frame[imu_j].point.y();
                //fp2d_norm.z = 0;
                fp2d_norm.z = it_per_id.feature_per_frame[imu_j].point.z();

                vkf.feature_points_id.push_back(it_per_id.feature_id);
                vkf.feature_points_2d_uv.push_back(fp2d_uv);
                vkf.feature_points_2d_norm.push_back(fp2d_norm);
                vkf.feature_points_flag.push_back(it_per_id.solve_flag);
            }

        }
        printf("!!!!!! validKeyFrameFeatNum: %d \n",validKeyFrameFeatNum);
        pub_keyframe_point.publish(point_cloud);
        pub_viokeyframe.publish(vkf);
        printf("exiting bug...\n");
    }
    // printf("exiting bug...\n");

}
visualization_msgs::Marker marg_lines_cloud;  // 全局变量用来保存所有的线段
std::list<visualization_msgs::Marker> marg_lines_cloud_last10frame;
void pubLinesCloud(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r, vector<Eigen::Quaterniond> cam_faces)
{
    visualization_msgs::Marker lines;
    lines.header = header;
    lines.header.frame_id = "world";
    lines.ns = "lines";
    lines.type = visualization_msgs::Marker::LINE_LIST;
    lines.action = visualization_msgs::Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    lines.id = 0; //key_poses_id++;
    lines.scale.x = 0.03;
    lines.scale.y = 0.03;
    lines.scale.z = 0.03;
    lines.color.b = 1.0;
    lines.color.a = 1.0;
printf("line features to pub: %d \n", estimator.f_manager.linefeature.size());
    for (auto &it_per_id : estimator.f_manager.linefeature)
    {
        int used_num;
        used_num = it_per_id.linefeature_per_frame.size();
//
//        if (!(used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2))
//            continue;
//        //if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.is_triangulation == false)
            continue;

        // std::cout<< "used num: " <<used_num<<" line id: "<<it_per_id.feature_id<<std::endl;



        ///////////////////////////////////////
        double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);
        Eigen::Quaterniond t0x = cam_faces[face_id];
        //////////////////////////////////////








        int imu_i = it_per_id.start_frame;

        Vector3d pc, nc, vc;
        // pc = it_per_id.line_plucker.head(3);
        // nc = pc.cross(vc);
        nc = it_per_id.line_plucker.head(3);
        vc = it_per_id.line_plucker.tail(3);
        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs = it_per_id.linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs(0), obs(1), 1.0);
        Vector3d p21 = Vector3d(obs(2), obs(3), 1.0);
        Vector2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vector3d p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vector3d p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vector3d cam = Vector3d( 0, 0, 0 );

        Vector4d pi1 = pi_from_ppp(cam, p11, p12);
        Vector4d pi2 = pi_from_ppp(cam, p21, p22);

        Vector4d e1 = Lc * pi1;
        Vector4d e2 = Lc * pi2;
        e1 = e1/e1(3);
        e2 = e2/e2(3);

//
//        if(e1.norm() > 10 || e2.norm() > 10 || e1.norm() < 0.00001 || e2.norm() < 0.00001)
//            continue;

        //std::cout <<"visual: "<< it_per_id.feature_id <<" " << it_per_id.line_plucker <<"\n\n";
        Vector3d pts_1(e1(0),e1(1),e1(2));
        Vector3d pts_2(e2(0),e2(1),e2(2));

#if 0
        Vector3d w_pts_1 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_1 + estimator.tic[0])
                           + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
        Vector3d w_pts_2 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_2 + estimator.tic[0])
                           + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
#else
        Vector3d w_pts_1 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * t0x.toRotationMatrix() * pts_1 + estimator.tic[0])
                           + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
        Vector3d w_pts_2 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * t0x.toRotationMatrix() * pts_2 + estimator.tic[0])
                           + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
#endif

//std::cout<<"loop_correct_r: \n"<<loop_correct_r<<std::endl;
//        std::cout<<"++++++++++ start pt: "<<w_pts_1.transpose()<<", end pt: "<<w_pts_2.transpose()<<std::endl;
/*
        Vector3d diff_1 = it_per_id.ptw1 - w_pts_1;
        Vector3d diff_2 = it_per_id.ptw2 - w_pts_2;
        if(diff_1.norm() > 1 || diff_2.norm() > 1)
        {
            std::cout <<"visual: "<<it_per_id.removed_cnt<<" "<<it_per_id.all_obs_cnt<<" " << it_per_id.feature_id <<"\n";// << it_per_id.line_plucker <<"\n\n" << it_per_id.line_plk_init <<"\n\n";
            std::cout << it_per_id.Rj_ <<"\n" << it_per_id.tj_ <<"\n\n";
            std::cout << estimator.Rs[imu_i] <<"\n" << estimator.Ps[imu_i] <<"\n\n";
            std::cout << obs <<"\n\n" << it_per_id.obs_j<<"\n\n";

        }

        w_pts_1 = it_per_id.ptw1;
        w_pts_2 = it_per_id.ptw2;
*/
/*
        Vector3d w_pts_1 =  estimator.Rs[imu_i] * (estimator.ric[0] * pts_1 + estimator.tic[0])
                           + estimator.Ps[imu_i];
        Vector3d w_pts_2 = estimator.Rs[imu_i] * (estimator.ric[0] * pts_2 + estimator.tic[0])
                           + estimator.Ps[imu_i];

        Vector3d d = w_pts_1 - w_pts_2;
        if(d.norm() > 4.0 || d.norm() < 2.0)
            continue;
*/
        geometry_msgs::Point p;
        p.x = w_pts_1(0);
        p.y = w_pts_1(1);
        p.z = w_pts_1(2);
        lines.points.push_back(p);
        p.x = w_pts_2(0);
        p.y = w_pts_2(1);
        p.z = w_pts_2(2);
        lines.points.push_back(p);

    }
    std::cout<<" viewer lines.size: " <<lines.points.size() << std::endl;
    pub_lines.publish(lines);


//    visualization_msgs::Marker marg_lines_cloud_oneframe; // 最近一段时间的
//    marg_lines_cloud_oneframe.header = header;
//    marg_lines_cloud_oneframe.header.frame_id = "world";
//    marg_lines_cloud_oneframe.ns = "lines";
//    marg_lines_cloud_oneframe.type = visualization_msgs::Marker::LINE_LIST;
//    marg_lines_cloud_oneframe.action = visualization_msgs::Marker::ADD;
//    marg_lines_cloud_oneframe.pose.orientation.w = 1.0;
//    marg_lines_cloud_oneframe.lifetime = ros::Duration();
//
//    //marg_lines_cloud.id = 0; //key_poses_id++;
//    marg_lines_cloud_oneframe.scale.x = 0.05;
//    marg_lines_cloud_oneframe.scale.y = 0.05;
//    marg_lines_cloud_oneframe.scale.z = 0.05;
//    marg_lines_cloud_oneframe.color.g = 1.0;
//    marg_lines_cloud_oneframe.color.a = 1.0;

//////////////////////////////////////////////
    // all marglization line
    marg_lines_cloud.header = header;
    marg_lines_cloud.header.frame_id = "world";
    marg_lines_cloud.ns = "lines";
    marg_lines_cloud.type = visualization_msgs::Marker::LINE_LIST;
    marg_lines_cloud.action = visualization_msgs::Marker::ADD;
    marg_lines_cloud.pose.orientation.w = 1.0;
    marg_lines_cloud.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    //marg_lines_cloud.id = 0; //key_poses_id++;
    marg_lines_cloud.scale.x = 0.05;
    marg_lines_cloud.scale.y = 0.05;
    marg_lines_cloud.scale.z = 0.05;
    marg_lines_cloud.color.r = 1.0;
    marg_lines_cloud.color.a = 1.0;
    for (auto &it_per_id : estimator.f_manager.linefeature)
    {
//        int used_num;
//        used_num = it_per_id.linefeature_per_frame.size();
//        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
//            continue;
//        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
//        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.linefeature_per_frame.size() <= 2
            && it_per_id.is_triangulation == true )
        {
            int imu_i = it_per_id.start_frame;





            /////////////////////////////////
            double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
            int face_id = int(face_id1);
            Eigen::Quaterniond t0x = cam_faces[face_id];
            Eigen::Matrix3d R0x = t0x.toRotationMatrix();
            Eigen::Matrix4d T0x;  // , Tic, Tix;
            T0x.setIdentity();
            //Tic.setIdentity();
            T0x.topLeftCorner<3,3>() = R0x;

            //////////////////////////////////





            Vector3d pc, nc, vc;
            // pc = it_per_id.line_plucker.head(3);
            // nc = pc.cross(vc);
            nc = it_per_id.line_plucker.head(3);
            vc = it_per_id.line_plucker.tail(3);
            Matrix4d Lc;
            Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

            Vector4d obs = it_per_id.linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
            Vector3d p11 = Vector3d(obs(0), obs(1), 1.0);
            Vector3d p21 = Vector3d(obs(2), obs(3), 1.0);
            Vector2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
            ln = ln / ln.norm();

            Vector3d p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
            Vector3d p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
            Vector3d cam = Vector3d( 0, 0, 0 );

            Vector4d pi1 = pi_from_ppp(cam, p11, p12);
            Vector4d pi2 = pi_from_ppp(cam, p21, p22);

            Vector4d e1 = Lc * pi1;
            Vector4d e2 = Lc * pi2;
            e1 = e1/e1(3);
            e2 = e2/e2(3);

//            if(e1.norm() > 10 || e2.norm() > 10 || e1.norm() < 0.00001 || e2.norm() < 0.00001)
//                continue;
//
            double length = (e1-e2).norm();
            if(length > 10) continue;

            //std::cout << e1 <<"\n\n";
            Vector3d pts_1(e1(0),e1(1),e1(2));
            Vector3d pts_2(e2(0),e2(1),e2(2));

#if 0
            Vector3d w_pts_1 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_1 + estimator.tic[0])
                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
            Vector3d w_pts_2 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_2 + estimator.tic[0])
                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
#else
//            Vector3d w_pts_1 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * (t0x.toRotationMatrix() * pts_1 )+ estimator.tic[0])
//                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
//            Vector3d w_pts_2 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * (t0x.toRotationMatrix() * pts_2 )+ estimator.tic[0])
//                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;

            Vector3d w_pts_1 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * (T0x.topLeftCorner<3,3>() * pts_1 + T0x.topRightCorner<3,1>())+ estimator.tic[0])
                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;
            Vector3d w_pts_2 = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * (T0x.topLeftCorner<3,3>() * pts_2 + T0x.topRightCorner<3,1>())+ estimator.tic[0])
                               + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;

#endif
            //w_pts_1 = it_per_id.ptw1;
            //w_pts_2 = it_per_id.ptw2;

            geometry_msgs::Point p;
            p.x = w_pts_1(0);
            p.y = w_pts_1(1);
            p.z = w_pts_1(2);
            marg_lines_cloud.points.push_back(p);
//            marg_lines_cloud_oneframe.points.push_back(p);
            p.x = w_pts_2(0);
            p.y = w_pts_2(1);
            p.z = w_pts_2(2);
            marg_lines_cloud.points.push_back(p);
//            marg_lines_cloud_oneframe.points.push_back(p);
        }
    }
//    if(marg_lines_cloud_oneframe.points.size() > 0)
//        marg_lines_cloud_last10frame.push_back(marg_lines_cloud_oneframe);
//
//    if(marg_lines_cloud_last10frame.size() > 50)
//        marg_lines_cloud_last10frame.pop_front();
//
//    marg_lines_cloud.points.clear();
//    list<visualization_msgs::Marker>::iterator itor;
//    itor = marg_lines_cloud_last10frame.begin();
//    while(itor != marg_lines_cloud_last10frame.end())
//    {
//        for (int i = 0; i < itor->points.size(); ++i) {
//            marg_lines_cloud.points.push_back(itor->points.at(i));
//        }
//        itor++;
//    }

//    ofstream foutC("/home/hyj/catkin_ws/src/VINS-Mono/config/euroc/landmark.txt");
//    for (int i = 0; i < marg_lines_cloud.points.size();) {
//
//        geometry_msgs::Point pt1 = marg_lines_cloud.points.at(i);
//        geometry_msgs::Point pt2 = marg_lines_cloud.points.at(i+1);
//        i = i + 2;
//        foutC << pt1.x << " "
//              << pt1.y << " "
//              << pt1.z << " "
//              << pt2.x << " "
//              << pt2.y << " "
//              << pt2.z << "\n";
//    }
//    foutC.close();
    pub_marg_lines.publish(marg_lines_cloud);

}
