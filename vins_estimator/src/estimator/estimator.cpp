/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"
#include "../featureTracker/fisheye_undist.hpp"
#include "../depth_generation/depth_camera_manager.h"
#include "../featureTracker/feature_tracker_fisheye.hpp"
#include "../featureTracker/linefeature_tracker.h"
//#include "../featureTracker/feature_tracker_pinhole.hpp"

Eigen::Vector3d relocalize_t{Eigen::Vector3d(0, 0, 0)};
Eigen::Matrix3d relocalize_r{Eigen::Matrix3d::Identity()};
Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    // sum_t_feature = 0.0;
    // begin_time_count = 10;
    initFirstPoseFlag = false;

    failure_occur = 0;

    last_pose.setZero();
}

void Estimator::setParameter()
{
     if (FISHEYE) {
        if (USE_GPU) {
            //featureTracker = new FeatureTracker::FisheyeFeatureTrackerCuda(this);
        } else {
            featureTracker = new FeatureTracker::FisheyeFeatureTrackerOpenMP(this);
            // lineFeatureTracker = new LineFeatureTracker(this);
        }
    } else {
        //featureTracker = new FeatureTracker::PinholeFeatureTrackerCuda(this);
    }

    f_manager.ft = featureTracker;

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    //if(USE_LINE){
    /// roger line
    lineProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    baseline_ = BASE_LINE;
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    
    featureTracker->readIntrinsicParameter(CAM_NAMES);

    processThread   = std::thread(&Estimator::processMeasurements, this);
    if (FISHEYE && ENABLE_DEPTH) {
        depthThread   = std::thread(&Estimator::processDepthGeneration, this);
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    static int img_track_count = 0;
    static double sum_time = 0;
    inputImageCnt++;
    FeatureFrame featureFrame;
    TicToc featureTrackerTime;
cv::Mat im = _img.clone();
cv::Mat im_r;
if (!_img1.empty()) {
    im_r = _img1.clone();
}
else
{
    im_r = im;
}
//_img.copyTo(im);
PointLineFeature pointLineFeature;
pointLineFeature = featureTracker->trackImage(t, _img, _img1);
featureFrame = pointLineFeature.featureFrame;
    double dt = featureTrackerTime.toc();
    sum_time += dt;
    img_track_count ++;

    if(inputImageCnt % 2 == 0)
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        windowed_images.push(make_pair(t, make_pair(im,im_r)));
        mBuf.unlock();
    }
}


bool Estimator::is_next_odometry_frame() {
    return (inputImageCnt % 2 == 1);
}


void Estimator::inputFisheyeImage(double t, const CvImages & fisheye_imgs_up, 
        const CvImages & fisheye_imgs_down, const cv::Mat  up_raw, const cv::Mat  down_raw)
{
    static int img_track_count = 0;
    static double sum_time = 0;
    inputImageCnt++;

    LineFeatureFrame lineFeatureFrame;
    FeatureFrame featureFrame;
    TicToc featureTrackerTime;
/// 读取fisheye图像并做tracking, 并获得点特征，线特征
    //featureTracker->img_orig_up = up_raw;
    //featureTracker->img_orig_down = down_raw;
    featureTracker->set_orig_imgs(up_raw, down_raw);
#if 0
    featureFrame = featureTracker->trackImage(t, fisheye_imgs_up, fisheye_imgs_down, lineFeatureFrame);
#else
    PointLineFeature pointLineFeature;
    pointLineFeature = featureTracker->trackImage(t, fisheye_imgs_up, fisheye_imgs_down);
    featureFrame = pointLineFeature.featureFrame;
   // frame_cnt++;
    // featureTracker->readImage(ptr->image.rowRange(0 , ROW));
    lineFeatureFrame = pointLineFeature.lineFeatureFrame;
#endif
    CvImages im, im_r;
    for (int i = 0; i < fisheye_imgs_up.size(); i++){
        cv::Mat imm  = fisheye_imgs_up[i].clone();
        cv::Mat imm_r = fisheye_imgs_down[i].clone();
        im.push_back(imm);
        im_r.push_back(imm_r);
    }
    if(inputImageCnt % 2 == 0)
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        lineFeatureBuf.push(make_pair(t, lineFeatureFrame));
        windowed_images.push(make_pair(t, make_pair(im,im_r)));
        if (FISHEYE && ENABLE_DEPTH) {
            fisheye_imgs_upBuf.push(fisheye_imgs_up);
            fisheye_imgs_downBuf.push(fisheye_imgs_down);
            fisheye_imgs_stampBuf.push(t);
        }
        mBuf.unlock();
    }

    double dt = featureTrackerTime.toc();

    if (inputImageCnt > 100) {
        sum_time += dt;
        img_track_count ++;
    }

    if(ENABLE_PERF_OUTPUT) {
        printf("featureTracker time: AVG %f NOW %f inputImageCnt %d Bufsize %ld imgs buf Size %ld\n", 
            sum_time/img_track_count, dt, inputImageCnt, featureBuf.size(), fisheye_imgs_upBuf_cuda.size());
    }
   
}

//void Estimator::inputFisheyeImage(double t, const CvCudaImages & fisheye_imgs_up_cuda,
//        const CvCudaImages & fisheye_imgs_down_cuda, bool is_blank_init)
//{
//    static int img_track_count = 0;
//    static double sum_time = 0;
//    if (!is_blank_init) {
//        inputImageCnt++;
//    }
//
//    FeatureFrame featureFrame;
//    TicToc featureTrackerTime;
//
//    if (is_blank_init) {
//        featureFrame = ((FeatureTracker::FisheyeFeatureTrackerCuda*)featureTracker)->
//            trackImage_blank_init(t, fisheye_imgs_up_cuda, fisheye_imgs_down_cuda);
//            return;
//    } else {
//        featureFrame = featureTracker->trackImage(t, fisheye_imgs_up_cuda, fisheye_imgs_down_cuda);
//    }
//
//    if(inputImageCnt % 2 == 0)
//    {
//        mBuf.lock();
//        featureBuf.push(make_pair(t, featureFrame));
//        if (FISHEYE && ENABLE_DEPTH) {
//            fisheye_imgs_upBuf_cuda.push(fisheye_imgs_up_cuda);
//            fisheye_imgs_downBuf_cuda.push(fisheye_imgs_down_cuda);
//            fisheye_imgs_stampBuf.push(t);
//        }
//        mBuf.unlock();
//    }
//
//    double dt = featureTrackerTime.toc();
//
//    if (inputImageCnt > 100) {
//        sum_time += dt;
//        img_track_count ++;
//    }
//
//    if(ENABLE_PERF_OUTPUT) {
//        printf("featureTracker time: AVG %f NOW %f inputImageCnt %d Bufsize %ld imgs buf Size %ld\n",
//            sum_time/img_track_count, dt, inputImageCnt, featureBuf.size(), fisheye_imgs_upBuf_cuda.size());
//    }
//
//}

double base = 0;

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    double dt_device = t - ros::Time::now().toSec();
    mBuf.lock();

    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));

    if (fast_prop_inited) {
        double dt = t - latest_time;
        if (WARN_IMU_DURATION && (dt > (1.5/IMU_FREQ) || dt < (0.5/IMU_FREQ))) {
            ROS_WARN("[inputIMU] IMU sample duration not stable %4.2fms. Check your IMU and system performance", dt*1000);
        }

        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);

        // static int count = 0;
        // if (count++ % (int)(2*IMU_FREQ/IMAGE_FREQ) == 0) {
        //     double imu_propagate_dt = t - (Headers[frame_count] + td);
        //     printf("[inputIMU] IMU Propagate dt %4.1f ms Device dt %3.1fms", imu_propagate_dt*1000, dt_device*1000);
        // }
    }

    mBuf.unlock();

}

void Estimator::inputFeature(double t, const FeatureFrame &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    double t_ss = 0;
    double t_s = 0;
    double t_e = 0;
    if(t1 <= accBuf.back().first)
    {
        t_ss = accBuf.front().first;

        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }

        t_s = accBuf.front().first;
        while (accBuf.front().first < t1)
        {
            t_e = accBuf.front().first;
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }

    if (fabs(t_s - t0) > 0.01 || fabs(t_e - t1) > 0.01) {
        ROS_WARN("IMU wrong sampling dt1 %f dts0 %fms dts %f dte %f\n", t1 - t0, t_ss - t0, t_s - t0, t_e - t0);
    }


    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processDepthGeneration() {
    if (!FISHEYE) {
        ROS_ERROR("Depth generation is only vaild for dual fisheye now");
        return;
    } else {
        std::cout << "Launch depth generation thread" << std::endl;
    }

    std::vector<cv::cuda::GpuMat> fisheye_imgs_up_cuda, fisheye_imgs_down_cuda;
    //std::vector<cv::Mat> fisheye_imgs_up_cuda, fisheye_imgs_down_cuda;
    std::vector<cv::Mat> fisheye_imgs_up, fisheye_imgs_down;

    while(ros::ok()) {
        if (!fisheye_imgs_upBuf.empty() || !fisheye_imgs_upBuf_cuda.empty()) {
            double t = fisheye_imgs_stampBuf.front();
            if (USE_GPU) {
                fisheye_imgs_up_cuda = fisheye_imgs_upBuf_cuda.front();
                fisheye_imgs_down_cuda = fisheye_imgs_downBuf_cuda.front();

                mBuf.lock();
                fisheye_imgs_upBuf_cuda.pop();
                fisheye_imgs_downBuf_cuda.pop();
                fisheye_imgs_stampBuf.pop();
                mBuf.unlock();
            } else {
                fisheye_imgs_up = fisheye_imgs_upBuf.front();
                fisheye_imgs_down = fisheye_imgs_downBuf.front();

                mBuf.lock();
                fisheye_imgs_upBuf.pop();
                fisheye_imgs_downBuf.pop();
                fisheye_imgs_stampBuf.pop();
                mBuf.unlock();
            }
            //Use imu propaget for depth cloud, this is for realtime peformance;
            while(!IMUAvailable(t + td)) {
                printf("Depth wait for IMU ... \n");
                std::chrono::milliseconds dura(5);
                std::this_thread::sleep_for(dura);
            }

            TicToc tic;
            if (USE_GPU) {
                depth_cam_manager->update_images_to_buf(fisheye_imgs_up_cuda, fisheye_imgs_down_cuda);
            } else {
                depth_cam_manager->update_images_to_buf(fisheye_imgs_up, fisheye_imgs_down);
            }

            if (ENABLE_PERF_OUTPUT) {
                ROS_INFO("Depth generation cost %fms", tic.toc());
            }
            
            while(odometry_buf.size() == 0) {
                //wait for odom
                std::chrono::milliseconds dura(5);
                std::this_thread::sleep_for(dura);
            }

            //1e-3 is for avoiding floating error
            //First is older than this frame
            while (odometry_buf.size() > 0 && odometry_buf.front().first < t - 1e-3 ) {
                odomBuf.lock();
                odometry_buf.pop();
                odomBuf.unlock();
            }

            if(odometry_buf.size() == 0 || fabs(odometry_buf.front().first - t) > 1e-3) {
                ROS_WARN("No suitable odometry find; skiping");
                continue;
            } else {
                if (ENABLE_PERF_OUTPUT) {
                    ROS_INFO("ODOM dt for depth %fms", (odometry_buf.front().first - t)*1000);
                }
            }

            Eigen::Vector3d _sync_last_P = odometry_buf.front().second.second;
            Eigen::Matrix3d _sync_last_R = odometry_buf.front().second.first;
            
            odomBuf.lock();
            odometry_buf.pop();
            odomBuf.unlock();
            
            depth_cam_manager->pub_depths_from_buf(ros::Time(t), this->ric[0], this->tic[0], _sync_last_R, _sync_last_P);
            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(t);

            TicToc tic_pub;
            ROS_INFO("Pub flatten images cost %fms", tic_pub.toc());

            fisheye_imgs_up.clear();
            fisheye_imgs_down.clear();
        } else {
            std::chrono::milliseconds dura(5);
            std::this_thread::sleep_for(dura);
        }
    }
}

static void calcRectifyMap(Estimator &estimator){

    cv::Mat D1,D2;
    printf("entering calcRectifyMap, estimator.intr_assigned = %d\n", estimator.intr_assigned);
    if (!estimator.intr_assigned) {

        Matrix3d ric0 = estimator.ric[0];
        Matrix3d ric1 = estimator.ric[1];
        Vector3d tic0 = estimator.tic[0];
        Vector3d tic1 = estimator.tic[1];
        Matrix4d t0, t1, t01;
        t0.setIdentity();
        t1.setIdentity();
        t0.topLeftCorner<3, 3>() = ric0;
        t1.topLeftCorner<3, 3>() = ric1;
        t0.topRightCorner<3, 1>() = tic0;
        t1.topRightCorner<3, 1>() = tic1;
        t01 = t1.inverse() * t0;
        Matrix3d Rot = t01.topLeftCorner<3, 3>();
        Vector3d Trans = t01.topRightCorner<3, 1>();
        Matrix3d intrL, intrR;
        intrL.setIdentity();
        //intrR.setIdentity();
        intrL(0, 0) = estimator.featureTracker->focal_x;
        intrL(1, 1) = estimator.featureTracker->focal_y;
        intrL(0, 2) = estimator.featureTracker->center_x;
        intrL(1, 2) = estimator.featureTracker->center_y;
        intrR = intrL;

        Eigen::VectorXd dist1, dist2;
        dist1.resize(5);
        dist2.resize(5);
        dist1.setZero();
        dist2.setZero();
        cv::eigen2cv(dist1, D1);
        cv::eigen2cv(dist2, D2);

        cv::eigen2cv(intrL, estimator.K1_s);
        cv::eigen2cv(intrR, estimator.K2_s);
        cv::eigen2cv(Rot, estimator.R_s);
        cv::eigen2cv(Trans, estimator.T_s);
//        cv::eigen2cv(dist1, D1);
//        cv::eigen2cv(dist2, D2);

//inline CvSize cvSize( int width, int height );
        cv::Size imgSize(estimator.featureTracker->cen_width, estimator.featureTracker->cen_height);
        //cv::Mat R1, R2, P1, P2, Q_;
        cv::Mat Q_;

        cv::Rect validRoi[2];
        stereoRectify(estimator.K1_s, D1, estimator.K2_s, D2, imgSize, estimator.R_s, estimator.T_s, estimator.R1_s,
                      estimator.R2_s, estimator.P1_s, estimator.P2_s, Q_,
                      cv::CALIB_ZERO_DISPARITY, 0, imgSize, &validRoi[0], &validRoi[1]);
        std::cout << Q_ << std::endl;
        cv::initUndistortRectifyMap(estimator.K1_s, D1, estimator.R1_s, estimator.P1_s, imgSize, CV_32FC1,
                                    estimator.map11_s,
                                    estimator.map12_s);
        cv::initUndistortRectifyMap(estimator.K2_s, D2, estimator.R2_s, estimator.P2_s, imgSize, CV_32FC1,
                                    estimator.map21_s,
                                    estimator.map22_s);
        Q_.convertTo(estimator.Q_s, CV_32F);
        std::cout << estimator.Q_s << std::endl;
        std::cout << estimator.K1_s << std::endl;
        estimator.intr_assigned = true;
        //cv::cv2eigen()
        estimator.baseline = Trans;
        // intr_assigned = 1; //true;
    }
}
void Estimator::processMeasurements()
{

    static int mea_track_count = 0;
    static double mea_sum_time = 0;
    while (1)
    {
        //printf("process measurments\n");
        TicToc t_process;
        pair<double, FeatureFrame > feature;
        pair<double,LineFeatureFrame > lineFeature;
        pair<double, pair<CvImages,CvImages>> curImgs;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!featureBuf.empty())
        {


            /// 这里开始接受feature_tracker模块累积起来的特征点，特征线
            feature = featureBuf.front();
            lineFeature = lineFeatureBuf.front();
            curImgs = windowed_images.front();
            curTime = feature.first + td;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... TD%f\n", td);
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if(USE_IMU) {
                getIMUInterval(prevTime, curTime, accVector, gyrVector);
                if (curTime - prevTime > 2.1/IMAGE_FREQ || accVector.size()/(curTime - prevTime ) < IMU_FREQ*0.8) {
                    ROS_WARN("Long image dt %fms or wrong IMU rate %fhz", (curTime - prevTime)*1000, accVector.size()/(curTime - prevTime));
                } 
            }

            featureBuf.pop();
            lineFeatureBuf.pop();
            windowed_images.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }

            /// 一个很重要的入口 processMeasurements
            //windowed_imgs[frame_count] = make_pair(curImgs.first, curImgs.second[0]);
            assert(feature.first == lineFeature.first);
            windowed_imgs[frame_count] = make_pair(curImgs.first, curImgs.second);
            processImage(feature.second, lineFeature.second, feature.first);

            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubIMUBias(latest_Ba, latest_Bg, header);
            //These cost 5ms, ~1/6 percent on manifold2


            {
#if 0
                Eigen::Vector3d tmp_T0 = this->Ps[0];
                Eigen::Vector3d tmp_T1 = this->Ps[1];
                Eigen::Vector3d delta_T = tmp_T0 - tmp_T1;
                // estimator.travelled_distance = estimator.travelled_distance + sqrt(delta_T(0)*delta_T(0) + delta_T(1)*delta_T(1) + delta_T(2)*delta_T(2));
                this->travelled_distance = this->travelled_distance +
                                           delta_T.norm(); //sqrt(delta_T(0)*delta_T(0) + delta_T(1)*delta_T(1) + delta_T(2)*delta_T(2));
#else
                //Eigen::Vector3d tmp_T0 = this->Ps[WINDOW_SIZE-1];
                Eigen::Vector3d tmp_T1 = this->Ps[WINDOW_SIZE];
                Eigen::Vector3d delta_T = last_pose - tmp_T1;
                last_pose = this->Ps[WINDOW_SIZE];
                this->travelled_distance = this->travelled_distance +
                                           delta_T.norm(); //sqrt(delta_T(0)*delta_T(0) + delta_T(1)*delta_T(1) + delta_T(2)*delta_T(2));
#endif

            }
            calcRectifyMap(*this);
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubLinesCloud(*this, header, relocalize_t, relocalize_r, featureTracker->cam_faces);
            pubKeyframe(*this);
            pubTF(*this, header);

            double dt = t_process.toc();
            mea_sum_time += dt;
            mea_track_count ++;

            if(ENABLE_PERF_OUTPUT) {
                ROS_INFO("process measurement time: AVG %f NOW %f\n", mea_sum_time/mea_track_count, dt );
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
   // cout << "init T0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    latest_P = Eigen::Vector3d::Zero();
    latest_V = Eigen::Vector3d::Zero();
    latest_Q = Eigen::Quaterniond::Identity();
    fast_prop_inited = false;
    initial_timestamp = 0;
    //  all_image_frame.clear();

    all_image_frame.clear();
    relocalize = false;
    retrive_data_vector.clear();
    relocalize_t = Eigen::Vector3d(0, 0, 0);
    relocalize_r = Eigen::Matrix3d::Identity();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    last_pose.setZero();
}

void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    else
    {
        printf("frame_count == 0 \n");
        std::cout<<"Rs[0]:\n"<<Rs[0]<<std::endl;
        std::cout<<"Ps[0]:\n"<<Ps[0]<<std::endl;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const FeatureFrame &image,const LineFeatureFrame &lineFeatureImage, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    /// 此时，多目tracking已经在前面完成了，可以把特征点数据整理成vins-fusion的组织方式了
    //TODO 这里是为了数据组织，把数据改成vins-fusion的组织方式
    printf("in processImage, input linefeature size: %d \n", lineFeatureImage.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, lineFeatureImage, td))
    {
        marginalization_flag = MARGIN_OLD;
        printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------> keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------> non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            /// 用手眼方程的闭式解求出相机到imu旋转的初值
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {

        base = ros::Time::now().toSec();

        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    solver_flag = NON_LINEAR;
////////////////////////////////////
/////////////////////////////////////    SolveOdometry()的内容（除了SlideWindow, 这是在外面做的）
                    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
                    if(USE_LINE) {
                        if (baseline_ > 0){
                            f_manager.triangulateLine(baseline_);
                        }
                        else {
                            f_manager.triangulateLine(Ps, tic, ric);
                        }
                        onlyLineOpt(featureTracker->cam_faces);
                        optimization(featureTracker->cam_faces);
#ifdef LINEINCAM
                        LineBAincamera();
#else
//        LineBA();
#endif
//////////////////////////////////////
                        slideWindow();
                    }
                    else {
                        optimization(featureTracker->cam_faces);
                        slideWindow();
                    }
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            ROS_INFO("Init by pose pnp...");
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            TicToc t_ic;
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (ENABLE_PERF_OUTPUT) {
                ROS_INFO("Triangulation cost %3.1fms..", t_ic.toc());
            }
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                
                solver_flag = NON_LINEAR;
#if 0
                optimization();
                slideWindow();
#else
                printf("base_line: %0.3f \n",baseline_);
                if(USE_LINE) {
                    if (baseline_ > 0){
                        f_manager.triangulateLine(baseline_);
                    }
                    else {
                        //f_manager.triangulateLine(Ps, tic, ric);
                        //f_manager.triangulateLine2(Ps, Rs, tic, ric, featureTracker->cam_faces);
                        f_manager.triangulateLine2(Ps, Rs, tic, ric, featureTracker->cam_faces);

                    }
                    onlyLineOpt(featureTracker->cam_faces);
                    optimization(featureTracker->cam_faces);
#ifdef LINEINCAM
                    LineBAincamera();
#else
//        LineBA();
#endif
                    slideWindow();
                }
                else {
                    optimization(featureTracker->cam_faces);
                    slideWindow();
                }
#endif
                // set<int> removeIndex;
                // outliersRejection(removeIndex);
                // exit(-1);

                ROS_INFO("Initialization finish!");
                printf(" ================ Initialization finish! ================\n");
            }
else{
#if 0
    if(USE_LINE){
        slideWindow();
    }
#endif
}
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization(featureTracker->cam_faces);

            if(frame_count == WINDOW_SIZE)
            {
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            /// 用匀速模型预测下一帧pose
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        if(!USE_IMU) {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        }
        else{
            /// roger @ 20220102
            /// 如果加了这句话, 弊大于利(scratch that)
            //  f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        }
        TicToc t_ic;
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        
        if(ENABLE_PERF_OUTPUT) {        
            ROS_INFO("Triangulation cost %3.1fms..", t_ic.toc());
        }
        if(USE_LINE){
            if (baseline_ > 0){
                f_manager.triangulateLine(baseline_);
            }
            else {
                // f_manager.triangulateLine(Ps, tic, ric);
                f_manager.triangulateLine2(Ps, Rs, tic, ric, featureTracker->cam_faces);
            }
            onlyLineOpt(featureTracker->cam_faces);
            optimization(featureTracker->cam_faces);
#ifdef LINEINCAM
            LineBAincamera();
#else
//        LineBA();
#endif
        }
        else {
            optimization(featureTracker->cam_faces);
        }
        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("after optimization cost %fms..", t_ic.toc());
        }
        
        set<int> removeIndex;
        outliersRejection(removeIndex);
        if (ENABLE_PERF_OUTPUT) {
            ROS_INFO("Remove %ld outlier", removeIndex.size());
        }
        
        f_manager.removeOutlier(removeIndex);
        predictPtsInNextFrame();
        
        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("solver costs: %fms", t_solve.toc());
        }

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            //exit(-1);
            cv::waitKey(-1);
            return;
        }

        slideWindow();

        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("to slideWindow costs: %fms", t_solve.toc());
        }

        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];

        odomBuf.lock();
        odometry_buf.push(make_pair( header, make_pair(last_R, last_P)));
        odomBuf.unlock();

        updateLatestStates();
        if(ENABLE_PERF_OUTPUT) {
            ROS_INFO("after updateLatestStates costs: %fms", t_solve.toc());
        }
    }  
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &_it : f_manager.feature)
    {
        auto & it_per_id = _it.second;
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        cv::Mat inliers1;
        double foc1 = 460;
        double f_thr1 = 2;
        bool pnp_succ1 = solvePnPRansac(pts_3_vector, pts_2_vector, K, D, rvec, t, true, 100, f_thr1 / foc1, 0.99, inliers1);
        std::cout<<"sfm solvePnPRansac inliers1: "<<inliers1.t()<<std::endl;
        cv::Scalar ss1 = sum(inliers1);
        //printf("solvePnPRansac inlier ratio: %d/%d \n", ss[0],inliers.size);
        printf("sfm solvePnPRansac inlier ratio: %d/%d \n", inliers1.rows, pts_2_vector.size());

        double inlier_ratio1 = (double)inliers1.rows / (double)pts_2_vector.size();
        //if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        if(!pnp_succ1 | inlier_ratio1 < 0.5)
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double(vector<Eigen::Quaterniond> cam_faces)
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    ////////////////////////////////////
#if 1

    Eigen::Matrix4d Tic;
    Tic.setIdentity();
    Tic.topLeftCorner<3, 3>() = ric[0];
    Tic.topRightCorner<3, 1>() = tic[0];

for(int i = 0; i < cam_faces.size(); i++) {
    Eigen::Quaterniond t0x = cam_faces[i];
    Eigen::Matrix3d R0x = t0x.toRotationMatrix();
    Eigen::Matrix4d T0x, Tix;
    T0x.setIdentity();

    T0x.topLeftCorner<3, 3>() = R0x;


    Tix = Tic * T0x;
    Matrix3d Rix = Tix.topLeftCorner<3, 3>();
    Quaterniond qix{Rix};
    Vector3d tix = Tix.topRightCorner<3, 1>();
#if 0
    double para_Ex_Pose2[SIZE_POSE];
    para_Ex_Pose2[0] = tix.x();
    para_Ex_Pose2[1] = tix.y();
    para_Ex_Pose2[2] = tix.z();
    //Quaterniond q{ric[i]};
    para_Ex_Pose2[3] = qix.x();
    para_Ex_Pose2[4] = qix.y();
    para_Ex_Pose2[5] = qix.z();
    para_Ex_Pose2[6] = qix.w();
#else
    para_Ex_Pose22[i][0] = tix.x();
    para_Ex_Pose22[i][1] = tix.y();
    para_Ex_Pose22[i][2] = tix.z();
    //Quaterniond q{ric[i]};
    para_Ex_Pose22[i][3] = qix.x();
    para_Ex_Pose22[i][4] = qix.y();
    para_Ex_Pose22[i][5] = qix.z();
    para_Ex_Pose22[i][6] = qix.w();


#endif
}
#endif














    auto deps = f_manager.getDepthVector();
    param_feature_id.clear();
    // printf("Solve features: %ld;", deps.size());
    for (auto & it : deps) {
        // ROS_INFO("Feature %d invdepth %f feature index %d", it.first, it.second, param_feature_id.size());
        para_Feature[param_feature_id.size()][0] = it.second;
        param_feature_id_to_index[it.first] = param_feature_id.size();
        param_feature_id.push_back(it.first);
        
    }

    if (USE_LINE) {
#ifdef LINEINCAM
        MatrixXd lineorth = f_manager.getLineOrthVectorInCamera();
#else
        MatrixXd lineorth = f_manager.getLineOrthVector(Ps, tic, ric,featureTracker->cam_faces);
#endif

        for (int i = 0; i < f_manager.getLineFeatureCount(); ++i) {
            para_LineFeature[i][0] = lineorth.row(i)[0];
            para_LineFeature[i][1] = lineorth.row(i)[1];
            para_LineFeature[i][2] = lineorth.row(i)[2];
            para_LineFeature[i][3] = lineorth.row(i)[3];
            if (i > NUM_OF_F)
                std::cerr << " 1000  1000 1000 1000 1000 \n\n";
        }
    }
    para_Td[0][0] = td;
}

void Estimator::double2vector(vector<Eigen::Quaterniond> cam_faces)
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        printf("######## y_diff: %f deg\n", y_diff *180.0/M_PI);
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    MatrixXd lineorth_vec(f_manager.getLineFeatureCount(), 4);
#if 1
    if(USE_LINE) {
    //std::cout <<"----------\n"<< Rwow1 <<"\n"<<twow1<<std::endl;
    //MatrixXd lineorth_vec(f_manager.getLineFeatureCount(), 4);
    for (int i = 0; i < f_manager.getLineFeatureCount(); ++i) {
        Vector4d orth(para_LineFeature[i][0],
                      para_LineFeature[i][1],
                      para_LineFeature[i][2],
                      para_LineFeature[i][3]);
        lineorth_vec.row(i) = orth;

    }
#ifdef LINEINCAM
    f_manager.setLineOrthInCamera(lineorth_vec);
#else
    f_manager.setLineOrth(lineorth_vec, Ps, Rs, tic, ric, featureTracker->cam_faces);
#endif
}
#endif

    std::map<int, double> deps;
    for (unsigned int i = 0; i < param_feature_id.size(); i++) {
        int _id;
        try{
            _id = param_feature_id[i];
        }
        catch(...){
            printf("i: %d, param_feature_id size: %d\n", i, param_feature_id.size());
            printf("i: %d, param_feature_id size: %d, param_feature_id[i]: %d\n", i, param_feature_id.size(), param_feature_id[i]);
            _id = param_feature_id[i];
        }
        // ROS_INFO("Id %d depth %f", i, 1/para_Feature[i][0]);
        deps[_id] = para_Feature[i][0];
    }

    f_manager.setDepth(deps);

    if(USE_IMU)
        td = para_Td[0][0];

}
void Estimator::double2vector2(vector<Eigen::Quaterniond> cam_faces)
{
    // 六自由度优化的时候，整个窗口会在空间中任意优化，这时候我们需要把第一帧在yaw,position上的增量给去掉，因为vins在这几个方向上不可观，他们优化的增量也不可信。
    // 所以这里的操作过程就相当于是 fix 第一帧的 yaw 和 postion, 使得整个轨迹不会在空间中任意飘。
    // 相机姿态需要变化考虑优化以后，把yaw量旋转回去
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);    //优化之前的0th的姿态
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // 优化以后的0th的姿态
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5]).toRotationMatrix());

    // 优化前后，yaw的变化
    double y_diff = origin_R0.x() - origin_R00.x();
    printf("######## y_diff: %f deg\n", y_diff *180.0/M_PI);
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    // 由于VI系统的（绝对位置x,y,z,以及yaw）是不可观的。而优化过程中没有固定yaw角，因此yaw会朝着使得误差函数最小的方向优化，但这不一定是正确的。
    // 所以这里把 yaw角的变化量给旋转回去。
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        // Position 也转移到yaw角优化前的 0th坐在的世界坐标下
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    // 跟yaw没关系，所以不用管优化前后yaw的变化
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }


    // 先把line旋转到相机坐标系下
    Matrix3d Rwow1 = rot_diff;
    Vector3d tw1b(para_Pose[0][0],para_Pose[0][1],para_Pose[0][2]);
    Vector3d twow1 = -Rwow1 * tw1b + origin_P0;

    //std::cout <<"----------\n"<< Rwow1 <<"\n"<<twow1<<std::endl;
    MatrixXd lineorth_vec(f_manager.getLineFeatureCount(),4);;
    for (int i = 0; i < f_manager.getLineFeatureCount(); ++i) {
        Vector4d orth(para_LineFeature[i][0],
                      para_LineFeature[i][1],
                      para_LineFeature[i][2],
                      para_LineFeature[i][3]);

        // 将line_w优化以后的角度变化yaw的变化旋转回去
        /// world body frame
        Vector6d line_w1 = orth_to_plk(orth);
        Vector6d line_wo = plk_to_pose(line_w1, Rwow1,twow1);
        /// orth is still expressed in world body frame
        orth = plk_to_orth(line_wo);

        lineorth_vec.row(i) = orth;

    }
    f_manager.setLineOrth(lineorth_vec,Ps,Rs,tic,ric, featureTracker->cam_faces);

#if 0
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
#else
#if 0
    auto deps = f_manager.getDepthVector();
    param_feature_id.clear();
    // printf("Solve features: %ld;", deps.size());
    for (auto & it : deps) {
        // ROS_INFO("Feature %d invdepth %f feature index %d", it.first, it.second, param_feature_id.size());
        para_Feature[param_feature_id.size()][0] = it.second;
        param_feature_id_to_index[it.first] = param_feature_id.size();
        param_feature_id.push_back(it.first);

    }
#else
    std::map<int, double> deps;
    for (unsigned int i = 0; i < param_feature_id.size(); i++) {
        int _id = param_feature_id[i];
        // ROS_INFO("Id %d depth %f", i, 1/para_Feature[i][0]);
        deps[_id] = para_Feature[i][0];
    }

    f_manager.setDepth(deps);
#endif
#endif
}
bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}
void  Estimator::onlyLineOpt(vector<Eigen::Quaterniond> cam_faces)
{
    //固定pose， 只优化line的参数，用来调试line的一些参数，看ba优化出来的最好line地图是啥样

#if 0
    printf("===== before onlyLineOpt ===== \n");
   f_manager.removeLineOutlier(Ps,tic,ric,featureTracker->cam_faces);
#endif

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)    // 将窗口内的 p,q 加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // p,q
        // 固定 pose
        problem.SetParameterBlockConstant(para_Pose[i]);
    }
    // for (int i = 0; i < NUM_OF_CAM; i++)         // 外参数
    for (int i = 0; i < 1; i++)         // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        // 固定 外参数
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);

    }

    if (USE_LINE) {
        for (int i = 0; i < cam_faces.size(); i++) {
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(para_Ex_Pose22[i], SIZE_POSE, local_parameterization);
            problem.SetParameterBlockConstant(para_Ex_Pose22[i]);
        }
    }

    vector2double(featureTracker->cam_faces);// 将那些保存在 vector向量里的参数 移到 double指针数组里去


    Eigen::Matrix4d Tic;  //T0x, Tic;
    //T0x.setIdentity();
    Tic.setIdentity();
    //T0x.topLeftCorner<3,3>() = R0x;
    Tic.topLeftCorner<3,3>() = ric[0];
    Tic.topRightCorner<3,1>() = tic[0];

    // 所有特征
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了

        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
        /// roger @ 20220213
        // if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
           // printf("===== after onlyLineOpt0 ===== \n");
            continue;

        ++feature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式
        /*
        std::cout << para_LineFeature[feature_index][0] <<" "
                << para_LineFeature[feature_index][1] <<" "
                << para_LineFeature[feature_index][2] <<" "
                << para_LineFeature[feature_index][3] <<"\n";
        */
        ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
        problem.AddParameterBlock( para_LineFeature[feature_index], SIZE_LINE, local_parameterization_line);  // p,q

        int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;



        double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);
        Eigen::Quaterniond t0x = cam_faces[face_id];
        Eigen::Matrix3d R0x = t0x.toRotationMatrix();
        Eigen::Matrix4d T0x, Tic, Tix;
        T0x.setIdentity();
        Tic.setIdentity();
        T0x.topLeftCorner<3,3>() = R0x;
        Tic.topLeftCorner<3,3>() = ric[0];
        Tic.topRightCorner<3,1>() = tic[0];

        Tix = Tic * T0x;
        Matrix3d Rix = Tix.topLeftCorner<3,3>();
        Quaterniond qix{Rix};
        Vector3d tix = Tix.topRightCorner<3,1>();
        double para_Ex_Pose2[SIZE_POSE];
        para_Ex_Pose2[0] = tix.x();
        para_Ex_Pose2[1] = tix.y();
        para_Ex_Pose2[2] = tix.z();
        //Quaterniond q{ric[i]};
        para_Ex_Pose2[3] = qix.x();
        para_Ex_Pose2[4] = qix.y();
        para_Ex_Pose2[5] = qix.z();
        para_Ex_Pose2[6] = qix.w();



        for (auto &it_per_frame : it_per_id.linefeature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                //continue;
            }
            Vector4d obs = it_per_frame.lineobs.head(4);                          // 在第j帧图像上的观测



            //double face_id1 = it_per_frame.lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
            //int face_id = int(face_id1);
            // Eigen::Quaterniond t0x = cam_faces[face_id];



            lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
#if 0
            problem.AddResidualBlock(f, loss_function,
                                     para_Pose[imu_j],
                                     para_Ex_Pose[0],
                                     para_LineFeature[feature_index]);
#else
//            problem.AddResidualBlock(f, loss_function,
//                                     para_Pose[imu_j],
//                                     para_Ex_Pose2,
//                                     para_LineFeature[feature_index]);
            problem.AddResidualBlock(f, loss_function,
                                     para_Pose[imu_j],
                                     para_Ex_Pose22[face_id],
                                     para_LineFeature[feature_index]);
#endif
            f_m_cnt++;
        }
    }
    // printf("===== after onlyLineOpt1 ===== \n");
    // if(feature_index < 3)
    if (feature_index < 0)
    //if(false)
    {
        printf("===== after onlyLineOpt3 ===== \n");
        return;
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.trust_region_strategy_type = ceres::DOGLEG;

    options.max_num_iterations = NUM_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve (options, &problem, & summary);
#if 1
    std::cout<<"summary:\n"<<summary.BriefReport()<<std::endl;
#endif
    //std::cout <<"!!!!!!!!!!!!!onlyLineOpt!!!!!!!!!!!!!\n";
    double2vector(featureTracker->cam_faces);
    //std::cout << summary.FullReport()<<std::endl;


   // printf("===== after onlyLineOpt2 ===== NUM_ITERATIONS: %d\n",NUM_ITERATIONS);
    f_manager.removeLineOutlier(Ps,tic,ric,featureTracker->cam_faces);



}
void  Estimator::LineBA()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)    // 将窗口内的 p,q 加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // p,q
        // 固定 pose
        //problem.SetParameterBlockConstant(para_Pose[i]);
    }
    //  fix the first camera pose in window
    // problem.SetParameterBlockConstant( para_Pose[0] );
    //problem.SetParameterBlockConstant( para_Pose[1] );
    //problem.SetParameterBlockConstant( para_Pose[2] );

    for (int i = 0; i < NUM_OF_CAM; i++)         // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        // 固定 外参数
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);

    }
    vector2double(featureTracker->cam_faces);// 将那些保存在 vector向量里的参数 移到 double指针数组里去
    //std::cout << Ps[9] <<"\n" << Ps[10]<<"\n";

    // 窗口里各帧之间的 imu factor
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)      // 由于有时候会有静止的情况出现，这时候视差一直不够，关键帧一直没有选，预积分量一直累计，可能出现时间超过10s的情况？
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);            // 预积分误差项: 误差，雅克比的计算
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }


//     所有特征
//    int f_m_cnt = 0;
//    int feature_index = -1;
//    for (auto &it_per_id : f_manager.feature)
//    {
//        it_per_id.used_num = it_per_id.feature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
//        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
//            continue;
//
//        ++feature_index;                     // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式
//
//        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
//
//        Vector3d pts_i = it_per_id.feature_per_frame[0].point;            // 图像上第一次观测到这个特征的坐标
//
//        for (auto &it_per_frame : it_per_id.feature_per_frame)
//        {
//            imu_j++;
//            if (imu_i == imu_j)
//            {
//                continue;
//            }
//            Vector3d pts_j = it_per_frame.point;                          // 在第j帧图像上的观测
//            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);     // 特征重投影误差
//            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
//            f_m_cnt++;
//        }
//    }

/////////////////////////////////////
    // Line feature
    int line_m_cnt = 0;
    int linefeature_index = -1;
    for (auto &it_per_id : f_manager.linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            continue;

        ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

        ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
        problem.AddParameterBlock( para_LineFeature[linefeature_index], SIZE_LINE, local_parameterization_line);  // p,q

        int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                //continue;
            }
            Vector4d obs = it_per_frame.lineobs.head(4);                          // 在第j帧图像上的观测
            lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
            problem.AddResidualBlock(f, loss_function,
                                     para_Pose[imu_j],
                                     para_Ex_Pose[0],
                                     para_LineFeature[linefeature_index]);
            line_m_cnt++;
        }
    }

    std::cout << "------------ linefeature_index : " << linefeature_index <<"\n";
    if(linefeature_index < 3)
    {
        // return;
    }

////////////////////////////////////////

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve (options, &problem, & summary);

    double2vector2(featureTracker->cam_faces);
    //std::cout << Ps[9] <<"\n" << Ps[10]<<"\n";
    std::cout << summary.FullReport()<<std::endl;

    f_manager.removeLineOutlier(Ps,tic,ric,featureTracker->cam_faces);


}


void  Estimator::LineBAincamera()
{
    //固定pose， 只优化line的参数，用来调试line的一些参数，看ba优化出来的最好line地图是啥样
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)    // 将窗口内的 p,q 加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // p,q
        // 固定 pose
        //problem.SetParameterBlockConstant(para_Pose[i]);
    }
    //  fix the first camera pose in window
    //problem.SetParameterBlockConstant( para_Pose[0] );
    //problem.SetParameterBlockConstant( para_Pose[1] );
    //problem.SetParameterBlockConstant( para_Pose[2] );

    for (int i = 0; i < NUM_OF_CAM; i++)         // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);

        // 固定 外参数
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);

    }
    vector2double(featureTracker->cam_faces);// 将那些保存在 vector向量里的参数 移到 double指针数组里去
    //std::cout << Ps[9] <<"\n" << Ps[10]<<"\n";
    //std::cout<<"11111111111\n";
    // 窗口里各帧之间的 imu factor
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)      // 由于有时候会有静止的情况出现，这时候视差一直不够，关键帧一直没有选，预积分量一直累计，可能出现时间超过10s的情况？
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);            // 预积分误差项: 误差，雅克比的计算
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }


    // 所有特征
    int f_m_cnt = 0;
//    int feature_index = -1;
//    for (auto &it_per_id : f_manager.feature)
//    {
//        it_per_id.used_num = it_per_id.feature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
//        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
//            continue;
//
//        ++feature_index;                     // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式
//
//        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
//
//        Vector3d pts_i = it_per_id.feature_per_frame[0].point;            // 图像上第一次观测到这个特征的坐标
//        for (auto &it_per_frame : it_per_id.feature_per_frame)
//        {
//            imu_j++;
//            if (imu_i == imu_j)
//            {
//                continue;
//            }
//            Vector3d pts_j = it_per_frame.point;                          // 在第j帧图像上的观测
//            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);     // 特征重投影误差
//            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
//            f_m_cnt++;
//        }
//    }

/////////////////////////////////////
    // Line feature
    int line_m_cnt = 0;
    int linefeature_index = -1;
    for (auto &it_per_id : f_manager.linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            continue;

        ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

        ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
        problem.AddParameterBlock( para_LineFeature[linefeature_index], SIZE_LINE, local_parameterization_line);  // p,q

        int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)
        {
            imu_j++;
            Vector4d obs = it_per_frame.lineobs.head(4);                          // 在第j帧图像上的观测
            if (imu_i == imu_j)
            {
                lineProjectionFactor_instartframe *f = new lineProjectionFactor_instartframe(obs);     // 特征重投影误差
                problem.AddResidualBlock(f, loss_function,
                                         para_LineFeature[linefeature_index]);
            } else
            {
                lineProjectionFactor_incamera *f = new lineProjectionFactor_incamera(obs);     // 特征重投影误差
                problem.AddResidualBlock(f, loss_function,
                                         para_Pose[imu_i],             // 特征都是第i帧初始化的
                                         para_Pose[imu_j],
                                         para_Ex_Pose[0],
                                         para_LineFeature[linefeature_index]);

            }
            line_m_cnt++;
        }
    }

    std::cout << "------------ linefeature_index : " << linefeature_index <<"\n";
    if(linefeature_index < 3)
    {
        //    return;
    }
////////////////////////////////////////

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve (options, &problem, & summary);

    double2vector(featureTracker->cam_faces);
    //  std::cout << summary.FullReport()<<std::endl;

    f_manager.removeLineOutlier(Ps,tic,ric,featureTracker->cam_faces);


}
//#define DebugFactor
#if 0
void Estimator::optimizationwithLine()
{
    frame_cnt_ ++;
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)    // 将窗口内的 p,q,v,ba,bg加入优化变量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // p,q
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);                // v,ba,bg
    }
    for (int i = 0; i < NUM_OF_CAM; i++)         // 外参数
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    TicToc t_whole, t_prepare;    // 统计程序运行时间
    TicToc t_solver;
    vector2double();// 将那些保存在 vector向量里的参数 移到 double指针数组里去

#ifdef DebugFactor
    /* Debug: 监视 prior factor*/
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    int marg_residual_size=0;
    std::vector<ceres::ResidualBlockId> imufactor_residual_block_ids;
    //end debud
#endif
    // 滑动窗口marg以后，上一次的prior factor
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        ceres::ResidualBlockId block_id = problem.AddResidualBlock(marginalization_factor, NULL,
                                                                   last_marginalization_parameter_blocks);
#ifdef DebugFactor
        marg_residual_size = marginalization_factor->num_residuals();  // used to debug
        residual_block_ids.push_back(block_id);           // used to debug
#endif
    }

    // 窗口里各帧之间的 imu factor
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)      // 由于有时候会有静止的情况出现，这时候视差一直不够，关键帧一直没有选，预积分量一直累计，可能出现时间超过10s的情况？
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);            // 预积分误差项: 误差，雅克比的计算
        ceres::ResidualBlockId block_id = problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
#ifdef DebugFactor
        imufactor_residual_block_ids.push_back(block_id);
#endif
    }

    // 所有特征
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            continue;

        ++feature_index;                     // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;            // 图像上第一次观测到这个特征的坐标

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;                          // 在第j帧图像上的观测
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);     // 特征重投影误差
            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            f_m_cnt++;
        }
    }
/////////////////////////////////////
    // Line feature
    int line_m_cnt = 0;
    int linefeature_index = -1;
    for (auto &it_per_id : f_manager.linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            continue;

        ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

        ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
        problem.AddParameterBlock( para_LineFeature[linefeature_index], SIZE_LINE, local_parameterization_line);  // p,q

        int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                //continue;
            }
            Vector4d obs = it_per_frame.lineobs;                          // 在第j帧图像上的观测
            lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
            problem.AddResidualBlock(f, loss_function,
                                     para_Pose[imu_j],
                                     para_Ex_Pose[0],
                                     para_LineFeature[linefeature_index]);
            line_m_cnt++;
        }
    }
    ROS_INFO("lineFactor: %d, pointFactor:%d", line_m_cnt, f_m_cnt);

    // if(line_m_cnt > 20)
    // {
    //     double scale = std::min(f_m_cnt /(2. * line_m_cnt), 10.);
    //     lineProjectionFactor::sqrt_info =  scale * FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    //     std::cout << "========== Line factor weight ========== \n" << scale  << std::endl;
    // }

////////////////////////////////////////

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;

//    if (marginalization_flag == MARGIN_OLD)
//        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
//    else
//        options.max_solver_time_in_seconds = SOLVER_TIME;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //cout << summary.BriefReport() << endl;
//    ROS_INFO("Points Lines Iterations : %d", static_cast<int>(summary.iterations.size()));
    sum_solver_time_ += t_solver.toc();
    mean_solver_time_ = sum_solver_time_/frame_cnt_;
    ROS_INFO("Points Lines solver costs: %f", mean_solver_time_);


#ifdef DebugFactor
    //----------- debug --------------------
    // marg factor
    Eigen::VectorXd err(marg_residual_size);
    err.setZero();
    ceres::Problem::EvaluateOptions opts;
    opts.residual_blocks = residual_block_ids;
    double total_cost = 0.0;
    vector<double> residuals;
    problem.Evaluate(opts,&total_cost, &residuals, nullptr, nullptr);
    for(int i=0; i< marg_residual_size; i++)
    {
        //std::cout << residuals[i]<<std::endl;
        err[i] = residuals[i];
    }
    /*
    std::ofstream file;
    file.open("/home/hyj/VINS-err-before.txt",std::ofstream::app);
    file.setf(ios::fixed, ios::floatfield);
    file.precision(5);
    file << err.squaredNorm() <<" "<<marg_residual_size<< std::endl;
    file.close();
     */
    //std::cout <<"error size: "<<residuals.size() <<" "<<marg_residual_size<<std::endl;
    std::cout<<" error befor: " << err.squaredNorm()<<" " << total_cost <<std::endl;

    // imu factor
    ceres::Problem::EvaluateOptions imufactor_opts;
    imufactor_opts.residual_blocks = imufactor_residual_block_ids;
    double total_cost_imufactor = 0.0;
    vector<double> residuals_imufactor;
    problem.Evaluate(imufactor_opts,&total_cost_imufactor, &residuals_imufactor, nullptr, nullptr);
    Eigen::VectorXd imufactor_err(residuals_imufactor.size());
    imufactor_err.setZero();
    for(int i=0; i< residuals_imufactor.size(); i++)
    {
        //std::cout << residuals[i]<<std::endl;
        imufactor_err[i] = residuals_imufactor[i];
    }
    std::cout<<" IMU error befor: " << imufactor_err.squaredNorm()<<" " << total_cost_imufactor <<std::endl;
    // --------------------  end debug -------------------------
#endif

    //double2vector();

    double2vector2();   // Line pose change
    TicToc t_culling;
    f_manager.removeLineOutlier(Ps,tic,ric);   // remove Line outlier
    ROS_INFO("culling line feautre: %f ms", t_culling.toc());

#ifdef DebugFactor
    // ----------------  debug  ----------------------
    vector2double();
    Eigen::VectorXd err2(marg_residual_size);
    err2.setZero();
    vector<double> residuals2;
    problem.Evaluate(opts,&total_cost, &residuals2, nullptr, nullptr);
    for(int i=0; i< marg_residual_size; i++)
    {
        err[i] = residuals2[i];
    }
    /*
    std::ofstream file;
    file.open("/home/hyj/VINS-err-after.txt",std::ofstream::app);
    file.setf(ios::fixed, ios::floatfield);
    file.precision(5);
    file << err.squaredNorm() <<" "<<marg_residual_size<< std::endl;
    file.close();
     */
    //std::cout <<"error size: "<<residuals.size() <<" "<<marg_residual_size<<std::endl;
    std::cout<<"error after: " << err.squaredNorm()<<" "<< total_cost <<std::endl;
    // imu factor
    double total_cost_imufactor2 = 0.0;
    vector<double> residuals_imufactor2;
    problem.Evaluate(imufactor_opts,&total_cost_imufactor2, &residuals_imufactor2, nullptr, nullptr);
    Eigen::VectorXd imufactor_err2(residuals_imufactor2.size());
    imufactor_err2.setZero();
    for(int i=0; i< residuals_imufactor2.size(); i++)
    {
        //std::cout << residuals[i]<<std::endl;
        imufactor_err2[i] = residuals_imufactor2[i];
    }
    std::cout<<" IMU error after: " << imufactor_err2.squaredNorm()<<" " << total_cost_imufactor <<std::endl;
    //------------------- end debug  --------------------------------
#endif

    // 将优化以后要marg掉的部分转为prior factor
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        // 构建一个新的 prior info
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        /*
           将最老帧上约束转变为 prior, 那有哪些约束是跟这个最老的帧相关的呢？
           1. 上一次优化以后留下的 prior 里可能存在
           2. 跟最老帧 存在 预积分imu 约束
           3. 最老帧上有很多特征观测约束
        */
        // 1. 上一次优化以后留下的 prior 里可能存在约束
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])    // 最老的一帧给丢掉
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 2. 最老的两帧之间的 预积分 factor
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});  // vector<int>{0, 1} 表示要marg的参数下表，比如这里对应para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 3. 最老帧上有很多特征观测约束
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)    // 遍历所有特征
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)             // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)   //遍历这个特征的所有观测
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                   vector<int>{0, 3});// vector<int>{0, 3} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        {
            // Line feature
            int linefeature_index = -1;
            for (auto &it_per_id : f_manager.linefeature)
            {
                it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
                if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
                    continue;
                ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

                int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;
                if (imu_i != 0)             // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                    continue;

                for (auto &it_per_frame : it_per_id.linefeature_per_frame)
                {
                    imu_j++;

                    std::vector<int> drop_set;
                    if(imu_i == imu_j)
                    {
//                        drop_set = vector<int>{0, 2};   // marg pose and feature,  !!!! do not need marg, just drop they  !!!
                        continue;
                    }else
                    {
                        drop_set = vector<int>{2};      // marg feature
                    }

                    Vector4d obs = it_per_frame.lineobs;                          // 在第j帧图像上的观测
                    lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差

                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                                                                                   drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }

        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];   //
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    sum_marg_time_ += t_whole_marginalization.toc();
    mean_marg_time_ = sum_marg_time_/frame_cnt_;
    ROS_INFO("whole marginalization costs: %f", mean_marg_time_);

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}
#endif


void Estimator::optimization(vector<Eigen::Quaterniond> cam_faces)
{
    TicToc t_whole, t_prepare;
    vector2double(featureTracker->cam_faces);

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    if (USE_LINE) {
        for (int i = 0; i < cam_faces.size(); i++) {
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(para_Ex_Pose22[i], SIZE_POSE, local_parameterization);
            problem.SetParameterBlockConstant(para_Ex_Pose22[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;

    // for (auto &_it : f_manager.feature)
    for (int _id : param_feature_id){
        auto & it_per_id = f_manager.feature[_id];
        it_per_id.used_num = it_per_id.feature_per_frame.size();
 
        int feature_index = param_feature_id_to_index[it_per_id.feature_id];

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // ROS_INFO("Adding feature id %d initial depth", it_per_id.feature_id, it_);
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                // std::vector<double*> param_blocks;
                // param_blocks.push_back(para_Pose[imu_i]);
                // param_blocks.push_back(para_Pose[imu_j]);
                // param_blocks.push_back(para_Ex_Pose[0]);
                // param_blocks.push_back(para_Feature[feature_index]);
                // param_blocks.push_back(para_Td[0]);
                // ROS_INFO("Check ProjectionTwoFrameOneCamFactor");
                // f_td->check(param_blocks.data());
                // exit(-1);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[it_per_id.main_cam], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {    
                //For stereo point; main cam must be 0 now
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);

                    // std::vector<double*> param_blocks;
                    // param_blocks.push_back(para_Pose[imu_i]);
                    // param_blocks.push_back(para_Pose[imu_j]);
                    // param_blocks.push_back(para_Ex_Pose[0]);
                    // param_blocks.push_back(para_Ex_Pose[1]);
                    // param_blocks.push_back(para_Feature[feature_index]);             
                    // param_blocks.push_back(para_Td[0]);
                    // ROS_INFO("Check ProjectionTwoFrameTwoCamFactor");
                    // f->check(param_blocks.data());
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    
                    std::vector<double*> param_blocks;
                    param_blocks.push_back(para_Ex_Pose[0]);
                    param_blocks.push_back(para_Ex_Pose[1]);
                    param_blocks.push_back(para_Feature[feature_index]);
                    param_blocks.push_back(para_Td[0]);
                    // ROS_INFO("Check ProjectionOneFrameTwoCamFactor ID: %d, index %d depth init %f Velocity L %f %f %f R %f %f %f", it_per_id.feature_id, feature_index, 
                    //     para_Feature[feature_index][0],
                    //     it_per_id.feature_per_frame[0].velocity.x(), it_per_id.feature_per_frame[0].velocity.y(), it_per_id.feature_per_frame[0].velocity.z(),
                    //     it_per_frame.velocityRight.x(), it_per_frame.velocityRight.y(), it_per_frame.velocityRight.z()
                    //     );
                    // f->check(param_blocks.data());
                    // exit(-1);

                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            
            }
            f_m_cnt++;
        }
    
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

///    ///////////////////////////////S
    int line_m_cnt = 0;
    int linefeature_index = -1;
if(USE_LINE){
    line_m_cnt = 0;
    linefeature_index = -1;
        for (auto &it_per_id : f_manager.linefeature)
        {
            it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
            if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
                continue;

            ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

            ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();
            problem.AddParameterBlock( para_LineFeature[linefeature_index], SIZE_LINE, local_parameterization_line);  // p,q

            int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;

            ///////////////////////////////////////////////
            double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
            int face_id = int(face_id1);
            Eigen::Quaterniond t0x = cam_faces[face_id];
            Eigen::Matrix3d R0x = t0x.toRotationMatrix();
            Eigen::Matrix4d T0x, Tic, Tix;
            T0x.setIdentity();
            Tic.setIdentity();
            T0x.topLeftCorner<3,3>() = R0x;
            Tic.topLeftCorner<3,3>() = ric[0];
            Tic.topRightCorner<3,1>() = tic[0];

            Tix = Tic * T0x;
            Matrix3d Rix = Tix.topLeftCorner<3,3>();
            Quaterniond qix{Rix};
            Vector3d tix = Tix.topRightCorner<3,1>();
            double para_Ex_Pose2[SIZE_POSE];
            para_Ex_Pose2[0] = tix.x();
            para_Ex_Pose2[1] = tix.y();
            para_Ex_Pose2[2] = tix.z();
            //Quaterniond q{ric[i]};
            para_Ex_Pose2[3] = qix.x();
            para_Ex_Pose2[4] = qix.y();
            para_Ex_Pose2[5] = qix.z();
            para_Ex_Pose2[6] = qix.w();





            for (auto &it_per_frame : it_per_id.linefeature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                {
                    //continue;
                }
                Vector4d obs = it_per_frame.lineobs.head(4);                          // 在第j帧图像上的观测
                lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
#if 0
                problem.AddResidualBlock(f, loss_function,
                                         para_Pose[imu_j],
                                         para_Ex_Pose[0],
                                         para_LineFeature[linefeature_index]);
#else
//                problem.AddResidualBlock(f, loss_function,
//                                         para_Pose[imu_j],
//                                         para_Ex_Pose2,
//                                         para_LineFeature[linefeature_index]);
                problem.AddResidualBlock(f, loss_function,
                                         para_Pose[imu_j],
                                         para_Ex_Pose22[face_id],
                                         para_LineFeature[linefeature_index]);
#endif
                line_m_cnt++;
            }
        }
        ROS_INFO("lineFactor: %d, pointFactor:%d", line_m_cnt, f_m_cnt);

    }






    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.check_gradients = true;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;
    // std::cout << summary.FullReport() << endl;
    static double sum_iterations = 0;
    static double sum_solve_time = 0;
    static int solve_count = 0;
    sum_iterations = sum_iterations + summary.iterations.size();
    sum_solve_time = sum_solve_time + summary.total_time_in_seconds;
    solve_count += 1;

    if (ENABLE_PERF_OUTPUT) {
        ROS_INFO("AVG Iter %f time %fms Iterations : %d solver costs: %f \n", 
            sum_iterations/solve_count, sum_solve_time*1000/solve_count,
            static_cast<int>(summary.iterations.size()),  t_solver.toc());
    }

    if(!USE_LINE) {
        double2vector(featureTracker->cam_faces);
    }
    else
    {
        double2vector2(featureTracker->cam_faces);   /// Line pose change
        //double2vector();
        f_manager.removeLineOutlier(Ps,tic,ric,featureTracker->cam_faces);   /// remove Line outlier
    }
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double(featureTracker->cam_faces);

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        int oldest_keyframe_feat_cnt = 0;
        int oldest_keyframe_full_trace_feat_cnt = 0;
        int min_trace_num; //  = 5; // 3; // 5;
        if (MIN_TRACE_TO_MARG == 0)
        {
            min_trace_num = 1;   //   WINDOW_SIZE;
        }
        else
        {
            min_trace_num = MIN_TRACE_TO_MARG;
        }
        for (int _id : param_feature_id) {
            auto & it_per_id = f_manager.feature[_id];

            int feature_index = param_feature_id_to_index[it_per_id.feature_id];

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            /// roger @ 20220206
            // if ( (imu_i != 0) ||  (it_per_id.feature_per_frame.size() != WINDOW_SIZE+1) ) {
            if (imu_i != 0){
                continue;
            }
            oldest_keyframe_feat_cnt++;

            // if(it_per_id.feature_per_frame.size() !=  WINDOW_SIZE+1 ){
            // if(it_per_id.feature_per_frame.size() !=  WINDOW_SIZE + 0 ){
            if(it_per_id.feature_per_frame.size() <  min_trace_num ){
                continue;
            }
            oldest_keyframe_full_trace_feat_cnt++;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if(imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                    vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[it_per_id.main_cam], para_Feature[feature_index], para_Td[0]},
                                                                                    vector<int>{0, 3});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                if(STEREO && it_per_frame.is_stereo)
                {
                    Vector3d pts_j_right = it_per_frame.pointRight;
                    if(imu_i != imu_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[it_per_id.main_cam], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 4});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                        vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{2});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }


        printf("|||||||||||||||||||||||||||||||||||||||||| [oldest_keyframe_feat_cnt(trace >= %d) / oldest_keyframe_feat_cnt]: [%d / %d]\n",min_trace_num, oldest_keyframe_full_trace_feat_cnt,oldest_keyframe_feat_cnt);

        if(USE_LINE){
/// roger line

            // Line feature
            int linefeature_index = -1;
            for (auto &it_per_id : f_manager.linefeature)
            {
                it_per_id.used_num = it_per_id.linefeature_per_frame.size();                // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
                if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))  // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
                    continue;
                ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

                int imu_i = it_per_id.start_frame,imu_j = imu_i - 1;
                if (imu_i != 0)             // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                    continue;


                ///////////////////////////////////////////////
                double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);   // it_per_id.linefeature_per_frame[0].lineobs(4);
                int face_id = int(face_id1);
                Eigen::Quaterniond t0x = cam_faces[face_id];
                Eigen::Matrix3d R0x = t0x.toRotationMatrix();
                Eigen::Matrix4d T0x, Tic, Tix;
                T0x.setIdentity();
                Tic.setIdentity();
                T0x.topLeftCorner<3,3>() = R0x;
                Tic.topLeftCorner<3,3>() = ric[0];
                Tic.topRightCorner<3,1>() = tic[0];

                Tix = Tic * T0x;
                Matrix3d Rix = Tix.topLeftCorner<3,3>();
                Quaterniond qix{Rix};
                Vector3d tix = Tix.topRightCorner<3,1>();
                double para_Ex_Pose2[SIZE_POSE];
                para_Ex_Pose2[0] = tix.x();
                para_Ex_Pose2[1] = tix.y();
                para_Ex_Pose2[2] = tix.z();
                //Quaterniond q{ric[i]};
                para_Ex_Pose2[3] = qix.x();
                para_Ex_Pose2[4] = qix.y();
                para_Ex_Pose2[5] = qix.z();
                para_Ex_Pose2[6] = qix.w();





                for (auto &it_per_frame : it_per_id.linefeature_per_frame)
                {
                    imu_j++;

                    std::vector<int> drop_set;
                    if(imu_i == imu_j)
                    {
//                        drop_set = vector<int>{0, 2};   // marg pose and feature,  !!!! do not need marg, just drop they  !!!
                        continue;
                    }else
                    {
                        drop_set = vector<int>{2};      // marg feature
                    }

                    Vector4d obs = it_per_frame.lineobs.head(4);                          // 在第j帧图像上的观测
                    lineProjectionFactor *f = new lineProjectionFactor(obs);     // 特征重投影误差
#if 0
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_j], para_Ex_Pose[0], para_LineFeature[linefeature_index]},
                                                                                   drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
#else
//                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
//                                                                                   vector<double *>{para_Pose[imu_j], para_Ex_Pose2, para_LineFeature[linefeature_index]},
//                                                                                   drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                   vector<double *>{para_Pose[imu_j], para_Ex_Pose22[face_id], para_LineFeature[linefeature_index]},
                                                                                   drop_set);// vector<int>{0, 2} 表示要marg的参数下标，比如这里对应para_Pose[imu_i], para_Feature[feature_index]
#endif
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }

        }







        TicToc t_pre_margin;
        marginalization_info->preMarginalize();

        // ROS_INFO("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        // ROS_INFO("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        /// roger line multi_face
        for (int i = 0; i < cam_faces.size(); i++){
            addr_shift[reinterpret_cast<long>(para_Ex_Pose22[i])] = para_Ex_Pose22[i];
        }

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double(featureTracker->cam_faces);
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            // ROS_INFO("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_INFO("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            // ROS_INFO("begin marginalization");
            marginalization_info->marginalize();
            ROS_INFO("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            /// roger line multi_face
            for (int i = 0; i < cam_faces.size(); i++){
                addr_shift[reinterpret_cast<long>(para_Ex_Pose22[i])] = para_Ex_Pose22[i];
            }


            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    if(ENABLE_PERF_OUTPUT) {
        ROS_INFO("whole marginalization costs: %fms \n", t_whole_marginalization.toc());
    }
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                windowed_imgs[i] = windowed_imgs[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            windowed_imgs[WINDOW_SIZE] = windowed_imgs[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            windowed_imgs[frame_count - 1] = windowed_imgs[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1, featureTracker->cam_faces);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;
    map<int, Eigen::Vector3d> predictPts1;

    for (auto &_it : f_manager.feature)
    {
        auto & it_per_id = _it.second;
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);

                                                              /// 确保该特征点被持续跟踪到当前帧
             if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            /// roger @ 20220127
            /// if((int)it_per_id.feature_per_frame.size() >= 1 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                Vector3d pts_cam1 = ric[1].transpose() * (pts_local - tic[1]);
                int ptsIndex = it_per_id.feature_id;
                /// 同时预测左目和右目（in case 右目也有匹配点）的特征点在单位球上的位置
                predictPts[ptsIndex] = pts_cam;
                predictPts1[ptsIndex] = pts_cam1;
            }
        }
    }
    featureTracker->setPrediction(predictPts, predictPts1);
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    return (pts_cj.normalized() - uvj).norm();
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    // std::cout<<"aaaaaaaaaaaaaaaa"<<std::endl;
    int reMovedNum = 0;
    for (int _id : param_feature_id) {
        auto & it_per_id = f_manager.feature[_id];
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;

        Vector3d pts_w = Rs[imu_i] * (ric[it_per_id.main_cam] * (depth * pts_i) + tic[it_per_id.main_cam]) + Ps[imu_i];
        // ROS_INFO("PT %d, STEREO %d w %3.2f %3.2f %3.2f drone %3.2f %3.2f %3.2f ptun %3.2f %3.2f %3.2f, depth %f", 
        //     it_per_id.feature_id,
        //     it_per_id.feature_per_frame.front().is_stereo, 
        //     pts_w.x(), pts_w.y(), pts_w.z(),
        //     Ps[imu_i].x(), Ps[imu_i].y(), Ps[imu_i].z(),
        //     pts_i.x(), pts_i.y(), pts_i.z(),
        //     depth
        // );

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {

            imu_j++;
                
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;     

                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[it_per_id.main_cam], tic[it_per_id.main_cam], 
                                                    Rs[imu_j], Ps[imu_j], ric[it_per_id.main_cam], tic[it_per_id.main_cam],
                                                    depth, pts_i, pts_j);
                // printf("ptun   %3.2f %3.2f %3.2f: %3.2f\n", pts_j.x(), pts_j.y(), pts_j.z(), tmp_error);
                                                
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;

                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("ptright %3.2f %3.2f %3.2f: %3.2f\n", pts_j_right.x(), pts_j_right.y(), pts_j_right.z(), tmp_error*FOCAL_LENGTH);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    // printf("ptright %3.2f %3.2f %3.2f: %3.2f\n", pts_j_right.x(), pts_j_right.y(), pts_j_right.z(), tmp_error*FOCAL_LENGTH);
                }       
            }
        }

        // printf("\n");
        double ave_err = err / errCnt;
        //std::cout<<"~~~~~~~~~~~~~~~~ mean reprojection error in pixel: "<<ave_err * FOCAL_LENGTH << std::endl;
        //Looks we have some bugs on outlier rejection!
        if(ave_err * FOCAL_LENGTH > THRES_OUTLIER) {
            //FOCAL_LENGTH
            // ROS_INFO("Removing feature %d on cam %d...  error %f", it_per_id.feature_id, it_per_id.main_cam, ave_err * FOCAL_LENGTH);
            if(PRINT_LOG) {
                printf("THRES_OUTLIER set to %0.3f, FOCAL_LENGTH set to: %0.3f, Removing feature %d on cam %d...  error %f\n",
                       THRES_OUTLIER, FOCAL_LENGTH, it_per_id.feature_id, it_per_id.main_cam, ave_err * FOCAL_LENGTH);
            }
            removeIndex.insert(it_per_id.feature_id);
            reMovedNum++;
        }


    }
    std::cout<<"$$$$$$$$$$$$$$$$$$$$ ~~~~~~~~~~~~~~~~ removed feature num: "<<reMovedNum << std::endl;
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    // ROS_INFO
    if (latest_time < 10) {
        return;
    }

    double dt = t - latest_time;
    if (WARN_IMU_DURATION && dt > (1.5/IMU_FREQ)) {
        ROS_ERROR("[FastPredictIMU] dt %4.1fms t %f lt %f", dt*1000, (t-base)*1000, (latest_time-base)*1000);
    }

    latest_time = t;
    
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    latest_Q.normalize();
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mBuf.lock();

    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    // std::cout << "Ps[frame_count] is " << Ps[frame_count].transpose();
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    fast_prop_inited = true;
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;

    double re_propagate_dt = accBuf.back().first - latest_time;

    if (re_propagate_dt > 3.0/IMAGE_FREQ) {
        ROS_WARN("[updateLatestStates] Reprogate dt too high %4.1fms ", re_propagate_dt*1000);
    }

    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        double dt = t - latest_time;
        if (WARN_IMU_DURATION && dt > 1.5/IMU_FREQ) {
            ROS_ERROR("[updateLatestStates]IMU sample duration too high %4.2fms. Check your IMU and system performance", dt*1000);
            // exit(-1);
        }
        
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mBuf.unlock();
}
