#pragma once

#include "feature_tracker.h"
#include "fisheye_undist.hpp"

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

#include <iostream>
#include <queue>


#include <opencv2/features2d.hpp>
#if 0
#include <opencv2/line_descriptor.hpp>
#else
#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "line_descriptor/include/line_descriptor/descriptor_custom.hpp"
#endif
using namespace std;


using namespace cv::line_descriptor;
using namespace std;
using namespace cv;
using namespace camodocal;

struct Line
{
    Point2f StartPt;
    Point2f EndPt;
    Vector2d StartP;
    Vector2d EndP;
    float lineWidth;
    Point2f Vp;

    Point2f Center;
    Point2f unitDir; // [cos(theta), sin(theta)]
    float length;
    float theta;
    double face_id;
    // para_a * x + para_b * y + c = 0
    float para_a;
    float para_b;
    float para_c;

    float image_dx;
    float image_dy;
    float line_grad_avg;

    float xMin;
    float xMax;
    float yMin;
    float yMax;
    unsigned short id;
    int colorIdx;
};

class FrameLines
{
public:
    int frame_id;
    Mat img;

    vector<Line> vecLine;
    vector< int > lineID;

    // opencv3 lsd+lbd
    std::vector<KeyLine> keylsd;
    Mat lbd_descr;
};
typedef shared_ptr< FrameLines > FrameLinesPtr;

namespace FeatureTracker {

template<class CvMat>
class BaseFisheyeFeatureTracker : public BaseFeatureTracker{
public:
    //virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) = 0;
    //virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down, LineFeatureFrame &lineFeatureFrame) = 0;
    virtual PointLineFeature trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) = 0;
    virtual void readIntrinsicParameter(const vector<string> &calib_file) override;
    FisheyeUndist * get_fisheye_undist(unsigned int index = 0) {
        assert(index<fisheys_undists.size() && "Index Must smaller than camera number");
        return &fisheys_undists[index];
    }
//    BaseFisheyeFeatureTracker(Estimator * _estimator):
//            BaseFeatureTracker(_estimator),
//            t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0))),
//            t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)))
    BaseFisheyeFeatureTracker(Estimator * _estimator):
            BaseFeatureTracker(_estimator),
            t1(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0))),
            t_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)))
    {
        //t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        //t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        //t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
        if (!USE_NEW)
        {
            t2 = t1 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            t3 = t2 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            t4 = t3 * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            t_down = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0));
        }
        else
        {
            double centerFOV = CENTER_FOV * DEG_TO_RAD;
            double sideVerticalFOV = (FISHEYE_FOV - CENTER_FOV) * DEG_TO_RAD /2;
            double rot_angle = (centerFOV + sideVerticalFOV)/2;  // M_PI/2;//

            Eigen::Quaterniond t0;
            t0 = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
            t1 = t0 * Eigen::AngleAxis<double>(-rot_angle , Eigen::Vector3d(1, 0, 0));
            t2 = t1 * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            t3 = t2 * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            t4 =  t3 * Eigen::AngleAxis<double>(-(M_PI / 2 - rot_angle), Eigen::Vector3d(1, 0, 0)) * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0))* Eigen::AngleAxis<double>(M_PI / 2 - rot_angle, Eigen::Vector3d(1, 0, 0));
            t_down = Eigen::AngleAxisd(0, Eigen::Vector3d(1, 0, 0));
        }
    }

    virtual void setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) override;
    //virtual void setPrediction2(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPt_cam1 =  map<int, Eigen::Vector3d>()) override;
    virtual bool setPrediction2(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPt_cam1 =  Eigen::Vector3d()) override;
    virtual Eigen::Vector2d setPrediction3(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPt_cam1 =  Eigen::Vector3d()) override;

    /*
    Eigen::Quaterniond t1; /// t01, transform from cam1 to cam0
    Eigen::Quaterniond t2; /// t02, transform from cam2 to cam0
    Eigen::Quaterniond t3; /// t03, transform from cam3 to cam0
    Eigen::Quaterniond t4; /// t04, transform from cam4 to cam0
     */

protected:
    virtual FeatureFrame setup_feature_frame() override;
    
    std::mutex set_predict_lock;

    void addPointsFisheye();

    vector<cv::Point3f> undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye);
    vector<cv::Point3f> undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward);
    vector<cv::Point3f> ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &pts, 
                                    map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);

        
    virtual void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            cv::Mat imUpTop,
                            cv::Mat imDownTop,
                            cv::Mat imUpSide, 
                            cv::Mat imDownSide);

    vector<FisheyeUndist> fisheys_undists;

    cv::Mat imgUp;
    cv::Mat imgDown;
    //int move_side_to_top;
    bool move_side_to_top;
    //int move_side_to_top = 0;
    vector<int> up_top_ids;
    vector<int> down_top_ids;

    vector<int> up_top_ids_side_part;
    vector<int> down_top_ids_side_part;

    vector<cv::Point3f> up_top_pts_un;
    vector<cv::Point3f> down_top_pts_un;
    vector<cv::Point2f> up_top_pts_;
    vector<cv::Point2f> down_top_pts_;


    cv::Size top_size;
    cv::Size side_size;
    cv::Size side_size_single;

    vector<cv::Point2f> n_pts_up_top, n_pts_down_top, n_pts_up_side;
    std::map<int, cv::Point2f> predict_up_side, predict_up_top, predict_down_top, predict_down_side;
    vector<cv::Point2f> prev_up_top_pts, cur_up_top_pts, prev_up_side_pts, cur_up_side_pts, prev_down_top_pts, prev_down_side_pts;
    
    vector<cv::Point3f> prev_up_top_un_pts,  prev_up_side_un_pts, prev_down_top_un_pts, prev_down_side_un_pts;
    vector<cv::Point2f> cur_down_top_pts, cur_down_side_pts;

    vector<cv::Point3f> up_top_vel, up_side_vel, down_top_vel, down_side_vel;
    vector<cv::Point3f> cur_up_top_un_pts, cur_up_side_un_pts, cur_down_top_un_pts, cur_down_side_un_pts;

    vector<int> ids_up_top, ids_up_side, ids_down_top, ids_down_side;
    map<int, cv::Point2f> up_top_prevLeftPtsMap;
    map<int, cv::Point2f> down_top_prevLeftPtsMap;
    map<int, cv::Point2f> up_side_prevLeftPtsMap;
    map<int, cv::Point2f> down_side_prevLeftPtsMap;


    vector<int> track_up_top_cnt;
    vector<int> track_down_top_cnt;
    vector<int> track_up_side_cnt;
    vector<int> track_down_side_cnt;

    map<int, cv::Point3f> cur_up_top_un_pts_map, prev_up_top_un_pts_map;
    map<int, cv::Point3f> cur_down_top_un_pts_map, prev_down_top_un_pts_map;
    map<int, cv::Point3f> cur_up_side_un_pts_map, prev_up_side_un_pts_map;
    map<int, cv::Point3f> cur_down_side_un_pts_map, prev_down_side_un_pts_map;
    
    CvMat prev_up_top_img, prev_up_side_img, prev_down_top_img;

    // /*
    Eigen::Quaterniond t1; /// t01, transform from cam1 to cam0
    Eigen::Quaterniond t2; /// t02, transform from cam2 to cam0
    Eigen::Quaterniond t3; /// t03, transform from cam3 to cam0
    Eigen::Quaterniond t4; /// t04, transform from cam4 to cam0
    // */
    Eigen::Quaterniond t_down;



    /// roger line
    vector<Line> undistortedLineEndPoints();
    vector<Line> undistortedLineEndPoints2();

    void NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines, vector<pair<int, int> >& lineMatches);
    // void readImage(const cv::Mat &_img,const Mat& mask = Mat());
    void readImage(const cv::Mat &_img,const Mat& mask);
    FrameLinesPtr curframe_, forwframe_;

    cv::Mat undist_map1_, undist_map2_ , K_;

    //camodocal::CameraPtr m_camera;       // pinhole camera

//   // int frame_cnt;
//    vector<int> ids;                     // 每个特征点的id
//    vector<int> linetrack_cnt;           // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
//    // int allfeature_cnt;                  // 用来统计整个地图中有了多少条线，它将用来赋值
//
//    double sum_time;
//    double mean_time;

};

//class FisheyeFeatureTrackerCuda: public BaseFisheyeFeatureTracker<cv::cuda::GpuMat> {
//public:
//    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;
//
//    inline FeatureFrame trackImage_blank_init(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) {
//        is_blank_init = true;
//        auto ff = trackImage(_cur_time, fisheye_imgs_up, fisheye_imgs_down);
//        is_blank_init = false;
//        return ff;
//    }
//
//    FisheyeFeatureTrackerCuda(Estimator * _estimator): BaseFisheyeFeatureTracker<cv::cuda::GpuMat>(_estimator) {
//
//    }
//
//protected:
//    bool is_blank_init = false;
//    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down,
//                            cv::cuda::GpuMat imUpTop,
//                            cv::cuda::GpuMat imDownTop,
//                            cv::cuda::GpuMat imUpSide,
//                            cv::cuda::GpuMat imDownSide);
//    std::vector<cv::cuda::GpuMat> prev_up_top_pyr, prev_down_top_pyr, prev_up_side_pyr;
//
//};



class FisheyeFeatureTrackerOpenMP: public BaseFisheyeFeatureTracker<cv::Mat> {
    public:
        FisheyeFeatureTrackerOpenMP(Estimator * _estimator): BaseFisheyeFeatureTracker<cv::Mat>(_estimator) {
        }

        //virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;
    //    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down, LineFeatureFrame &lineFeatureFrame) override;
        virtual PointLineFeature trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;

    //vector<Line> undistortedLineEndPoints();







protected:
        std::vector<cv::Mat> * prev_up_top_pyr = nullptr, * prev_down_top_pyr = nullptr, * prev_up_side_pyr = nullptr;

};

//class FisheyeFeatureTrackerVWorks: public FisheyeFeatureTrackerCuda {
//public:
//    virtual FeatureFrame trackImage(double _cur_time, cv::InputArray fisheye_imgs_up, cv::InputArray fisheye_imgs_down) override;
//protected:
//#ifdef WITH_VWORKS
//    cv::cuda::GpuMat up_side_img_fix;
//    cv::cuda::GpuMat down_side_img_fix;
//    cv::cuda::GpuMat up_top_img_fix;
//    cv::cuda::GpuMat down_top_img_fix;
//
//    cv::cuda::GpuMat mask_up_top_fix, mask_down_top_fix, mask_up_side_fix;
//
//    vx_image vx_up_top_image;
//    vx_image vx_down_top_image;
//    vx_image vx_up_side_image;
//    vx_image vx_down_side_image;
//
//    vx_image vx_up_top_mask;
//    vx_image vx_down_top_mask;
//    vx_image vx_up_side_mask;
//
//    nvx::FeatureTracker* tracker_up_top = nullptr;InputArraytop_img, cv::cuda::GpuMat & down_top_img, cv::cuda::GpuMat & up_side_img, cv::cuda::GpuMat & down_side_img);
//
//    void process_vworks_tracking(nvx::FeatureTracker* _tracker, vector<int> & _ids, vector<cv::Point2f> & prev_pts, vector<cv::Point2f> & cur_pts,
//        vector<int> & _track, vector<cv::Point2f> & n_pts, map<int, int> &_id_by_index, bool debug_output=false);
//    bool first_frame = true;
//
//    map<int, int> up_top_id_by_index;
//    map<int, int> down_top_id_by_index;
//    map<int, int> up_side_id_by_index;
//#endif
//};

cv::cuda::GpuMat concat_side(const std::vector<cv::cuda::GpuMat> & arr);
cv::Mat concat_side(const std::vector<cv::Mat> & arr);
std::vector<cv::Mat> convertCPUMat(const std::vector<cv::cuda::GpuMat> & arr);


template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::setPrediction(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPts_cam1) {
    // std::cout << 
    set_predict_lock.lock();
    predict_up_top.clear();
    predict_up_side.clear();
    predict_down_top.clear();
    predict_down_side.clear();


    //cv::Size top_size;
    //cv::Size side_size;
    int topCol = top_size.width;
    int topRow = top_size.height;
    //int sideCol = side_size.width;
    //int sideRow = side_size.height;
    int sideCol = side_size_single.width;
    int sideRow = side_size_single.height;


    for (auto it : predictPts_cam0) {
        /// _id: 特征点在单位球坐标系下的归一化坐标的id
        int _id = it.first;
        //auto pt = it.second;
        Eigen::Vector3d pt = it.second;
        /// 这个归一化 [球坐标] 应该落在[ 前、后、左、右、中] 哪一部分针孔坐标系上, 并给出在针孔相机下的2d坐标
        auto ret = fisheys_undists[0].project_point_to_vcam_id(pt);
        /// 前面的id表示预测的特征点落在0，1，2，3，4哪个针孔相机内
        if (!CUBE_MAP) {
            if (ret.first >= 0) {
                if (ret.first == 0) {
                    predict_up_top[_id] = ret.second;
                } else if (ret.first > 1) {
                    //if (!CUBE_MAP) {
                    /// 从 单针孔图坐标 转换到 多针孔图坐标
                    cv::Point2f pt(ret.second.x + (ret.first - 1) * WIDTH, ret.second.y);
                    predict_up_side[_id] = pt;
                    // }
//                else{
//                    if (ret.first == 1){
//
//                    }
//                    if (ret.first == 2){
//
//                    }
//                    if (ret.first == 3){
//
//                    }
//                    if (ret.first == 4){
//
//                    }

                    //               }
                }
            }
        }
        else{
            if (ret.first >= 0) {
                if (ret.first == 0) {
                    predict_up_top[_id] = ret.second;
// if ()
                    cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow);
                    predict_up_side[_id] = pt;
                }
                if (ret.first == 1){
                    cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow + topCol);
                    predict_up_side[_id] = pt;
                }
                if (ret.first == 2){
                    cv::Point2f pt(ret.second.y + sideRow + topCol, topCol - 1 - ret.second.x + sideRow);
                    predict_up_side[_id] = pt;
                }
                if (ret.first == 3){
                    cv::Point2f pt(topCol - 1 - ret.second.x + sideRow, sideRow - 1 - ret.second.y + 0);
                    predict_up_side[_id] = pt;
                }
                if (ret.first == 4){
                    cv::Point2f pt(sideRow - 1 - ret.second.y + 0, ret.second.x + sideRow);
                    predict_up_side[_id] = pt;
                }
            }

        }
    }

    for (auto it : predictPts_cam1) {
        int _id = it.first;
        auto pt = it.second;
        auto ret = fisheys_undists[1].project_point_to_vcam_id(pt);
        /// 前面的id表示预测的特征点落在0，1，2，3，4哪个针孔相机内
        if (!CUBE_MAP) {
            if (ret.first >= 0) {
                if (ret.first == 0) {
                    predict_down_top[_id] = ret.second;
                } else if (ret.first > 1) {
                    cv::Point2f pt(ret.second.x + (ret.first - 1) * WIDTH, ret.second.y);
                    predict_down_side[_id] = pt;
                }
            }
        }
        else
        {
            if (ret.first >= 0) {
                if (ret.first == 0) {
                    predict_down_top[_id] = ret.second;

                    cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow);
                    predict_down_side[_id] = pt;
                    //predict_down_side[_id] = ret.second;
                }
                if (ret.first == 1){
                    cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow + topCol);
                    predict_down_side[_id] = pt;
                }
                if (ret.first == 2){
                    cv::Point2f pt(ret.second.y + sideRow + topCol, topCol - 1 - ret.second.x + sideRow);
                    predict_down_side[_id] = pt;
                }
                if (ret.first == 3){
                    cv::Point2f pt(topCol - 1 - ret.second.x + sideRow, sideRow - 1 - ret.second.y + 0);
                    predict_down_side[_id] = pt;
                }
                if (ret.first == 4){
                    cv::Point2f pt(sideRow - 1 - ret.second.y + 0, ret.second.x + sideRow);
                    predict_down_side[_id] = pt;
                }
            }

        }
    }
    set_predict_lock.unlock();
}
    template<class CvMat>
    Eigen::Vector2d BaseFisheyeFeatureTracker<CvMat>::setPrediction3(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPts_cam1) {
        // std::cout <<
//        set_predict_lock.lock();
        //      predict_up_top.clear();
        //    predict_up_side.clear();
        //  predict_down_top.clear();
        // predict_down_side.clear();


        //cv::Size top_size;
        //cv::Size side_size;
        int topCol = top_size.width;
        int topRow = top_size.height;
        //int sideCol = side_size.width;
        //int sideRow = side_size.height;
        int sideCol = side_size_single.width;
        int sideRow = side_size_single.height;

        //std::map<int, cv::Point2f> predict_up_side2, predict_up_top2, predict_down_top2, predict_down_side2;
        //for (auto it : predictPts_cam0) {
        /// _id: 特征点在单位球坐标系下的归一化坐标的id
        //  int _id = it.first;
        //auto pt = it.second;
        Eigen::Vector3d pt = predictPts_cam0; //it.second;
        /// 这个归一化 [球坐标] 应该落在[ 前、后、左、右、中] 哪一部分针孔坐标系上, 并给出在针孔相机下的2d坐标
        //auto ret = fisheys_undists[0].project_point_to_vcam_id2(pt);
Eigen::Vector2d ret = fisheys_undists[0].project_point_to_vcam_id2(pt);
return ret;
    }
        template<class CvMat>
  //  void BaseFisheyeFeatureTracker<CvMat>::setPrediction2(const map<int, Eigen::Vector3d> &predictPts_cam0, const map<int, Eigen::Vector3d> &predictPts_cam1) {
  bool BaseFisheyeFeatureTracker<CvMat>::setPrediction2(const Eigen::Vector3d predictPts_cam0, const Eigen::Vector3d predictPts_cam1) {
        // std::cout <<
//        set_predict_lock.lock();
  //      predict_up_top.clear();
    //    predict_up_side.clear();
      //  predict_down_top.clear();
       // predict_down_side.clear();


        //cv::Size top_size;
        //cv::Size side_size;
        int topCol = top_size.width;
        int topRow = top_size.height;
        //int sideCol = side_size.width;
        //int sideRow = side_size.height;
        int sideCol = side_size_single.width;
        int sideRow = side_size_single.height;

        //std::map<int, cv::Point2f> predict_up_side2, predict_up_top2, predict_down_top2, predict_down_side2;
        //for (auto it : predictPts_cam0) {
            /// _id: 特征点在单位球坐标系下的归一化坐标的id
          //  int _id = it.first;
            //auto pt = it.second;
            Eigen::Vector3d pt = predictPts_cam0; //it.second;
            /// 这个归一化 [球坐标] 应该落在[ 前、后、左、右、中] 哪一部分针孔坐标系上, 并给出在针孔相机下的2d坐标
            auto ret = fisheys_undists[0].project_point_to_vcam_id(pt);


            /// 前面的id表示预测的特征点落在0，1，2，3，4哪个针孔相机内
            if (!CUBE_MAP) {
                if (ret.first >= 0) {
                    if (ret.first == 0) {
                        return true; //predict_up_top2[_id] = ret.second;
                    }
                    else {
                        return false;
                    }
                 //   else if (ret.first > 1) {
                 //       //if (!CUBE_MAP) {
                        /// 从 单针孔图坐标 转换到 多针孔图坐标
                 //       cv::Point2f pt(ret.second.x + (ret.first - 1) * WIDTH, ret.second.y);
                  //      predict_up_side2[_id] = pt;
                        // }
//                else{
//                    if (ret.first == 1){
//
//                    }
//                    if (ret.first == 2){
//
//                    }
//                    if (ret.first == 3){
//
//                    }
//                    if (ret.first == 4){
//
//                    }

                        //               }
                    }
                else{
                    return false;
                }
                //}
            }
            else{
                if (ret.first >= 0) {
                    if (ret.first == 0) {
                        return true; //predict_up_top2[_id] = ret.second;
// if ()
                        // cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow);
                        // predict_up_side2[_id] = pt;
                    }
                    else
                    {
                        return false;
                    }
                    //predictPts_cam1.pus
#if 0
                    if (ret.first == 1){
                        cv::Point2f pt(ret.second.x + sideRow, ret.second.y + sideRow + topCol);
                        predict_up_side2[_id] = pt;
                    }
                    if (ret.first == 2){
                        cv::Point2f pt(ret.second.y + sideRow + topCol, topCol - 1 - ret.second.x + sideRow);
                        predict_up_side2[_id] = pt;
                    }
                    if (ret.first == 3){
                        cv::Point2f pt(topCol - 1 - ret.second.x + sideRow, sideRow - 1 - ret.second.y + 0);
                        predict_up_side2[_id] = pt;
                    }
                    if (ret.first == 4){
                        cv::Point2f pt(sideRow - 1 - ret.second.y + 0, ret.second.x + sideRow);
                        predict_up_side2[_id] = pt;
                    }
#endif
                }
                else
                {
                    return false;
                }

            }
        //}
return false;

    }
template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::drawTrackFisheye(const cv::Mat & img_up,
    const cv::Mat & img_down,
    cv::Mat imUpTop,
    cv::Mat imDownTop,
    cv::Mat imUpSide, 
    cv::Mat imDownSide)
{
    // ROS_INFO("Up image %d, down %d", imUp.size(), imDown.size());
    cv::Mat imTrack;
    cv::Mat fisheye_up;
    cv::Mat fisheye_down;
#if 0
    cv::Mat fisheyeConcat;
    cv::hconcat(img_orig_up, img_orig_down, fisheyeConcat);
    cv::imshow("raw fisheye",fisheyeConcat);
    cv::waitKey(2);
#endif
    // fisheye_up = img_orig_up;
    // fisheye_down = img_orig_down;

    int side_height = imUpSide.size().height;

    int cnt = 0;

    if (imUpTop.size().width == 0) {
        imUpTop = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3, cv::Scalar(0, 0, 0));
        cnt ++; 
    }

    if (imDownTop.size().width == 0) {
        imDownTop = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3, cv::Scalar(0, 0, 0)); 
        cnt ++; 
    }

    //128
    if (img_up.size().width == 1024) {
        fisheye_up = img_up(cv::Rect(190, 62, 900, 900));
        fisheye_down = img_down(cv::Rect(190, 62, 900, 900));
    } else {
        fisheye_up = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0)); 
        fisheye_down = cv::Mat(cv::Size(900, 900), CV_8UC3, cv::Scalar(0, 0, 0));
        fisheye_up = img_orig_up;
        fisheye_down = img_orig_down;
        cnt ++; 
    }

    cv::resize(fisheye_up, fisheye_up, cv::Size(WIDTH, WIDTH));
    cv::resize(fisheye_down, fisheye_down, cv::Size(WIDTH, WIDTH));
    if (fisheye_up.channels() != 3) {
        cv::cvtColor(fisheye_up,   fisheye_up,   cv::COLOR_GRAY2BGR);
        cv::cvtColor(fisheye_down, fisheye_down, cv::COLOR_GRAY2BGR);
    }

    if (imUpTop.channels() != 3) {
        if (!imUpTop.empty()) {
            cv::cvtColor(imUpTop, imUpTop, cv::COLOR_GRAY2BGR);
        }
    }
    
    if (imDownTop.channels() != 3) {
        if(!imDownTop.empty()) {
            cv::cvtColor(imDownTop, imDownTop, cv::COLOR_GRAY2BGR);
        }
    }
    
    if(imUpSide.channels() != 3) {
        if(!imUpSide.empty()) {
            cv::cvtColor(imUpSide, imUpSide, cv::COLOR_GRAY2BGR);
        }
    }

    if(imDownSide.channels() != 3) {
        if(!imDownSide.empty()) {
            cv::cvtColor(imDownSide, imDownSide, cv::COLOR_GRAY2BGR);
        }
    }

    if(enable_up_top) {
        drawTrackImage(imUpTop, cur_up_top_pts, ids_up_top, up_top_prevLeftPtsMap, predict_up_top);
    }

    if(enable_down_top) {
        drawTrackImage(imDownTop, cur_down_top_pts, ids_down_top, down_top_prevLeftPtsMap, predict_down_top);
    }

    if(enable_up_side) {
        //vector<cv::Point2f> cur_up_side_pts_temp = cur_up_side_pts;
        // for(int i = 0; i < )
        drawTrackImage(imUpSide, cur_up_side_pts, ids_up_side, up_side_prevLeftPtsMap, predict_up_side);
    }

    if(enable_down_side) {
        drawTrackImage(imDownSide, cur_down_side_pts, ids_down_side, pts_map(ids_up_side, cur_up_side_pts), predict_down_side);
    }

    //Show images
    int side_count = 3;
    if (enable_rear_side) {
        side_count = 4;
    }

    for (int i = 1; i < side_count + 1; i ++) {
        cv::line(imUpSide, cv::Point2d(i*WIDTH, 0), cv::Point2d(i*WIDTH, side_height), cv::Scalar(255, 0, 0), 1);
        cv::line(imDownSide, cv::Point2d(i*WIDTH, 0), cv::Point2d(i*WIDTH, side_height), cv::Scalar(255, 0, 0), 1);
    }

    if (enable_down_side) {
        if (!CUBE_MAP) {
            cv::vconcat(imUpSide, imDownSide, imTrack);
        }
        else{
            cv::hconcat(imUpSide, imDownSide, imTrack);
        }
    } else {
        imTrack = imUpSide;
    }

    cv::Mat top_cam;
    if(!CUBE_MAP){
        cv::hconcat(imUpTop, imDownTop, top_cam);
        cv::hconcat(fisheye_up, top_cam, top_cam);
        cv::hconcat(top_cam, fisheye_down, top_cam);
    }
    else {
        cv::Mat imUpTop2, imDownTop2, fisheye_up2, fisheye_down2;
        cv::rotate(imUpTop, imUpTop2, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(imDownTop, imDownTop2, cv::ROTATE_90_CLOCKWISE);
        cv::hconcat(imUpTop2, imDownTop2, top_cam);

        cv::rotate(fisheye_up, fisheye_up2, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(fisheye_down, fisheye_down2, cv::ROTATE_90_CLOCKWISE);
        cv::hconcat(fisheye_up2, top_cam, top_cam);
        cv::hconcat(top_cam, fisheye_down2, top_cam);

    }
    //cv::hconcat(fisheye_up, top_cam, top_cam);
    //cv::hconcat(top_cam, fisheye_down, top_cam);


    // ROS_INFO("Imtrack width %d", imUpSide.size().width);
    cv::resize(top_cam, top_cam, cv::Size(imUpSide.size().width, imUpSide.size().width/4));
    
    if (cnt < 3) {
if(!CUBE_MAP) {
    cv::vconcat(top_cam, imTrack, imTrack);
}
else {
    cv::Mat top_cam2;
    cv::rotate(top_cam, top_cam2, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::hconcat(top_cam2, imTrack, imTrack);
}
    }
    
    double fx = ((double)SHOW_WIDTH) / ((double) imUpSide.size().width);
    cv::resize(imTrack, imTrack, cv::Size(), fx, fx);
    //if (!CUBE_MAP) {
    if (true) {
        cv::imshow("tracking", imTrack);
    }
    else
    {
        //cv::imshow("tracking", imTrack.t());
        cv::Mat imgRotate;
        cv::rotate(imTrack, imgRotate, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imshow("tracking", imgRotate);
    }
    cv::waitKey(2);
}



template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::readIntrinsicParameter(const vector<string> &calib_file)
{
    /// 遍历相机文件，导入并加载相机参数，生成remap表
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);

        ROS_INFO("Use as fisheye %s", calib_file[i].c_str());
        //FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, true, WIDTH);
        FisheyeUndist un(calib_file[i].c_str(), i, FISHEYE_FOV, false, WIDTH);
        fisheys_undists.push_back(un);

    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::ptsVelocity3D(vector<int> &ids,
                                                                    vector<cv::Point3f> &cur_pts,
                                                                    map<int, cv::Point3f> &cur_id_pts,
                                                                    map<int, cv::Point3f> &prev_id_pts)
{
    // ROS_INFO("Pts %ld Prev pts %ld IDS %ld", cur_pts.size(), prev_id_pts.size(), ids.size());
    vector<cv::Point3f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts[ids[i]] = cur_pts[i];
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            std::map<int, cv::Point3f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (cur_pts[i].x - it->second.x) / dt;
                double v_y = (cur_pts[i].y - it->second.y) / dt;
                double v_z = (cur_pts[i].z - it->second.z) / dt;
                pts_velocity.push_back(cv::Point3f(v_x, v_y, v_z));
                // ROS_INFO("id %d Dt %f, cur pts %f %f %f prev %f %f %f vel %f %f %f", 
                //     ids[i],
                //     cur_pts[i].x, cur_pts[i].y, cur_pts[i].z,
                //     it->second.x, it->second.y, it->second.z,
                //     v_x, v_y, v_z);

            }
            else
                pts_velocity.push_back(cv::Point3f(0, 0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point3f(0, 0, 0));
        }
    }
    return pts_velocity;
}

template<class CvMat>
void BaseFisheyeFeatureTracker<CvMat>::addPointsFisheye()
{
    // ROS_INFO("Up top new pts %d", n_pts_up_top.size());
    for (auto p : n_pts_up_top)
    {
        cur_up_top_pts.push_back(p);
        ids_up_top.push_back(n_id++);
        track_up_top_cnt.push_back(1);
    }

    for (auto p : n_pts_down_top)
    {
        cur_down_top_pts.push_back(p);
        ids_down_top.push_back(n_id++);
        track_down_top_cnt.push_back(1);
    }

    for (auto p : n_pts_up_side)
    {
        cur_up_side_pts.push_back(p);
        ids_up_side.push_back(n_id++);
        track_up_side_cnt.push_back(1);
    }
}

template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye) {
    auto & cam = fisheye.cam_top;
    vector<cv::Point3f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        b.normalize();
#ifdef UNIT_SPHERE_ERROR
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
#else
        un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
#endif
    }
    return un_pts;
}

template<class CvMat>
FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() {
    FeatureFrame ff;
    ///                                              特征点id      [在cube_map上跟踪到的2d点坐标] [通过t0 t1 t2 t3 t4转化到单位球上的去畸变后的坐标] [单位球坐标下的速度]
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_top, cur_up_top_pts, cur_up_top_un_pts, up_top_vel, 0);   
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_side, cur_up_side_pts, cur_up_side_un_pts, up_side_vel, 0);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_top, cur_down_top_pts, cur_down_top_un_pts, down_top_vel, 1);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_side, cur_down_side_pts, cur_down_side_un_pts, down_side_vel, 1);

    return ff;
}


template<class CvMat>
vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward) {
    auto & cam = fisheye.cam_side;
    std::vector<cv::Point3f> un_pts;
    //Need to rotate pts
    //Side pos 1,2,3,4 is left front right
    //For downward camera, additational rotate 180 deg on x is required


    int topCol = top_size.width;
    int topRow = top_size.height;
    //int sideCol = side_size.width;
    //int sideRow = side_size.height;
    int sideCol = side_size_single.width;
    int sideRow = side_size_single.height;

    if (PRINT_LOG) {
        std::cout << "topCol, topRow, sideCol, sideRow:" << topCol << " " << topRow << " " << sideCol << " " << sideRow
                  << std::endl;
    }

 // vector<int> top_ids;
    for (unsigned int i = 0; i < pts.size(); i++) {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        int side_pos_id;
        side_pos_id = -1;
        if (!CUBE_MAP) {
            side_pos_id = floor(a.x() / WIDTH) + 1;
            /// 从 多针孔图坐标 转换到 单针孔图坐标
            a.x() = a.x() - floor(a.x() / WIDTH) * WIDTH;

            cam->liftProjective(a, b);

            if (side_pos_id == 1) {

                /// b0 = t02 * b2

                b = t1 * b;
            } else if (side_pos_id == 2) {
                b = t2 * b;
            } else if (side_pos_id == 3) {
                b = t3 * b;
            } else if (side_pos_id == 4) {
                b = t4 * b;
            } else {
                ROS_ERROR("Err pts img position; i %d side_pos_id %d!! x %f width %d", i, side_pos_id, a.x(),
                          top_size.width);
                assert(false && "ERROR Pts img position");
            }

            if (is_downward) {
                b = t_down * b;
            }
        }
    else
        {
            //int side_pos_id;
            //std::cout<<"vector2d: "<<a.transpose()<<", [a.x() a.y()]: ["<<a.x()<<" "<<a.y()<<"]"<<std::endl;
            if (a.x() >= sideRow & a.x()<=sideRow+topCol-1
                &a.y() >= sideRow & a.y()<=sideRow+topCol-1){
                 side_pos_id = 0;
                Eigen::Vector2d pt(a.x() - sideRow, a.y() - sideRow);
#if 1
                cam->liftProjective(pt, b);
                if (!is_downward) {
                    up_top_ids.push_back(i);
                    up_top_pts_.push_back(pts[i]);
                }
                else{
                    down_top_ids.push_back(i);
                    down_top_pts_.push_back(pts[i]);
                }
#endif
                //continue;
            }
            /*if(!is_downward){
                up_top_ids_side_part.push_back(i);
            }
            else{
                down_top_ids_side_part.push_back(i);
            }*/
            if (a.x() >= sideRow & a.x()<=sideRow+topCol-1
                &a.y() >= sideRow+topCol & a.y()<=2*sideRow+topCol-1){
                 side_pos_id = 1;
                Eigen::Vector2d pt(a.x() - sideRow, a.y() - sideRow - topCol);
                cam->liftProjective(pt, b);

                /// b0 = t02 * b2
                b = t1 * b;
                if(!is_downward){
                    up_top_ids_side_part.push_back(i);
                }
                else{
                    down_top_ids_side_part.push_back(i);
                }
            }
            if (a.x() >= sideRow+topCol & a.x()<=2*sideRow+topCol-1
                &a.y() >= sideRow & a.y()<=sideRow+topCol-1){
                 side_pos_id = 2;
                Eigen::Vector2d pt((topCol-1-(a.y() - sideRow)),(a.x() - (sideRow + topCol)));
                cam->liftProjective(pt, b);
                b = t2 * b;
                if(!is_downward){
                    up_top_ids_side_part.push_back(i);
                }
                else{
                    down_top_ids_side_part.push_back(i);
                }
            }
            if (a.x() >= sideRow & a.x()<=sideRow+topCol-1
                &a.y() >= 0 & a.y()<=sideRow-1){
                 side_pos_id = 3;
                Eigen::Vector2d pt((topCol-1-(a.x()-sideRow)),(sideRow-1-a.y()));
                cam->liftProjective(pt, b);
                b = t3 * b;
                if(!is_downward){
                    up_top_ids_side_part.push_back(i);
                }
                else{
                    down_top_ids_side_part.push_back(i);
                }
            }
            if (a.x() >= 0 & a.x()<=sideRow-1
                &a.y() >= sideRow & a.y()<=sideRow+topCol-1){
                 side_pos_id = 4;
                Eigen::Vector2d pt((a.y() - sideRow),(sideRow-1-a.x()));
                cam->liftProjective(pt, b);
                b = t4 * b;
                if(!is_downward){
                    up_top_ids_side_part.push_back(i);
                }
                else{
                    down_top_ids_side_part.push_back(i);
                }
            }
            if (is_downward) {
                b = t_down * b;
            }

        }

        b.normalize();
#ifdef UNIT_SPHERE_ERROR
    //if (move_side_to_top) {
    //    if (true) {
    if (move_side_to_top) {
        if (side_pos_id > 0) {
            un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
        }
    }
    else
    {
        un_pts.push_back(cv::Point3f(b.x(), b.y(), b.z()));
    }
    if (side_pos_id == 0)
    {
        if (!is_downward){
            up_top_pts_un.push_back(cv::Point3f(b.x(), b.y(), b.z()));
        }
        else
        {
            down_top_pts_un.push_back(cv::Point3f(b.x(), b.y(), b.z()));
        }

    }
#else
        if (fabs(b.z()) < 1e-3) {
            b.z() = 1e-3;
        }
        
        if (b.z() < - 1e-2) {
            //Is under plane, z is -1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), -1));
        } else if (b.z() > 1e-2) {
            //Is up plane, z is 1
            un_pts.push_back(cv::Point3f(b.x() / b.z(), b.y() / b.z(), 1));
        }
#endif
    }
    return un_pts;
}

};
