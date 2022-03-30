
#include "feature_tracker.h"
#include "../estimator/estimator.h"
#include "fisheye_undist.hpp"
#include "feature_tracker_fisheye.hpp"

#if 0
    #include "linefeature_tracker.h"
#endif


namespace FeatureTracker {

#define MATCHES_DIST_THRESHOLD 60  // 80 // 100 // 80 // 60 // 30 // 10 // 30 // 60 // 30
    void visualize_line_match(Mat imageMat1, Mat imageMat2,
                              std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                              std::vector<DMatch> good_matches)
    {
        //	Mat img_1;
        cv::Mat img1,img2;
        if (imageMat1.channels() != 3){
            cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
        }
        else{
            img1 = imageMat1;
        }

        if (imageMat2.channels() != 3){
            cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
        }
        else{
            img2 = imageMat2;
        }


        //    srand(time(NULL));
        int lowest = 0, highest = 255;
        int range = (highest - lowest) + 1;
        for (int k = 0; k < good_matches.size(); ++k) {
            DMatch mt = good_matches[k];

            KeyLine line1 = octave0_1[mt.queryIdx];  // trainIdx
            KeyLine line2 = octave0_2[mt.trainIdx];  //queryIdx


            unsigned int r = lowest + int(rand() % range);
            unsigned int g = lowest + int(rand() % range);
            unsigned int b = lowest + int(rand() % range);
            cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
            cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
            cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

            cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
            cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
            cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
            cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
            cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

        }
        /* plot matches */
        /*
        cv::Mat lsd_outImg;
        std::vector<char> lsd_mask( lsd_matches.size(), 1 );
        drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
        DrawLinesMatchesFlags::DEFAULT );

        imshow( "LSD matches", lsd_outImg );
        */
        if(SHOW_LINE) {
#if 0
            imshow("LSD matches1", img1);
#if 0
            imshow("LSD matches2", img2);
#endif
#else
            int gap = 30;
            // cv::Mat gap_image(img1.rows, gap, CV_8UC3, cv::Scalar(255, 255, 255));
            cv::Mat gap_image(img1.rows, gap, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::hconcat(img1, gap_image, gap_image);
            cv::hconcat(gap_image, img2, img2);
            cv::Mat img22;
            if(img2.cols > 800)
            {
                cv::resize(img2, img22, cv::Size(800, 400));
            }
            else
            {
                img22 = img2;
            }

            imshow("LSD matches 1 & 2 ", img22);
#endif
            waitKey(1);
        }
    }

cv::Mat concat_side(const std::vector<cv::Mat> & arr) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;

    if (!CUBE_MAP) {
        if (enable_rear_side) {
            cv::Mat NewImg(rows, cols * 4, arr[1].type());
            for (int i = 1; i < 5; i++) {
                arr[i].copyTo(NewImg(cv::Rect(cols * (i - 1), 0, cols, rows)));
            }
            return NewImg;
        } else {
            cv::Mat NewImg(rows, cols * 3, arr[1].type());
            for (int i = 1; i < 4; i++) {
                arr[i].copyTo(NewImg(cv::Rect(cols * (i - 1), 0, cols, rows)));
            }
            return NewImg;
        }
    }
    else
    {
        //if (enable_down_side) {
        if (true) {
            cv::Mat NewImg(cols + 2 * rows, cols + 2 * rows, arr[1].type());
            NewImg.setTo(0);

            //if (arr[0].cols > 2) {
            if (!arr[0].empty()) {
                arr[0].copyTo(NewImg(cv::Rect(rows, rows, cols, cols)));
            }
            arr[1].copyTo(NewImg(cv::Rect(rows, rows + cols, cols, rows)));


            //cv::Mat BlankImg(rows, rows, arr[1].type(), 0);
#if 0
            cv::Mat BlankImg(rows, rows, arr[1].type());
            BlankImg.setTo(0);
            BlankImg.copyTo(NewImg(cv::Rect(0, 0, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(rows + cols, 0, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(0, rows + cols, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(rows + cols, rows + cols, rows, rows)));
#endif

            cv::Mat temp2 = arr[2];
            cv::Mat temp22;
            cv::rotate(temp2, temp22, cv::ROTATE_90_COUNTERCLOCKWISE);
            temp22.copyTo(NewImg(cv::Rect(rows + cols, rows, rows, cols)));

            cv::Mat temp3 = arr[3];
            cv::Mat temp33;
            cv::rotate(temp3, temp33, cv::ROTATE_180);
            temp33.copyTo(NewImg(cv::Rect(rows, 0, cols, rows)));

            cv::Mat temp4 = arr[4];
            cv::Mat temp44;
            cv::rotate(temp4, temp44, cv::ROTATE_90_CLOCKWISE);
            temp44.copyTo(NewImg(cv::Rect(0, rows, rows, cols)));
            cv::Mat asfa121 = cv::imread("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/concat.png");

            if (asfa121.cols < 10) {
                imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/concat.png", NewImg);
            }
            return NewImg;
        }
#if 0
        if (enable_up_side) {
            cv::Mat NewImg(cols + 2 * rows, cols + 2 * rows, arr[1].type());
            arr[0].copyTo(NewImg(cv::Rect(rows, rows, cols, cols)));
            arr[1].copyTo(NewImg(cv::Rect(rows, rows + cols, cols, rows)));


            //cv::Mat BlankImg(rows, rows, arr[1].type(), 0);
#if 0
            cv::Mat BlankImg(rows, rows, arr[1].type());
            BlankImg.setTo(0);
            BlankImg.copyTo(NewImg(cv::Rect(0, 0, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(rows + cols, 0, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(0, rows + cols, rows, rows)));
            BlankImg.copyTo(NewImg(cv::Rect(rows + cols, rows + cols, rows, rows)));
#endif

            cv::Mat temp2 = arr[2];
            cv::Mat temp22;
            cv::rotate(temp2, temp22, cv::ROTATE_90_COUNTERCLOCKWISE);
            temp22.copyTo(NewImg(cv::Rect(rows + cols, rows, rows, cols)));

            cv::Mat temp3 = arr[3];
            cv::Mat temp33;
            cv::rotate(temp3, temp33, cv::ROTATE_180);
            temp33.copyTo(NewImg(cv::Rect(rows, 0, cols, rows)));

            cv::Mat temp4 = arr[4];
            cv::Mat temp44;
            cv::rotate(temp4, temp44, cv::ROTATE_90_CLOCKWISE);
            temp44.copyTo(NewImg(cv::Rect(0, rows, rows, cols)));
            cv::Mat asfa121 = cv::imread("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/concat.png");

            if (asfa121.cols < 10) {
                imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/concat.png", NewImg);
            }
            return NewImg;
        }
#endif
    }
}

 static void trackLine(const cv::Mat &_img, int side_rows){

#if 0
     featureTracker->readImage(_img);
    int num_cam = 0;
    int stereo_track = 0;
    vector<set<int>> hash_ids(num_cam);
    for (int i = 0; i < num_cam; i++)
    {
        if (i != 1 || !stereo_track)  // 单目
        {
            auto un_lines = BaseFisheyeFeatureTracker<CvMat>::undistortedLineEndPoints();

            //auto &cur_lines = trackerData.curframe_->vecLine;
            auto &ids = BaseFisheyeFeatureTracker<CvMat>::curframe_->lineID;

            for (unsigned int j = 0; j < ids.size(); j++)
            {

                int p_id = ids[j];
                hash_ids[i].insert(p_id);
                geometry_msgs::Point32 p;
                p.x = un_lines[j].StartPt.x;
                p.y = un_lines[j].StartPt.y;
                p.z = 1;

                feature_lines->points.push_back(p);
                id_of_line.values.push_back(p_id * num_cam + i);
                //std::cout<< "feature tracking id: " <<p_id * num_cam + i<<" "<<p_id<<"\n";
                u_of_endpoint.values.push_back(un_lines[j].EndPt.x);
                v_of_endpoint.values.push_back(un_lines[j].EndPt.y);
                //ROS_ASSERT(inBorder(cur_pts[j]));
            }
        }

    }
#else
    cv::Mat img;
    img = _img.clone();
    /// step 1: line extraction
    std::vector<KeyLine> lsd, keylsd;
#if 0
    Ptr<LSDDetector> lsd_;
    lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
    lsd_->detect( img, lsd, 2, 2 );
#else

     Ptr<line_descriptor::LSDDetectorC> lsd_ = line_descriptor::LSDDetectorC::createLSDDetectorC();
     line_descriptor::LSDDetectorC::LSDOptions opts;
     opts.refine       = 1;     //1     	The way found lines will be refined
     opts.scale        = 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
     opts.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
     opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
     opts.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
     opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
     opts.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
     opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
     double min_line_length = 0.125;
    // int side_rows = fisheys_undists[0].sideImgHeight;
     if (img.rows >  side_rows) {
         opts.min_length = min_line_length * (std::min(img.cols / 3, img.rows / 3));
     }
     else {
         opts.min_length = min_line_length * (std::min(img.cols, img.rows));
     }
     //  std::vector<KeyLine> lsd, keylsd;
     lsd_->detect( img, lsd, 2, 1, opts);
#endif

    Mat lbd_descr, keylbd_descr;
    /// step 2: lbd descriptor
   // TicToc t_lbd;
    Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor( );
    bd_->compute( img, lsd, lbd_descr );

    ///
    for ( int i = 0; i < (int) lsd.size(); i++ )
    {
        if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
        {
            keylsd.push_back( lsd[i] );
            keylbd_descr.push_back( lbd_descr.row( i ) );
        }
    }
#endif
}
    template<class CvMat>
    //FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() {
    vector<Line> BaseFisheyeFeatureTracker<CvMat>::undistortedLineEndPoints2()
    {
        vector<Line> un_lines;
        un_lines = curframe_->vecLine;
#if 0
        float fx = K_.at<float>(0, 0);
        float fy = K_.at<float>(1, 1);
        float cx = K_.at<float>(0, 2);
        float cy = K_.at<float>(1, 2);
#else

#if 0
        float fx = fisheys_undists[0].f_center;
        float fy = fisheys_undists[0].f_center;
        float cx = fisheys_undists[0].imgWidth/2;
        float cy = fisheys_undists[0].imgWidth/2;
        if (focal_x == 0) {
            printf("\nassigning pinhle params... , focal_x = %0.3f \n", focal_x);
            focal_x = fx;
            focal_y = fy;
            center_x = cx;
            center_y = cy;
            cen_width = fisheys_undists[0].imgWidth;
            //cen_height = fisheys_undists[0].imgHeight;
            cen_height = fisheys_undists[0].imgWidth;

        }
        else {
            printf("\npinhle params already assigned... , focal_x = %0.3f \n", focal_x);
        }
#else
        float fx = fisheys_undists[0].f_center;
        float fy = fisheys_undists[0].f_center;
        float cx = fisheys_undists[0].imgWidth/2;
        float cy = fisheys_undists[0].imgWidth/2;
#endif
#endif
        for (unsigned int i = 0; i <curframe_->vecLine.size(); i++)
        {
            un_lines[i].StartPt.x = (curframe_->vecLine[i].StartPt.x - cx)/fx;
            un_lines[i].StartPt.y = (curframe_->vecLine[i].StartPt.y - cy)/fy;
            un_lines[i].EndPt.x = (curframe_->vecLine[i].EndPt.x - cx)/fx;
            un_lines[i].EndPt.y = (curframe_->vecLine[i].EndPt.y - cy)/fy;
        }
        return un_lines;

#if 0
        vector<cv::Point3f> BaseFisheyeFeatureTracker<CvMat>::undistortedPtsSide_(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward) {
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

#endif

    }



    template<class CvMat>
    //FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() {
    vector<Line> BaseFisheyeFeatureTracker<CvMat>::undistortedLineEndPoints()
    {
        vector<Line> un_lines;
        un_lines = curframe_->vecLine;
#if 0
        float fx = K_.at<float>(0, 0);
        float fy = K_.at<float>(1, 1);
        float cx = K_.at<float>(0, 2);
        float cy = K_.at<float>(1, 2);
#else

#if 0
        float fx = fisheys_undists[0].f_center;
        float fy = fisheys_undists[0].f_center;
        float cx = fisheys_undists[0].imgWidth/2;
        float cy = fisheys_undists[0].imgWidth/2;
        if (focal_x == 0) {
            printf("\nassigning pinhle params... , focal_x = %0.3f \n", focal_x);
            focal_x = fx;
            focal_y = fy;
            center_x = cx;
            center_y = cy;
            cen_width = fisheys_undists[0].imgWidth;
            //cen_height = fisheys_undists[0].imgHeight;
            cen_height = fisheys_undists[0].imgWidth;

        }
        else {
            printf("\npinhle params already assigned... , focal_x = %0.3f \n", focal_x);
        }
#else
        float fx = fisheys_undists[0].f_center;
        float fy = fisheys_undists[0].f_center;
        float cx = fisheys_undists[0].imgWidth/2;
        float cy = fisheys_undists[0].imgWidth/2;
#endif
#endif
        for (unsigned int i = 0; i <curframe_->vecLine.size(); i++)
        {
#if 0
            un_lines[i].StartPt.x = (curframe_->vecLine[i].StartPt.x - cx)/fx;
            un_lines[i].StartPt.y = (curframe_->vecLine[i].StartPt.y - cy)/fy;
            un_lines[i].EndPt.x = (curframe_->vecLine[i].EndPt.x - cx)/fx;
            un_lines[i].EndPt.y = (curframe_->vecLine[i].EndPt.y - cy)/fy;
#else
            un_lines[i].StartPt.x = (curframe_->vecLine[i].StartP.x() - cx)/fx;
            un_lines[i].StartPt.y = (curframe_->vecLine[i].StartP.y() - cy)/fy;
            un_lines[i].EndPt.x = (curframe_->vecLine[i].EndP.x() - cx)/fx;
            un_lines[i].EndPt.y = (curframe_->vecLine[i].EndP.y() - cy)/fy;

#endif
        }
        return un_lines;
    }
    template<class CvMat>
    void BaseFisheyeFeatureTracker<CvMat>::NearbyLineTracking(const vector<Line> forw_lines, const vector<Line> cur_lines,
                                                vector<pair<int, int> > &lineMatches) {

        float th = 3.1415926/9;
        float dth = 30 * 30;
        for (size_t i = 0; i < forw_lines.size(); ++i) {
            Line lf = forw_lines.at(i);
            Line best_match;
            size_t best_j = 100000;
            size_t best_i = 100000;
            float grad_err_min_j = 100000;
            float grad_err_min_i = 100000;
            vector<Line> candidate;

            // 从 forw --> cur 查找
            for(size_t j = 0; j < cur_lines.size(); ++j) {
                Line lc = cur_lines.at(j);
                // condition 1
                Point2f d = lf.Center - lc.Center;
                float dist = d.dot(d);
                if( dist > dth) continue;  //
                // condition 2
                float delta_theta1 = fabs(lf.theta - lc.theta);
                float delta_theta2 = 3.1415926 - delta_theta1;
                if( delta_theta1 < th || delta_theta2 < th)
                {
                    //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                    candidate.push_back(lc);
                    //float cost = fabs(lf.image_dx - lc.image_dx) + fabs( lf.image_dy - lc.image_dy) + 0.1 * dist;
                    float cost = fabs(lf.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                    //std::cout<< "line match cost: "<< cost <<" "<< cost - sqrt( dist )<<" "<< sqrt( dist ) <<"\n\n";
                    if(cost < grad_err_min_j)
                    {
                        best_match = lc;
                        grad_err_min_j = cost;
                        best_j = j;
                    }
                }

            }
            if(grad_err_min_j > 50) continue;  // 没找到

            //std::cout<< "!!!!!!!!! minimal cost: "<<grad_err_min_j <<"\n\n";

            // 如果 forw --> cur 找到了 best, 那我们反过来再验证下
            if(best_j < cur_lines.size())
            {
                // 反过来，从 cur --> forw 查找
                Line lc = cur_lines.at(best_j);
                for (int k = 0; k < forw_lines.size(); ++k)
                {
                    Line lk = forw_lines.at(k);

                    // condition 1
                    Point2f d = lk.Center - lc.Center;
                    float dist = d.dot(d);
                    if( dist > dth) continue;  //
                    // condition 2
                    float delta_theta1 = fabs(lk.theta - lc.theta);
                    float delta_theta2 = 3.1415926 - delta_theta1;
                    if( delta_theta1 < th || delta_theta2 < th)
                    {
                        //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                        //candidate.push_back(lk);
                        //float cost = fabs(lk.image_dx - lc.image_dx) + fabs( lk.image_dy - lc.image_dy) + dist;
                        float cost = fabs(lk.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                        if(cost < grad_err_min_i)
                        {
                            grad_err_min_i = cost;
                            best_i = k;
                        }
                    }

                }
            }

            if( grad_err_min_i < 50 && best_i == i){

                //std::cout<< "line match cost: "<<grad_err_min_j<<" "<<grad_err_min_i <<"\n\n";
                lineMatches.push_back(make_pair(best_j,i));
            }
            /*
            vector<Line> l;
            l.push_back(lf);
            vector<Line> best;
            best.push_back(best_match);
            visualizeLineTrackCandidate(l,forwframe_->img,"forwframe_");
            visualizeLineTrackCandidate(best,curframe_->img,"curframe_best");
            visualizeLineTrackCandidate(candidate,curframe_->img,"curframe_");
            cv::waitKey(0);
            */
        }

    }
    //static int decideFaceId(Eigen::Vector2d a, vector<FisheyeUndist> &fisheys_undists)
    //static int decideFaceId(Eigen::Vector2d a, int topCol, int topRow, int sideCol, int sideRow)
  //  static double decideFaceId(Point2f a, int topCol, int topRow, int sideCol, int sideRow)
    static pair<double, Vector2d>  decideFaceId(Point2f a, int topCol, int topRow, int sideCol, int sideRow)
    {
//            int topCol = fisheys_undists[0].top_size.width;
//    int topRow = top_size.height;
//    //int sideCol = side_size.width;
//    //int sideRow = side_size.height;
//    int sideCol = side_size_single.width;
//    int sideRow = side_size_single.height;


       // int side_cols = fisheys_undists[0].imgWidth;
       // int side_rows = fisheys_undists[0].sideImgHeight;
        pair<double, Vector2d> ret;
        double side_pos_id;
        //std::cout<<"vector2d: "<<a.transpose()<<", [a.x() a.y()]: ["<<a.x()<<" "<<a.y()<<"]"<<std::endl;
        if (a.x >= sideRow & a.x<=sideRow+topCol-1
            &a.y >= sideRow & a.y<=sideRow+topCol-1){
            side_pos_id = 0;
            Eigen::Vector2d pt(a.x - sideRow, a.y - sideRow);
            ret.second = pt;
            //continue;
        }
        /*if(!is_downward){
            up_top_ids_side_part.push_back(i);
        }
        else{
            down_top_ids_side_part.push_back(i);
        }*/
        if (a.x >= sideRow & a.x<=sideRow+topCol-1
            &a.y >= sideRow+topCol & a.y<=2*sideRow+topCol-1){
            side_pos_id = 1;
            Eigen::Vector2d pt(a.x - sideRow, a.y - sideRow - topCol);
            ret.second = pt;
        }
        if (a.x >= sideRow+topCol & a.x<=2*sideRow+topCol-1
            &a.y >= sideRow & a.y<=sideRow+topCol-1){
            side_pos_id = 2;
            Eigen::Vector2d pt((topCol-1-(a.y - sideRow)),(a.x - (sideRow + topCol)));
            ret.second = pt;
        }
        if (a.x >= sideRow & a.x<=sideRow+topCol-1
            &a.y >= 0 & a.y<=sideRow-1){
            side_pos_id = 3;
            Eigen::Vector2d pt((topCol-1-(a.x-sideRow)),(sideRow-1-a.y));
            ret.second = pt;
        }
        if (a.x >= 0 & a.x<=sideRow-1
            &a.y >= sideRow & a.y<=sideRow+topCol-1){
            side_pos_id = 4;
            Eigen::Vector2d pt((a.y - sideRow),(sideRow-1-a.x));
            ret.second = pt;
        }

        ret.first = side_pos_id;
   //     ret.second = pt;
// return side_pos_id;
        return ret;
    }
    static cv::Mat setMaskLine(int row, int col, vector<FisheyeUndist> &fisheys_undists) {

//    int topCol = top_size.width;
//    int topRow = top_size.height;
//    //int sideCol = side_size.width;
//    //int sideRow = side_size.height;
//    int sideCol = side_size_single.width;
//    int sideRow = side_size_single.height;

//assert(fxx > 0);


        double centerFOV = CENTER_FOV * DEG_TO_RAD;
        double sideVerticalFOV = (FISHEYE_FOV - CENTER_FOV) * DEG_TO_RAD / 2;
        double rot_angle = (centerFOV + sideVerticalFOV) / 2;  // M_PI/2;//

        /// side_image中剩余的有效实际fov
        double side_fov_valid = (FISHEYE_FOV_ACTUAL - CENTER_FOV) / 2 * DEG_TO_RAD;

        // std::cout<<"===== side_fov_valid: "<< side_fov_valid<<std::endl;
        // std::cout<<"===== FISHEYE_FOV_ACTUAL: "<< FISHEYE_FOV_ACTUAL<<std::endl;
        // std::cout<<"===== CENTER_FOV: "<< CENTER_FOV<<std::endl;

        double side_fov_valid_ratio = side_fov_valid / sideVerticalFOV;

        Eigen::Matrix3d intrMat_side;
        intrMat_side << fisheys_undists[0].f_center, 0.0, fisheys_undists[0].imgWidth / 2,
                0.0, fisheys_undists[0].f_center, fisheys_undists[0].imgWidth / 2,
                0.0, 0.0, 1.0;

        int side_cols = fisheys_undists[0].imgWidth;
        int side_rows = fisheys_undists[0].sideImgHeight;
        Eigen::Vector3d ctrlPt1;
        ctrlPt1(0) = side_cols / 2.0;
        ctrlPt1(1) = 0.0;
        ctrlPt1(2) = 1.0;
        Eigen::Vector3d ctrlPt_un1 = intrMat_side.inverse() * ctrlPt1;
        Eigen::Vector3d ctrlPt_un2 = Eigen::AngleAxis<double>(-side_fov_valid, Eigen::Vector3d(1, 0, 0)) * ctrlPt_un1;
        ctrlPt_un2(0) = ctrlPt_un2(0) / ctrlPt_un2(2);
        ctrlPt_un2(1) = ctrlPt_un2(1) / ctrlPt_un2(2);
        ctrlPt_un2(2) = 1;

        //std::cout<<"side_rows: "<<side_rows<<" side_cols: "<< side_cols<<std::endl;
        //std::cout<<"intrMat_side:\n"<<intrMat_side<<std::endl;
        //std::cout<<"ctrlPt_un1: \n"<<ctrlPt_un1.transpose()<<std::endl;
        //std::cout<<"ctrlPt_un2: \n"<<ctrlPt_un2.transpose()<<std::endl;


        Eigen::Vector3d ctrlPt2 = intrMat_side * ctrlPt_un2;

        int valid_y = (int) (ctrlPt2(1) / ctrlPt2(2));
        if (valid_y > side_rows) {
            valid_y = side_rows;
        }

        cv::Mat mask_full;
        cv::Mat mask_full_eroded;

#if 0
        cv::Mat side_mask = cv::Mat(side_rows, side_cols, CV_8UC1, cv::Scalar(0));
        cv::Mat mask_full = cv::Mat(row, col,  CV_8UC1, cv::Scalar(0));
        cv::Mat  center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(255));
        cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1, cv::Scalar(255));
        cv::Mat side_mask_valid1, side_mask_valid2, side_mask_valid3, side_mask_valid4;
#else
        if (row > side_rows) {
//            cv::Mat side_mask = cv::Mat(side_rows, side_cols, CV_8UC1);
//            side_mask.setTo(0);
//            cv::Mat mask_full = cv::Mat(row, col, CV_8UC1);
//            mask_full.setTo(0);
//            cv::Mat center_mask = cv::Mat(side_cols, side_cols, CV_8UC1);
//            center_mask.setTo(255);
//            cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1);
//            side_mask_valid.setTo(255);

int blank_pixels = 5; // 0; // 5;
            cv::Mat side_mask = cv::Mat(side_rows, side_cols, CV_8UC1, cv::Scalar(0));
            mask_full = cv::Mat(row, col, CV_8UC1, cv::Scalar(0));
#if 0
            cv::Mat center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(255));
            cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1, cv::Scalar(255));
#else
            cv::Mat center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(0));
            center_mask(cv::Rect(blank_pixels, blank_pixels, side_cols-2*blank_pixels, side_cols-2*blank_pixels)) = cv::Scalar(255);

            cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1, cv::Scalar(0));
            side_mask_valid(cv::Rect(blank_pixels, blank_pixels, side_cols-2*blank_pixels, valid_y-2*blank_pixels)) = cv::Scalar(255);

            //cv::Rect(0,0,10,10);

#endif
            //cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1, cv::Scalar(255));
            //cv::Mat side_mask_valid1, side_mask_valid2, side_mask_valid3, side_mask_valid4;



            center_mask.copyTo(mask_full(cv::Rect(side_rows, side_rows, side_cols, side_cols)));
            side_mask_valid.copyTo(side_mask(cv::Rect(0, 0, side_cols, valid_y)));

            //side_mask_valid.copyTo(mask_full(cv::Rect(side_rows, side_rows + side_cols, side_cols, side_rows)));
            side_mask.copyTo(mask_full(cv::Rect(side_rows, side_rows + side_cols, side_cols, side_rows)));


            cv::Mat temp2 = side_mask; // side_mask_valid;
            cv::Mat temp22;
            cv::rotate(temp2, temp22, cv::ROTATE_90_COUNTERCLOCKWISE);
            temp22.copyTo(mask_full(cv::Rect(side_rows + side_cols, side_rows, side_rows, side_cols)));

            cv::Mat temp3 = side_mask; // side_mask_valid;
            cv::Mat temp33;
            cv::rotate(temp3, temp33, cv::ROTATE_180);
            temp33.copyTo(mask_full(cv::Rect(side_rows, 0, side_cols, side_rows)));

            cv::Mat temp4 = side_mask; // side_mask_valid;
            cv::Mat temp44;
            cv::rotate(temp4, temp44, cv::ROTATE_90_CLOCKWISE);
            temp44.copyTo(mask_full(cv::Rect(0, side_rows, side_rows, side_cols)));




            /// mask_full1(cv::Rect(side_rows, side_rows, side_cols, side_cols)).setTo(0); // = cv::Scalar(0); // 255;
            /// mask_full1(cv::Rect(0, 0, 1, 2)).setTo(0); // = cv::Scalar(0); // 255;

#if 0
            int erodeSize = 20; // 40; // 20; //50;
            cv::Mat erode_element = getStructuringElement(cv::MORPH_RECT, cv::Size(erodeSize, erodeSize));
            // cv::Mat mask_full_eroded;
            cv::erode(mask_full, mask_full_eroded, erode_element);
#else
            mask_full_eroded = mask_full;
#endif

#if 0
            cv::Mat mask_full2;
            cv::hconcat(mask_full, mask_full_eroded, mask_full2);
             cv::imshow("raw cube mask and eroded cube mask", mask_full2);
             cv::waitKey(2);
#endif
            // cv::Mat side_mask_valid1, side_mask_valid2, side_mask_valid3, side_mask_valid4;
        }
#endif

        //mask_full(cv::Rect(side_rows, side_rows, side_cols, side_cols)) = cv::Scalar(0); // 255;
        //cv::Mat mask_full1 = mask_full.clone();
        //mask_full1(cv::Rect(side_rows, side_rows, side_cols, side_cols)).setTo(0); // = cv::Scalar(0); // 255;
        //mask_full1(cv::Rect(0, 0, 1, 2)).setTo(0); // = cv::Scalar(0); // 255;
        //center_mask.copyTo(mask_full(cv::Rect(side_rows, side_rows, side_cols, side_cols)));
#if 0

        side_mask_valid.copyTo(side_mask(cv::Rect(0, 0, side_cols, valid_y)));

        side_mask_valid.copyTo(mask_full(cv::Rect(side_rows, side_rows + side_cols, side_cols, side_rows)));



        cv::Mat temp2 = side_mask_valid;
        cv::Mat temp22;
        cv::rotate(temp2, temp22, cv::ROTATE_90_COUNTERCLOCKWISE);
        temp22.copyTo(mask_full(cv::Rect(side_rows + side_cols, side_rows, side_rows, side_cols)));

        cv::Mat temp3 = side_mask_valid;
        cv::Mat temp33;
        cv::rotate(temp3, temp33, cv::ROTATE_180);
        temp33.copyTo(mask_full(cv::Rect(side_rows, 0, side_cols, side_rows)));

        cv::Mat temp4 = side_mask_valid;
        cv::Mat temp44;
        cv::rotate(temp4, temp44, cv::ROTATE_90_CLOCKWISE);
        temp44.copyTo(mask_full(cv::Rect(0, side_rows, side_rows, side_cols)));


        cv::imshow("raw cube mask", mask_full);
        cv::waitKey(2);
#endif

#if 0
        double centerFOV;
double sideVerticalFOV;
double rot_angle;
float side_fov_valid;
float side_fov_valid_ratio;
int side_cols = cxx * 2;

   // int cols = arr[1].cols;
   // int rows = arr[1].rows;

std::cout<<"side_rows: "<<side_rows<<" side_cols: "<< side_cols<<std::endl;
 //   arr[i].copyTo(NewImg(cv::Rect(cols * (i - 1), 0, cols, rows)));

    // cv::Mat side_mask, side_mask_valid, side_mask_valid1, side_mask_valid2, side_mask_valid3, side_mask_valid4;
    //cv::Mat side_mask,
    //side_mask_valid; //, side_mask_valid1, side_mask_valid2, side_mask_valid3, side_mask_valid4;
    //cv::Mat mask1 = cv::Mat(row, col, CV_8UC1, cv::Scalar(0));

    //cv::Mat side_mask = cv::Mat(side_rows, side_cols, CV_8UC1, cv::Scalar(0));
//    cv::Mat side_mask_valid =
    //cv::Mat mask_full,
    //cv::Mat mask_full = cv::Mat(row, col,  CV_8UC1, cv::Scalar(0));
    //cv::Mat center_mask;
   // cv::Mat  center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(255));



    Eigen::Matrix3d intrMat_side;
    intrMat_side<< fxx,0.0,cxx,
                    0.0,fyy,cyy,
                    0.0,0.0,1.0;
   //  std::cout<<"intrMat_side:\n"<<intrMat_side<<std::endl;
//if (USE_NEW) {
//if (CUBE_MAP) {
if (false) {
    centerFOV = CENTER_FOV * DEG_TO_RAD;
    sideVerticalFOV = (FISHEYE_FOV - CENTER_FOV) * DEG_TO_RAD /2;
    rot_angle = (centerFOV + sideVerticalFOV)/2;  // M_PI/2;//

    /// side_image中剩余的有效实际fov
    side_fov_valid = (FISHEYE_FOV_ACTUAL - CENTER_FOV) / 2 * DEG_TO_RAD;
    side_fov_valid_ratio = side_fov_valid / sideVerticalFOV;
    //side_mask = cv::Mat(side_rows, side_cols, CV_8UC1, cv::Scalar(0));
    //mask_full = cv::Mat(row, col,  CV_8UC1, cv::Scalar(0));

    //center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(255));
    cv::Mat  center_mask = cv::Mat(side_cols, side_cols, CV_8UC1, cv::Scalar(255));
    cv::Mat side_mask = cv::Mat(side_rows, side_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat mask_full = cv::Mat(row, col,  CV_8UC1, cv::Scalar(0));


    center_mask.copyTo(mask_full(cv::Rect(side_rows, side_rows, side_cols, side_cols)));

    side_mask.setTo(0);
    Eigen::Vector3d ctrlPt1;
    ctrlPt1(0) = side_cols/2;
    ctrlPt1(1) = 0.0;
    ctrlPt1(2) = 1.0;
    Eigen::Vector3d ctrlPt_un1 = intrMat_side.inverse() * ctrlPt1;
    Eigen::Vector3d ctrlPt_un2 = Eigen::AngleAxis<double>(-side_fov_valid , Eigen::Vector3d(1, 0, 0)) * ctrlPt_un1;
    //std::cout<<"ctrlPt_un1: \n"<<ctrlPt_un1.transpose()<<std::endl;
    //std::cout<<"ctrlPt_un2: \n"<<ctrlPt_un2.transpose()<<std::endl;
    Eigen::Vector3d ctrlPt2 = intrMat_side * ctrlPt_un2;

    int valid_y = ctrlPt2(1) / ctrlPt2(2);
    if(valid_y > side_rows){
        valid_y = side_rows;
    }
    cv::Mat side_mask_valid = cv::Mat(valid_y, side_cols, CV_8UC1, cv::Scalar(255));
    side_mask_valid.copyTo(side_mask(cv::Rect(0, 0, side_cols, valid_y)));

    side_mask_valid.copyTo(mask_full(cv::Rect(side_rows, side_rows + side_cols, side_cols, side_rows)));



    cv::Mat temp2 = side_mask_valid;
    cv::Mat temp22;
    cv::rotate(temp2, temp22, cv::ROTATE_90_COUNTERCLOCKWISE);
    temp22.copyTo(mask_full(cv::Rect(side_rows + side_cols, side_rows, side_rows, side_cols)));

    cv::Mat temp3 = side_mask_valid;
    cv::Mat temp33;
    cv::rotate(temp3, temp33, cv::ROTATE_180);
    temp33.copyTo(mask_full(cv::Rect(side_rows, 0, side_cols, side_rows)));

    cv::Mat temp4 = side_mask_valid;
    cv::Mat temp44;
    cv::rotate(temp4, temp44, cv::ROTATE_90_CLOCKWISE);
    temp44.copyTo(mask_full(cv::Rect(0, side_rows, side_rows, side_cols)));


    cv::imshow("raw cube mask", mask_full);
    cv::waitKey(2);

    //side_mask_valid1 = side_mask_valid.clone();
//    if (side_fov_valid_ratio <= 0.5)
//    {
//
//    }
//    else
//    {
//
//    }


}
#else

#endif

        cv::Mat mask;
        if (row == side_rows || side_cols == col) {
            mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
            //cv::imshow("pinhole mask", mask);
            //cv::waitKey(2);
        } else {
            // mask = mask_full;
            mask = mask_full_eroded;
            //cv::imshow("cube mask", mask);
            //cv::waitKey(2);
        }

#if 0
        cv::imshow("detect mask", mask);
        cv::waitKey(2);
#endif

        // prefer to keep features that are tracked for long time
#if 0
        vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

//    for (unsigned int i = 0; i < cur_pts.size(); i++)
//        cnt_pts_id.push_back(make_pair(track_up_top_cnt[i], make_pair(cur_pts[i], ids[i])));
//
//    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
//    {
//        return a.first > b.first;
//    });

//    cur_pts.clear();
//    ids.clear();
//    track_cnt.clear();

        for (int i = 0;i <  cur_pts.size(); i++)
        {
            if (mask.at<uchar>(cur_pts[i]) == 255)
            {
//            cur_pts.push_back(it.second.first);
//            ids.push_back(it.second.second);
//            track_cnt.push_back(it.first);
                cv::circle(mask, cur_pts[i], MIN_DIST, 0, -1);
            }
        }
#endif
        return mask;
    }
    template<class CvMat>
    //void BaseFisheyeFeatureTracker<CvMat>::readImage(const cv::Mat &_img, const Mat& mask = Mat())
    void BaseFisheyeFeatureTracker<CvMat>::readImage(const cv::Mat &_img, const Mat& mask)
    {
        cv::Mat img;
        TicToc t_p;
        frame_cnt++;


        int topCol = top_size.width;
        int topRow = top_size.height;
        //int sideCol = side_size.width;
        //int sideRow = side_size.height;
        int sideCol = side_size_single.width;
        int sideRow = side_size_single.height;


        //cv::remap(_img, img, undist_map1_, undist_map2_, CV_INTER_LINEAR);
        img = _img.clone();
//    cv::imshow("lineimg",img);
//    cv::waitKey(1);
        //ROS_INFO("undistortImage costs: %fms", t_p.toc());

        /// if (EQUALIZE)   // 直方图均衡化
        if (false)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(img, img);
        }

        bool first_img = false;
        if (forwframe_ == nullptr) // 系统初始化的第一帧图像
        {
            forwframe_.reset(new FrameLines);
            curframe_.reset(new FrameLines);
            forwframe_->img = img;
            curframe_->img = img;
            first_img = true;
        }
        else
        {
            forwframe_.reset(new FrameLines);  // 初始化一个新的帧
            forwframe_->img = img;
        }

        // step 1: line extraction
        TicToc t_li;
        std::vector<KeyLine> lsd_bak, lsd, keylsd;

        double min_length;
        double min_line_length = 0.125;        //     0.25;      //    0.125;

        int side_rows = fisheys_undists[0].sideImgHeight;
        if (img.rows >  side_rows) {
            min_length = min_line_length * (std::min(img.cols / 3, img.rows / 3));
        }
        else {
            min_length = min_line_length * (std::min(img.cols, img.rows));
        }
        /////////////////////////
#if 0
        Ptr<LSDDetector> lsd_;
        //Ptr<LineSegmentDetector> lsd_;
        double scale        = 0.8; // 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
        double sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
        double quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
        double ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
        double log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
        double density_th   = 0.6;  // 0.75; // 0.6; // 0.75; // 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
        double n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.

        //double min_line_length = 0.125;  // 0.25; // 0.125; // 0.25; // 0.125; // 0.25; // 0.3; // 0.25; // 0.3; // 0.125; // 0.3; // 0.4; // 0.3; // 0.2; //  0.3; // 0.4; // 0.25; // 0.125;
#if 1
        // lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector(LSD_REFINE_STD);
        lsd_ = cv::line_descriptor::LSDDetector::createLSDDetector();
#else
        lsd_ = createLineSegmentDetector(LSD_REFINE_STD, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins);
#endif
        #if 0
        lsd_->detect( img, lsd, 2, 2 );
#else
        // lsd_->detect( img, lsd, 2, 2 ,mask);
        // lsd_->detect( img, lsd_bak, 2, 2 ,mask);
        lsd_->detect( img, lsd_bak, 2, 2);
        cv::Mat mask_;
        cv::cvtColor(mask, mask_, CV_GRAY2BGR);

        for ( int i = 0; i < (int) lsd_bak.size(); i++ ) {
            //if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
            /// roger @ 20220215
            Point2f start111 = lsd_bak[i].getStartPoint();
            Point2f end111 = lsd_bak[i].getEndPoint();
            double face_id_start11;
            double face_id_end11 ;

           pair<double, Vector2d> ret1 = decideFaceId(start111, topCol, topRow, sideCol, sideRow);
           pair<double, Vector2d> ret2 = decideFaceId(end111, topCol, topRow, sideCol, sideRow);
           face_id_start11 = ret1.first;
           face_id_end11 = ret2.first;
            uchar value1 = mask.at<uchar>(int(start111.y), int(start111.x));
            uchar value2 = mask.at<uchar>(int(end111.y), int(end111.x));
            bool valid_area = (int(value1) == (255)) && (int(value2) == (255));
            //bool valid_area = (int(value1) == (255)) && (int(value2) == (255));
          // bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 == 0); //  && (int(face_id_start1) == int(face_id_end2)); //  () () ();
           bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 >= 0);
            if (no_cross_face11 && valid_area) {
                lsd.push_back(lsd_bak[i]);
            }

        }
        int valid_size = lsd.size();
//         lsd.clear();
        //Mat keylbd_descr, lbd_descr_bak;


                Mat lbd_descr_bak, keylbd_descr;
        // step 2: lbd descriptor
        TicToc t_lbd;
        Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor( );
        bd_->compute( img, lsd_bak, lbd_descr_bak );
        lsd.clear();

        //   printf("lsd_bak.size: %d, lbd_descr_bak.rows %d, lbd_descr_bak.cols: %d\n", lsd_bak.size(), lbd_descr_bak.rows, lbd_descr_bak.cols);
        int ind = 0;
        // Mat lbd_descr(valid_size, 32, t)
//         Mat lbd_descr = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3);
        // Mat lbd_descr = cv::Mat(cv::Size( lbd_descr_bak.cols, lbd_descr_bak.rows), lbd_descr_bak.type());
        Mat lbd_descr = cv::Mat(cv::Size( lbd_descr_bak.cols, valid_size), lbd_descr_bak.type());
        //   printf("lsd_bak.size: %d, lbd_descr.rows %d, lbd_descr.cols: %d\n", lsd_bak.size(), lbd_descr.rows, lbd_descr.cols);
        // lbd_descr.resize(valid_size, 32);
        for ( int i = 0; i < (int) lsd_bak.size(); i++ ) {
            //if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
            /// roger @ 20220215
            Point2f start111 = lsd_bak[i].getStartPoint();
            Point2f end111 = lsd_bak[i].getEndPoint();
            double face_id_start11;
            double face_id_end11;

            pair<double, Vector2d> ret1 = decideFaceId(start111, topCol, topRow, sideCol, sideRow);
            pair<double, Vector2d> ret2 = decideFaceId(end111, topCol, topRow, sideCol, sideRow);
            face_id_start11 = ret1.first;
            face_id_end11 = ret2.first;
            uchar value1 = mask.at<uchar>(int(start111.y), int(start111.x));
            uchar value2 = mask.at<uchar>(int(end111.y), int(end111.x));
            bool valid_area = (int(value1) == (255)) && (int(value2) == (255));

            // bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 == 0); //  && (int(face_id_start1) == int(face_id_end2)); //  () () ();
            bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 >= 0);
            if (no_cross_face11 && valid_area) {
                lsd.push_back(lsd_bak[i]);
                // lbd_descr.push_back(lbd_descr_bak[i]);
                lbd_descr_bak.row(i).copyTo(lbd_descr.row(ind));
                ind++;
            }
        }

#endif
#else
        Ptr<line_descriptor::LSDDetectorC> lsd_ = line_descriptor::LSDDetectorC::createLSDDetectorC();
        line_descriptor::LSDDetectorC::LSDOptions opts;
        opts.refine       = 1;     //1     	The way found lines will be refined
        opts.scale        = 0.8; // 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
        opts.sigma_scale  = 0.4; // 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
        opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
        opts.ang_th       = 22.5;     //  75.0;    //  45.0;    //     11.25;    //    45.0; // 22.5;	//22.5	Gradient angle tolerance in degrees
        opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
        opts.density_th   = 0.6;   // 0.3;    //   0.4;  //  0.6; // 0.8;   // 0.6;  // 0.75; // 0.6; // 0.75; // 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
        opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
        // double min_line_length = 0.125;  // 0.25; // 0.125; // 0.25; // 0.125; // 0.25; // 0.3; // 0.25; // 0.3; // 0.125; // 0.3; // 0.4; // 0.3; // 0.2; //  0.3; // 0.4; // 0.25; // 0.125;
        side_rows = fisheys_undists[0].sideImgHeight;
        if (img.rows >  side_rows) {
            opts.min_length = min_line_length * (std::min(img.cols / 3, img.rows / 3));
        }
        else {
            opts.min_length = min_line_length * (std::min(img.cols, img.rows));
        }
       //  std::vector<KeyLine> lsd, keylsd;
#if 0
        lsd_->detect( img, lsd, 2, 1, opts, mask);
#else
        // lsd_->detect( img, lsd, 2, 1, opts);
        lsd_->detect( img, lsd_bak, 2, 1, opts);
        cv::Mat mask_;
        //cv::cvtColor(mask, mask_, CV_BGR2GRAY);
        cv::cvtColor(mask, mask_, CV_GRAY2BGR);
        for ( int i = 0; i < (int) lsd_bak.size(); i++ ) {
            //if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
            /// roger @ 20220215
            Point2f start111 = lsd_bak[i].getStartPoint();
            Point2f end111 = lsd_bak[i].getEndPoint();
            double face_id_start11;
            double face_id_end11 ;

            pair<double, Vector2d> ret1 = decideFaceId(start111, topCol, topRow, sideCol, sideRow);
            pair<double, Vector2d> ret2 = decideFaceId(end111, topCol, topRow, sideCol, sideRow);
            face_id_start11 = ret1.first;
            face_id_end11 = ret2.first;
#if 0
            Vec3b intensity1 = mask_.at<Vec3b>(300, 300);
            uchar blue = intensity1.val[0];
            uchar green = intensity1.val[1];
            uchar red = intensity1.val[2];
            cv::Scalar value1 = mask.at<uchar>(int(ret1.second.y()), int(ret1.second.x()));
            //int value11 = mask.at<int>(int(ret1.second.y()), int(ret1.second.x()));
            //uchar value2 = mask.at<uchar>(int(ret2.second.y()), int(ret2.second.x()));
            cv::Scalar value2 = mask.at<uchar>(int(ret2.second.y()), int(ret2.second.x()));
            Scalar intensity = mask.at<uchar>(Point(ret2.second.x(), ret2.second.y()));
            //int value22 = mask.at<int>(int(ret1.second.y()), int(ret1.second.x()));
            //printf("values: [%d, %d]\n",value1, value2);
            printf("values: [%d]\n",int(blue));
            std::cout<<"value: "<<value1<<std::endl;
            std::cout<<"blue: "<<blue<<std::endl;
            std::cout<<"==== intensity: "<<intensity<<std::endl;
            std::cout<<"-------------------value: "<<mask.at<cv::Scalar>(300,300)<<std::endl;
            std::cout<<"----------------------- -----value: "<<ret2.second.transpose()<<std::endl;
#endif

#if 1
//            uchar value1 = mask.at<uchar>(int(ret1.second.y()), int(ret1.second.x()));
//            uchar value2 = mask.at<uchar>(int(ret2.second.y()), int(ret2.second.x()));

            uchar value1 = mask.at<uchar>(int(start111.y), int(start111.x));
            uchar value2 = mask.at<uchar>(int(end111.y), int(end111.x));
           // printf("-------- values: [%d %d]\n", int(value1), int(value2));
#else
            Vec3b intensity1 = mask_.at<Vec3b>(int(ret1.second.y()), int(ret1.second.x()));
            uchar value1 = intensity1.val[0];
            Vec3b intensity2 = mask_.at<Vec3b>(int(ret2.second.y()), int(ret2.second.x()));
            uchar value2 = intensity2.val[0];
            printf("-------- values: [%d %d]\n", int(value1), int(value2));
            printf("-------- coord[x y]: [%d %d]\n", int(ret1.second.x()), int(ret1.second.y()));
#endif

#if 0
            int x_coord = int(ret1.second.x());
            int y_coord = int(ret1.second.y());
            // Vec3b intensity11 = mask_.at<Vec3b>(x_coord, y_coord);
            Vec3b intensity11 = mask_.at<Vec3b>(int(start111.y), int(start111.x));
            uchar blue = intensity11.val[0];
            printf("values: [%d]\n",int(blue));
#endif
            // bool valid_area = (value1 == cv::Scalar(255)) && (value2 == cv::Scalar(255));
            bool valid_area = (int(value1) == (255)) && (int(value2) == (255));

            // bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 == 0); //  && (int(face_id_start1) == int(face_id_end2)); //  () () ();
            bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 >= 0);
            if (no_cross_face11 && valid_area) {
                lsd.push_back(lsd_bak[i]);
            }

        }
        printf("lsd.size: %d\n",lsd.size());
int valid_size = lsd.size();

        Mat keylbd_descr, lbd_descr_bak;
        // step 2: lbd descriptor
        TicToc t_lbd;
        Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor( );
        bd_->compute( img, lsd_bak, lbd_descr_bak );
        lsd.clear();

     //   printf("lsd_bak.size: %d, lbd_descr_bak.rows %d, lbd_descr_bak.cols: %d\n", lsd_bak.size(), lbd_descr_bak.rows, lbd_descr_bak.cols);
        int ind = 0;
        // Mat lbd_descr(valid_size, 32, t)
//         Mat lbd_descr = cv::Mat(cv::Size(WIDTH, WIDTH), CV_8UC3);
        // Mat lbd_descr = cv::Mat(cv::Size( lbd_descr_bak.cols, lbd_descr_bak.rows), lbd_descr_bak.type());
        Mat lbd_descr = cv::Mat(cv::Size( lbd_descr_bak.cols, valid_size), lbd_descr_bak.type());
     //   printf("lsd_bak.size: %d, lbd_descr.rows %d, lbd_descr.cols: %d\n", lsd_bak.size(), lbd_descr.rows, lbd_descr.cols);
       // lbd_descr.resize(valid_size, 32);
        for ( int i = 0; i < (int) lsd_bak.size(); i++ ) {
            //if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
            /// roger @ 20220215
            Point2f start111 = lsd_bak[i].getStartPoint();
            Point2f end111 = lsd_bak[i].getEndPoint();
            double face_id_start11;
            double face_id_end11;

            pair<double, Vector2d> ret1 = decideFaceId(start111, topCol, topRow, sideCol, sideRow);
            pair<double, Vector2d> ret2 = decideFaceId(end111, topCol, topRow, sideCol, sideRow);
            face_id_start11 = ret1.first;
            face_id_end11 = ret2.first;
            uchar value1 = mask.at<uchar>(int(start111.y), int(start111.x));
            uchar value2 = mask.at<uchar>(int(end111.y), int(end111.x));
            bool valid_area = (int(value1) == (255)) && (int(value2) == (255));

            // bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 == 0); //  && (int(face_id_start1) == int(face_id_end2)); //  () () ();
            bool no_cross_face11 = (int(face_id_start11) == int(face_id_end11)) && (face_id_start11 >= 0);
            if (no_cross_face11 && valid_area) {
                lsd.push_back(lsd_bak[i]);
               // lbd_descr.push_back(lbd_descr_bak[i]);
                lbd_descr_bak.row(i).copyTo(lbd_descr.row(ind));
                ind++;
            }
        }
#endif
        // lsd_->detect( img, lsd, 3, 1, opts);

#endif
          // Line segments shorter than that are rejected



          assert(lbd_descr.rows == lsd.size());
        sum_time += t_li.toc();
//    ROS_INFO("line detect costs: %fms", t_li.toc());


//        Mat lbd_descr, keylbd_descr;
//        // step 2: lbd descriptor
//        TicToc t_lbd;
//        Ptr<BinaryDescriptor> bd_ = BinaryDescriptor::createBinaryDescriptor( );
//        bd_->compute( img, lsd, lbd_descr );

//////////////////////////




        for ( int i = 0; i < (int) lsd.size(); i++ )
        {
            //if( lsd[i].octave == 0 && lsd[i].lineLength >= 30)
            /// roger @ 20220215
            Point2f start11 = lsd[i].getStartPoint();
            Point2f end11 = lsd[i].getEndPoint();

            //double  face_id_start1 = decideFaceId(start11,  topCol,  topRow,  sideCol,  sideRow);
            //double  face_id_end1 = decideFaceId(end11,  topCol,  topRow,  sideCol,  sideRow);
            double face_id_start1;
            double face_id_end1 ;

            pair<double, Vector2d> ret11 = decideFaceId(start11, topCol, topRow, sideCol, sideRow);
            pair<double, Vector2d> ret22 = decideFaceId(end11, topCol, topRow, sideCol, sideRow);
            face_id_start1 = ret11.first;
            face_id_end1 = ret22.first;

            bool no_cross_face1 = (int(face_id_start1) == int(face_id_end1)); //  && (int(face_id_start1) == int(face_id_end2)); //  () () ();

            if( (lsd[i].octave == 0 && lsd[i].lineLength >= min_length) && no_cross_face1)
            {
                keylsd.push_back( lsd[i] );
                keylbd_descr.push_back( lbd_descr.row( i ) );
            }
        }
//    ROS_INFO("lbd_descr detect costs: %fms", keylsd.size() * t_lbd.toc() / lsd.size() );
        sum_time += keylsd.size() * t_lbd.toc() / lsd.size();
///////////////

        forwframe_->keylsd = keylsd;
        forwframe_->lbd_descr = keylbd_descr;

        for (size_t i = 0; i < forwframe_->keylsd.size(); ++i) {
            if(first_img)
                forwframe_->lineID.push_back(allfeature_cnt++);
            else
                forwframe_->lineID.push_back(-1);   // give a negative id
        }





        if(curframe_->keylsd.size() > 0)
        {

            /* compute matches */
            TicToc t_match;
            std::vector<DMatch> lsd_matches;
            Ptr<BinaryDescriptorMatcher> bdm_;
            bdm_ = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
            bdm_->match(forwframe_->lbd_descr, curframe_->lbd_descr, lsd_matches);
//        ROS_INFO("lbd_macht costs: %fms", t_match.toc());
            sum_time += t_match.toc();
            mean_time = sum_time/frame_cnt;
            ROS_INFO("line feature tracker mean costs: %fms", mean_time);

            /* select best matches */
            std::vector<DMatch> good_matches;
            std::vector<KeyLine> good_Keylines;
            good_matches.clear();
            for ( int i = 0; i < (int) lsd_matches.size(); i++ )
            {
                // printf("lsd_matches[%d].distance: %f \n", i, lsd_matches[i].distance);
                if( lsd_matches[i].distance < MATCHES_DIST_THRESHOLD ){

                    DMatch mt = lsd_matches[i];
                    KeyLine line1 =  forwframe_->keylsd[mt.queryIdx] ;
                    KeyLine line2 =  curframe_->keylsd[mt.trainIdx] ;
#if 0
                    Point2f serr = line1.getStartPoint() - line2.getStartPoint();
                    Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
#endif
                    Point2f start1, end1, start2, end2;
                    start1 = line1.getStartPoint();
                    start2 = line2.getStartPoint();
                    end1 = line1.getEndPoint();
                    end2 = line2.getEndPoint();

                    Point2f serr = start1 - start2;
                    Point2f eerr = end1 - end2;

#if 0
                    Eigen::Vector2d pts1, pte1, pts2, pte2;
                    pts1(0) = start1.x;
                    pts1(1) = start1.y;
                    pte1(0) = end1.x;
                    pte1(1) = end1.y;

                    pts2(0) = start2.x;
                    pts2(1) = start2.y;
                    pte2(0) = end2.x;
                    pte2(1) = end2.y;
#endif
 //                   double  face_id_start1 = decideFaceId(start1,  topCol,  topRow,  sideCol,  sideRow);
   //                 double  face_id_end1 = decideFaceId(end1,  topCol,  topRow,  sideCol,  sideRow);
     //               double  face_id_start2 = decideFaceId(start2,  topCol,  topRow,  sideCol,  sideRow);
       //             double  face_id_end2 = decideFaceId(end2,  topCol,  topRow,  sideCol,  sideRow);



                    double face_id_start1, face_id_start2;
                    double face_id_end1, face_id_end2 ;

                    pair<double, Vector2d> ret1 = decideFaceId(start1, topCol, topRow, sideCol, sideRow);
                    pair<double, Vector2d> ret11 = decideFaceId(end1, topCol, topRow, sideCol, sideRow);

                    pair<double, Vector2d> ret2 = decideFaceId(start2, topCol, topRow, sideCol, sideRow);
                    pair<double, Vector2d> ret22 = decideFaceId(end2, topCol, topRow, sideCol, sideRow);

                    face_id_start1 = ret1.first;
                    face_id_end1 = ret11.first;
                    face_id_start2 = ret2.first;
                    face_id_end2 = ret22.first;



                    bool no_cross_face = (int(face_id_start1) == int(face_id_start2)) && (int(face_id_start1) == int(face_id_end1)) && (int(face_id_start1) == int(face_id_end2)); //  () () ();
#if 0
                    std::cout<<"no_cross_face: "<<no_cross_face<<", start1: "<< start1<<", end1: "<< end1<<", start2: "<< start2<<", end2: "<< end2<<std::endl;
#endif
                    // if((serr.dot(serr) < 60 * 60) && (eerr.dot(eerr) < 60 * 60))   // 线段在图像里不会跑得特别远
                    if((serr.dot(serr) < 100 * 100) && (eerr.dot(eerr) < 100 * 100)&&(abs(line1.angle-line2.angle)<0.1) && no_cross_face)   // 线段在图像里不会跑得特别远
                    // if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&(abs(line1.angle-line2.angle)<0.1) && no_cross_face)   // 线段在图像里不会跑得特别远
                    // if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
                        good_matches.push_back( lsd_matches[i] );
                }

            }

            std::cout << forwframe_->lineID.size() <<" " <<curframe_->lineID.size();
            for (int k = 0; k < good_matches.size(); ++k) {
                DMatch mt = good_matches[k];
                forwframe_->lineID[mt.queryIdx] = curframe_->lineID[mt.trainIdx];

            }
            if(SHOW_LINE) {
                visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd,
                                     curframe_->keylsd, good_matches);
            }
            vector<KeyLine> vecLine_tracked, vecLine_new;
            vector< int > lineID_tracked, lineID_new;
            Mat DEscr_tracked, Descr_new;

            // 将跟踪的线和没跟踪上的线进行区分
            for (size_t i = 0; i < forwframe_->keylsd.size(); ++i)
            {
                if( forwframe_->lineID[i] == -1)
                {
                    forwframe_->lineID[i] = allfeature_cnt++;
                    vecLine_new.push_back(forwframe_->keylsd[i]);
                    lineID_new.push_back(forwframe_->lineID[i]);
                    Descr_new.push_back( forwframe_->lbd_descr.row( i ) );
                }
                else
                {
                    vecLine_tracked.push_back(forwframe_->keylsd[i]);
                    lineID_tracked.push_back(forwframe_->lineID[i]);
                    DEscr_tracked.push_back( forwframe_->lbd_descr.row( i ) );
                }
            }
#if 0
            int diff_n = 50 - vecLine_tracked.size();  // 跟踪的线特征少于50了，那就补充新的线特征, 还差多少条线
#else
            int diff_n = 500 - vecLine_tracked.size();  // 跟踪的线特征少于500了，那就补充新的线特征, 还差多少条线
#endif
            if( diff_n > 0)    // 补充线条
            {

                for (int k = 0; k < vecLine_new.size(); ++k) {
                    vecLine_tracked.push_back(vecLine_new[k]);
                    lineID_tracked.push_back(lineID_new[k]);
                    DEscr_tracked.push_back(Descr_new.row(k));
                }

            }

            forwframe_->keylsd = vecLine_tracked;
            forwframe_->lineID = lineID_tracked;
            forwframe_->lbd_descr = DEscr_tracked;

        }




        // 将opencv的KeyLine数据转为季哥的Line
        for (int j = 0; j < forwframe_->keylsd.size(); ++j) {
            Line l;
            KeyLine lsd = forwframe_->keylsd[j];
            l.StartPt = lsd.getStartPoint();
            l.EndPt = lsd.getEndPoint();
            l.length = lsd.lineLength;
//            Eigen::Vector2d pts, pte;
//            pts(0) = l.StartPt.x;
//            pts(1) = l.StartPt.y;
//            pte(0) = l.EndPt.x;
//            pte(1) = l.EndPt.y;
            /// 这个归一化 [球坐标] 应该落在[ 前、后、左、右、中] 哪一部分针孔坐标系上, 并给出在针孔相机下的2d坐标
#if 0
            Eigen::Vector3d pts, pte;
            pts(0) = (l.StartPt.x - center_x)/focal_x;
            pts(1) = (l.StartPt.y - center_y)/focal_y;
            pts(2) = 1;

            pte(0) = (l.EndPt.x - center_x)/focal_x;
            pte(1) = (l.EndPt.y - center_y)/focal_y;
            pte(2) = 1;
            auto rets = fisheys_undists[0].project_point_to_vcam_id(pts);
            auto rete = fisheys_undists[0].project_point_to_vcam_id(pte);
             assert(rets.first == rete.first);
// printf("face id: %d, %d \n", rets.first, rete.first);
#endif

//            int  face_id_start = decideFaceId(pts,  topCol,  topRow,  sideCol,  sideRow);
//            int  face_id_end = decideFaceId(pte,  topCol,  topRow,  sideCol,  sideRow);

          // double  face_id_start = decideFaceId(l.StartPt,  topCol,  topRow,  sideCol,  sideRow);
          // double  face_id_end = decideFaceId(l.EndPt,  topCol,  topRow,  sideCol,  sideRow);


            double face_id_start;
            double face_id_end ;

            pair<double, Vector2d> ret11 = decideFaceId(l.StartPt, topCol, topRow, sideCol, sideRow);
            pair<double, Vector2d> ret22 = decideFaceId(l.EndPt, topCol, topRow, sideCol, sideRow);
            face_id_start = ret11.first;
            face_id_end = ret22.first;
#if 0
            printf("face_id_start face_id_end: [%f %f] \n", face_id_start, face_id_end);
            std::cout<<"l.StartPt: "<< l.StartPt<<", l.EndPt: "<< l.EndPt<<std::endl;
#endif
           assert(int(face_id_start) == int(face_id_end));

#if 0
            l.StartPt = point2f(ret11.second.x(), ret11.second.y());
            l.EndPt = point2f(ret11.second.x(), ret11.second.y()); //ret22.second;
#endif
            l.StartP = ret11.second;
            l.EndP = ret22.second;

            l.face_id = face_id_start; // 1;
            forwframe_->vecLine.push_back(l);
        }
        curframe_ = forwframe_;


    }



    //FeatureFrame FisheyeFeatureTrackerOpenMP::trackImage(double _cur_time, cv::InputArray img0, cv::InputArray img1, LineFeatureFrame &lineFeatureFrame) {
    PointLineFeature FisheyeFeatureTrackerOpenMP::trackImage(double _cur_time, cv::InputArray img0, cv::InputArray img1) {
    // ROS_INFO("tracking fisheye cpu %ld:%ld", fisheye_imgs_up.size(), fisheye_imgs_down.size());


    /// fisheye trackImage entrance


    int PYR_LEVEL = LK_PYR_LEVEL;
    cv::Size WIN_SIZE = cv::Size(LK_WIN_SIZE, LK_WIN_SIZE);



    cur_time = _cur_time;
    static double count = 0;
    count += 1;

    CvImages fisheye_imgs_up;
    CvImages fisheye_imgs_down;

    img0.getMatVector(fisheye_imgs_up);
    img1.getMatVector(fisheye_imgs_down);
    TicToc t_r;




#if 0
    imwrite("img_up_0.png",  fisheye_imgs_up[0]);
    imwrite("img_up_1.png",  fisheye_imgs_up[1]);
    imwrite("img_down_0.png",  fisheye_imgs_down[0]);
    imwrite("img_down_1.png",  fisheye_imgs_down[1]);
#endif
    //std::cout<<"imgSize: ["<<fisheye_imgs_up[0].cols<<" "<<fisheye_imgs_up[0].rows<<"]"<<std::endl;


    cv::Mat up_side_img = concat_side(fisheye_imgs_up);
    //std::cout<<"fisheye_imgs_down[0].size: "<<fisheye_imgs_down[0].size<<std::endl;
    //std::cout<<"fisheye_imgs_down[1].size: "<<fisheye_imgs_down[1].size<<std::endl;
    cv::Mat down_side_img = concat_side(fisheye_imgs_down);
    cv::Mat &up_top_img = fisheye_imgs_up[0];
    cv::Mat &down_top_img = fisheye_imgs_down[0];
    //cv::Mat  up_top_img = fisheye_imgs_up[0];
    //cv::Mat  down_top_img = fisheye_imgs_down[0];

#if 0
    std::cout<<"================================== up_side_img size: ["<<up_side_img.cols<<" "<<up_side_img.rows<<"]"<<std::endl;

    imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_side_img_0.png",  up_side_img);
    imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/down_side_img_0.png",  down_side_img);
    imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_top_img_0.png",  up_top_img);
    imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/down_top_img_0.png",  down_top_img);
#endif


#if 0
    LineFeatureTracker trackerData;
#endif

/// roger line
        // cv::Mat centerImg = up_top_img.clone();
#if 1

LineFeatureFrame lineFeatureFrame;

/////////////////////////////////////////////////////////////////////

        float fxx = fisheys_undists[0].f_center;
        float fyy = fisheys_undists[0].f_center;
        float cxx = fisheys_undists[0].imgWidth/2;
        float cyy = fisheys_undists[0].imgWidth/2;
        if (focal_x == 0) {
            printf("\nassigning pinhle params... , focal_x = %0.3f \n", focal_x);
            focal_x = fxx;
            focal_y = fyy;
            center_x = cxx;
            center_y = cyy;
            cen_width = fisheys_undists[0].imgWidth;
            //cen_height = fisheys_undists[0].imgHeight;
            cen_height = fisheys_undists[0].imgWidth;
#if 0
            cam_faces.t01 = t1;
            cam_faces.t02 = t2;
            cam_faces.t03 = t3;
            cam_faces.t04 = t4;
#else
            // Eigen::Quaterniond t00 = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
            Eigen::Quaterniond t00 = Eigen::Quaterniond(Eigen::AngleAxisd(0, Eigen::Vector3d(1, 0, 0)));
            cam_faces.clear();
            cam_faces.push_back(t00);
            cam_faces.push_back(t1);
            cam_faces.push_back(t2);
            cam_faces.push_back(t3);
            cam_faces.push_back(t4);
            assert(cam_faces.size() == 5);
            //int img_rows_all = up_side_img.rows();
              cv::Mat mask_line = setMaskLine(up_side_img.rows, up_side_img.cols, fisheys_undists);

#endif
        }
        else {
            printf("\npinhle params already assigned... , focal_x = %0.3f \n", focal_x);
        }







/////////////////////////////////////////////////////////////////////

        // frame_cnt++;
        if(USE_LINE) {

           // int img_rows_all = up_side_img.rows();
           // cv::Mat mask_line = setMaskLine(up_side_img.rows(), up_side_img.cols(), fisheys_undists);
           cv::Mat mask_line = setMaskLine(up_side_img.rows, up_side_img.cols, fisheys_undists);
#if 0
           imshow("mask_line", mask_line);
           waitKey(2);
#endif

#if 0
            cv::Mat centerImg = up_top_img.clone();
#else
            cv::Mat centerImg = up_side_img.clone();
#endif
#if 1
            readImage(centerImg, mask_line);
#else
            readImage(centerImg, cv::Mat());
#endif
            int num_cam = 1;
            int stereo_track = 0;
            vector<set<int>> hash_ids(num_cam);
            // LineFeatureFrame lineFeatureFrame;
            for (int i = 0; i < num_cam; i++) {
                if (i != 1 || !stereo_track)  // 单目
                {
                    auto un_lines = undistortedLineEndPoints();
// printf("############################################################################tracked line num: %d \n",un_lines.size());
                    //auto &cur_lines = trackerData.curframe_->vecLine;
                    auto &ids = curframe_->lineID;

                    for (unsigned int j = 0; j < ids.size(); j++) {

                        int p_id = ids[j];
                        hash_ids[i].insert(p_id);
#if 0
                        geometry_msgs::Point32 p;
                        p.x = un_lines[j].StartPt.x;
                        p.y = un_lines[j].StartPt.y;
                        p.z = 1;

                        feature_lines->points.push_back(p);
                        id_of_line.values.push_back(p_id * num_cam + i);
                        //std::cout<< "feature tracking id: " <<p_id * num_cam + i<<" "<<p_id<<"\n";
                        u_of_endpoint.values.push_back(un_lines[j].EndPt.x);
                        v_of_endpoint.values.push_back(un_lines[j].EndPt.y);
#else
                        Vector4d obs_(un_lines[j].StartPt.x, un_lines[j].StartPt.y,
                                 un_lines[j].EndPt.x, un_lines[j].EndPt.y);
                        Vector5d obs__;
#if 0
                        obs__.head(4) = obs_;
                        obs__.tail(1) = 1.0;
#else
                        obs__.head<4>() = obs_;
#if 0
                        obs__.tail<1>() = 1.0;
#else
                        // obs__(4) = 1.0;
                        obs__(4) = un_lines[j].face_id;
                        // printf("faceId: %f \n", obs__(4));
#endif
#endif
                        //obs__(4) = 1.0;
                        lineFeatureFrame[p_id].emplace_back(i, obs__);
#endif
                        //ROS_ASSERT(inBorder(cur_pts[j]));
                    }
                }

            }
        }
        else
        {
           //  LineFeatureFrame lineFeatureFrame;
        }
#else
        trackLine(centerImg, fisheys_undists[0].sideImgHeight);
#endif
/// roger line




    std::vector<cv::Mat> *up_top_pyr = nullptr, *down_top_pyr = nullptr, *up_side_pyr = nullptr, *down_side_pyr = nullptr;
    double concat_cost = t_r.toc();
    move_side_to_top = false; //true;
    top_size = up_top_img.size();
    side_size = up_side_img.size();
    if (fisheye_imgs_up.size() > 1) {
        side_size_single = fisheye_imgs_up[1].size();
    } else {
        side_size_single = side_size;
    }

    //Clear All current pts
    cur_up_top_pts.clear();
    cur_up_side_pts.clear();
    cur_down_top_pts.clear();
    cur_down_side_pts.clear();

    cur_up_top_un_pts.clear();
    cur_up_side_un_pts.clear();
    cur_down_top_un_pts.clear();
    cur_down_side_un_pts.clear();


    TicToc t_pyr;

    /// 构建up_top，up_side，down_top，down_side四部分图像金字塔
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (enable_up_top) {
                // printf("Building up top pyr\n");
                up_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_top_img, *up_top_pyr, WIN_SIZE, PYR_LEVEL,
                                            true);//, cv::BORDER_REFLECT101, cv::BORDER_CONSTANT, false);
            }
        }

#pragma omp section
        {
            if (enable_down_top) {
                // printf("Building down top pyr\n");
                down_top_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_top_img, *down_top_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }

#pragma omp section
        {
            if (enable_up_side) {
                // printf("Building up side pyr\n");
                up_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(up_side_img, *up_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }

#pragma omp section
        {
            if (enable_down_side) {
                // printf("Building downn side pyr\n");
                down_side_pyr = new std::vector<cv::Mat>();
                cv::buildOpticalFlowPyramid(down_side_img, *down_side_pyr, WIN_SIZE, PYR_LEVEL, true);
            }
        }
    }


    cv::Mat asfa = cv::imread("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_top_img.png");

    if (asfa.cols < 10) {
        std::cout<<"up_top_img size: "<<up_top_img.size<<std::endl;
        std::cout<<"down_top_img size: "<<down_top_img.size<<std::endl;
        std::cout<<"up_side_img size: "<<up_side_img.size<<std::endl;
        std::cout<<"down_side_img size: "<<down_side_img.size<<std::endl;
        if (enable_up_top) {
            imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_top_img.png", up_top_img);
        }
        if (enable_down_top) {
            imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/down_top_img.png", down_top_img);
        }
        if (enable_up_side) {
            imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_side_img.png", up_side_img);
        }
        if (enable_down_side) {
            imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/down_side_img.png", down_side_img);
        }
        //imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_top_img.png",  up_top_img);
    }
    static double pyr_sum = 0;
    pyr_sum += t_pyr.toc();

    TicToc t_t;
    set_predict_lock.lock();

#pragma omp parallel sections
    {
#pragma omp section
        {
            //If has predict;
            if (enable_up_top) {
                // printf("Start track up top\n");

                //imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/up_top_img.png",  up_top_img);
                //std::cout<<"================================== up_top_img size: ["<<up_top_img.cols<<" "<<up_top_img.rows<<"]"<<std::endl;
                //std::cout<<"================================== prev_up_top_img size: ["<<prev_up_top_img.cols<<" "<<prev_up_top_img.rows<<"]"<<std::endl;
                //imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/prev_up_top_img.png",  prev_up_top_img);
                if (prev_up_top_img.cols > 10) {
                    //imwrite("/home/roger/vins-fisheye2/src/VINS-Fisheye/data/prev_up_top_img.png",  prev_up_top_img);
                    //cv::Mat combine;
                    //hconcat(up_top_img, prev_up_top_img, combine);
                    //cv::imshow("up_prv_cur_combine", combine);
                    //cv::waitKey(2);


//                    cv::imshow("up_top_img ", up_top_img );
//                    cv::waitKey(2);
//                    cv::imshow("prev_up_top_img", prev_up_top_img);
//                    cv::waitKey(2);
                }
                cur_up_top_pts = opticalflow_track(up_top_img, up_top_pyr, prev_up_top_img, prev_up_top_pyr,
                                                   prev_up_top_pts, ids_up_top, track_up_top_cnt, removed_pts,
                                                   predict_up_top);

                //std::cout << "ids_up_top: " << ids_up_top<<std::endl;
//                for(int i = 0; i < cur_up_top_pts.size();i++) {
//                    std::cout << "cur_up_top_pts: " << cur_up_top_pts[i].x <<" "<<cur_up_top_pts[i].y<< std::endl;
//                }
//                std::cout<<"#######################"<<std::endl;
//                for(int i = 0; i < prev_up_top_pts.size();i++) {
//                    std::cout << "prev_up_top_pts: " << prev_up_top_pts[i].x <<" "<<prev_up_top_pts[i].y<< std::endl;
//                }
//                std::cout<<"#######################"<<std::endl;
//                for(int i = 0; i < predict_up_top.size();i++) {
//                    std::cout << "predict_up_top: " << predict_up_top[i].x <<" "<<predict_up_top[i].y<< std::endl;
//                }
//                std::cout<<"#######################"<<std::endl;
                // printf("End track up top\n");
            }
        }

#pragma omp section
        {
            if (enable_up_side) {
                // printf("Start track up side\n");

                if (prev_up_side_img.cols > 10) {
//                    cv::Mat combine;
//                    hconcat(up_side_img.colRange(0,400), prev_up_side_img.colRange(0,400), combine);
//                    cv::imshow("up_side_prv_cur_combine", combine);
//                    cv::waitKey(2);


                }


                cur_up_side_pts = opticalflow_track(up_side_img, up_side_pyr, prev_up_side_img, prev_up_side_pyr,
                                                    prev_up_side_pts, ids_up_side, track_up_side_cnt, removed_pts,
                                                    predict_up_side);
                // printf("End track up side\n");
            }
        }

#pragma omp section
        {
            if (enable_down_top) {
                // printf("Start track down top\n");
                cur_down_top_pts = opticalflow_track(down_top_img, down_top_pyr, prev_down_top_img, prev_down_top_pyr,
                                                     prev_down_top_pts, ids_down_top, track_down_top_cnt, removed_pts,
                                                     predict_down_top);
                // printf("End track down top\n");
            }
        }
    }

    set_predict_lock.unlock();

    // setMaskFisheye();
    static double lk_sum = 0;
    lk_sum += t_t.toc();

    TicToc t_d;

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (enable_up_top) {
                detectPoints(up_top_img, cv::Mat(), n_pts_up_top, cur_up_top_pts, TOP_PTS_CNT, fisheys_undists);
            }
        }

#pragma omp section
        {
            if (enable_down_top) {
                detectPoints(down_top_img, cv::Mat(), n_pts_down_top, cur_down_top_pts, TOP_PTS_CNT, fisheys_undists);
            }
        }

#pragma omp section
        {
            if (enable_up_side) {
                detectPoints(up_side_img, cv::Mat(), n_pts_up_side, cur_up_side_pts, SIDE_PTS_CNT, fisheys_undists);
            }
        }
    }

    // ROS_INFO("Detect cost %fms", t_d.toc());

    static double detect_sum = 0;

    detect_sum = detect_sum + t_d.toc();

    addPointsFisheye();

    TicToc t_tk;
    {
        if (enable_down_side) {
            ids_down_side = ids_up_side;
            std::vector<cv::Point2f> down_side_init_pts = cur_up_side_pts;
            if (down_side_init_pts.size() > 0) {
                cur_down_side_pts = opticalflow_track(down_side_img, down_side_pyr, up_side_img,
                                                      up_side_pyr, down_side_init_pts, ids_down_side,
                                                      track_down_side_cnt, removed_pts, predict_down_side);
            }
        }
    }

    // ROS_INFO("Tracker 2 cost %fms", t_tk.toc());

    //Undist points
    ///像素坐标转归一化坐标
    ///同时把针孔模型下的坐标施加旋转转化到鱼眼模型下的坐标，为了方便公用后面的PnpRansac这些算法，这是解释了我的一大疑惑
    ///同时把针孔模型下的坐标施加旋转转化到鱼眼模型下的坐标，为了方便公用后面的PnpRansac这些算法，还有最重要的初始化部分代码逻辑都能共用，这是解释了我的一大疑惑

    if (CUBE_MAP) {
        up_top_ids.clear();
        down_top_ids.clear();
        up_top_pts_un.clear();
        down_top_pts_un.clear();
        up_top_pts_.clear();
        down_top_pts_.clear();

        up_top_ids_side_part.clear();
        down_top_ids_side_part.clear();
    }

    vector<cv::Point2f> cur_up_top_pts_bak = cur_up_top_pts;
    vector<cv::Point2f> cur_down_top_pts_bak = cur_down_top_pts;

//    if (move_side_to_top) {
//    vector<cv::Point2f> cur_up_top_pts_bak = cur_up_top_pts;
//    vector<cv::Point2f> cur_down_top_pts_bak = cur_down_top_pts;
//}
    cur_up_top_un_pts = undistortedPtsTop(cur_up_top_pts, fisheys_undists[0]);
    cur_down_top_un_pts = undistortedPtsTop(cur_down_top_pts, fisheys_undists[1]);




    vector<cv::Point2f> cur_up_side_pts_bak = cur_up_side_pts;
    vector<cv::Point2f> cur_down_side_pts_bak = cur_down_side_pts;

    vector<int> ids_up_side_bak = ids_up_side;
    vector<int> ids_down_side_bak = ids_down_side;
    if (move_side_to_top) {
        std::cout << "cur_up_top_pts_bak.size() - cur_up_top_pts.size(): "
                  << cur_up_top_pts_bak.size() - cur_up_top_pts.size() << std::endl;
        std::cout << "cur_down_top_pts_bak.size() - cur_down_top_pts.size(): "
                  << cur_down_top_pts_bak.size() - cur_down_top_pts.size() << std::endl;

//        vector<cv::Point2f> cur_up_side_pts_bak = cur_up_side_pts;
//        vector<cv::Point2f> cur_down_side_pts_bak = cur_down_side_pts;
//
//        vector<int> ids_up_side_bak = ids_up_side;
//        vector<int> ids_down_side_bak = ids_down_side;
    }
    cur_up_side_un_pts = undistortedPtsSide(cur_up_side_pts, fisheys_undists[0], false);
    cur_down_side_un_pts = undistortedPtsSide(cur_down_side_pts, fisheys_undists[1], true);

    //std::cout<<"cur_up_side_pts:\n"<<cur_up_side_pts<<std::endl;


if (move_side_to_top){
    std::cout<<"up_top_ids.size(): "<<up_top_ids.size()<<std::endl;
    std::cout<<"down_top_ids.size(): "<<down_top_ids.size()<<std::endl;
    std::cout<<"up_top_pts_un.size(): "<<up_top_pts_un.size()<<std::endl;
    std::cout<<"down_top_pts_un.size(): "<<down_top_pts_un.size()<<std::endl;
    std::cout<<"up_top_pts_.size(): "<<up_top_pts_.size()<<std::endl;
    std::cout<<"down_top_pts_.size(): "<<down_top_pts_.size()<<std::endl;
    std::cout<<"up_top_ids_side_part.size(): "<<up_top_ids_side_part.size()<<std::endl;
    std::cout<<"down_top_ids_side_part.size(): "<<down_top_ids_side_part.size()<<std::endl;
    std::cout<<"--------"<<std::endl;

    for (int i = 0; i < up_top_ids.size(); i++){
        ids_up_top.push_back(ids_up_side_bak[up_top_ids[i]]);
        cur_up_top_un_pts.push_back(up_top_pts_un[i]);
        cur_up_top_pts.push_back(up_top_pts_[i]);
    }
    for (int i = 0; i < down_top_ids.size(); i++){
        ids_down_top.push_back(ids_down_side_bak[down_top_ids[i]]);
        cur_down_top_un_pts.push_back(down_top_pts_un[i]);
        cur_down_top_pts.push_back(down_top_pts_[i]);
    }


    cur_up_side_pts.clear();
    cur_down_side_pts.clear();
    ids_up_side.clear();
    ids_down_side.clear();
    std::cout<<"cur_up_side_pts_bak.size() after clearing cur_up_side_pts: " << cur_up_side_pts_bak.size()<<std::endl;
    for(int i = 0; i < up_top_ids_side_part.size(); i++){
        ids_up_side.push_back(ids_up_side_bak[up_top_ids_side_part[i]]);
        cur_up_side_pts.push_back(cur_up_side_pts_bak[up_top_ids_side_part[i]]);
    }
    for(int i = 0; i < down_top_ids_side_part.size(); i++){
        ids_down_side.push_back(ids_down_side_bak[down_top_ids_side_part[i]]);
        cur_down_side_pts.push_back(cur_down_side_pts_bak[down_top_ids_side_part[i]]);
    }
}
    //Calculate Velocitys
    up_top_vel = ptsVelocity3D(ids_up_top, cur_up_top_un_pts, cur_up_top_un_pts_map, prev_up_top_un_pts_map);
    down_top_vel = ptsVelocity3D(ids_down_top, cur_down_top_un_pts, cur_down_top_un_pts_map, prev_down_top_un_pts_map);

    up_side_vel = ptsVelocity3D(ids_up_side, cur_up_side_un_pts, cur_up_side_un_pts_map, prev_up_side_un_pts_map);
    down_side_vel = ptsVelocity3D(ids_down_side, cur_down_side_un_pts, cur_down_side_un_pts_map, prev_down_side_un_pts_map);

    // ROS_INFO("Up top VEL %ld", up_top_vel.size());
    double tcost_all = t_r.toc();
    if (SHOW_TRACK) {
        drawTrackFisheye(cv::Mat(), cv::Mat(), up_top_img, down_top_img, up_side_img, down_side_img);
    }

        
    prev_up_top_img = up_top_img;
    prev_down_top_img = down_top_img;
    prev_up_side_img = up_side_img;

    if(prev_down_top_pyr != nullptr) {
        delete prev_down_top_pyr;
    }

    if(prev_up_top_pyr != nullptr) {
        delete prev_up_top_pyr;
    }

    if (prev_up_side_pyr!=nullptr) {
        delete prev_up_side_pyr;
    }

    if (down_side_pyr!=nullptr) {
        delete down_side_pyr;
    }

    prev_down_top_pyr = down_top_pyr;
    prev_up_top_pyr = up_top_pyr;
    prev_up_side_pyr = up_side_pyr;

    prev_up_top_pts = cur_up_top_pts;
    prev_down_top_pts = cur_down_top_pts;
    prev_up_side_pts = cur_up_side_pts;
    prev_down_side_pts = cur_down_side_pts;

    prev_up_top_un_pts = cur_up_top_un_pts;
    prev_down_top_un_pts = cur_down_top_un_pts;
    prev_up_side_un_pts = cur_up_side_un_pts;
    prev_down_side_un_pts = cur_down_side_un_pts;

    prev_up_top_un_pts_map = cur_up_top_un_pts_map;
    prev_down_top_un_pts_map = cur_down_top_un_pts_map;
    prev_up_side_un_pts_map = cur_up_side_un_pts_map;
    prev_down_side_un_pts_map = cur_up_side_un_pts_map;
    prev_time = cur_time;


    ///等号左边是用来可视化tracking点
    if (PRINT_LOG) {
        std::cout << "ids_up_top.size(): " << ids_up_top.size() << ", cur_up_top_pts.size(): " << cur_up_top_pts.size()
                  << std::endl;
        std::cout << "ids_down_top.size(): " << ids_down_top.size() << ", cur_down_top_pts.size(): "
                  << cur_down_top_pts.size() << std::endl;
        std::cout << "ids_up_side.size(): " << ids_up_side.size() << ", cur_up_side_pts.size(): "
                  << cur_up_side_pts.size() << std::endl;
        std::cout << "ids_down_side.size(): " << ids_down_side.size() << ", cur_down_side_pts.size(): "
                  << cur_down_side_pts.size() << std::endl;
    }
    up_top_prevLeftPtsMap = pts_map(ids_up_top, cur_up_top_pts);
    down_top_prevLeftPtsMap = pts_map(ids_down_top, cur_down_top_pts);
    up_side_prevLeftPtsMap = pts_map(ids_up_side, cur_up_side_pts);
    down_side_prevLeftPtsMap = pts_map(ids_down_side, cur_down_side_pts);

    // hasPrediction = false;
    auto ff = setup_feature_frame();
    
    static double whole_sum = 0.0;

    whole_sum += t_r.toc();

    printf("FT Whole %fms; AVG %fms\n DetectAVG %fms PYRAvg %fms LKAvg %fms Concat %fms PTS %ld T\n", 
        t_r.toc(), whole_sum/count, detect_sum/count, pyr_sum/count, lk_sum/count, concat_cost, ff.size());
    PointLineFeature pointLineFeature;
        pointLineFeature.featureFrame = ff;
        pointLineFeature.lineFeatureFrame = lineFeatureFrame;
    //return ff;
        return pointLineFeature;
}


};