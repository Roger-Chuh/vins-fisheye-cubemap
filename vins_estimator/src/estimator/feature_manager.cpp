/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"
int lineFeaturePerId::endFrame()
{
    return start_frame + linefeature_per_frame.size() - 1;
}
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &_it : feature)
    {
        auto & it = _it.second;
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4 && it.good_for_solving)
        {
            cnt++;
        }
    }
    return cnt;
}

int FeatureManager::getLineFeatureCount()
{
    int cnt = 0;
    for (auto &it : linefeature)
    {

        it.used_num = it.linefeature_per_frame.size();

        if (it.used_num >= LINE_MIN_OBS && it.start_frame < WINDOW_SIZE - 2 && it.is_triangulation)
        {
            cnt++;
        }
    }
    return cnt;
}
MatrixXd FeatureManager::getLineOrthVectorInCamera()
{
    MatrixXd lineorth_vec(getLineFeatureCount(),4);
    int feature_index = -1;
    for (auto &it_per_id : linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))
            continue;

        lineorth_vec.row(++feature_index) = plk_to_orth(it_per_id.line_plucker);

    }
    return lineorth_vec;
}
void FeatureManager::setLineOrthInCamera(MatrixXd x)
{
    int feature_index = -1;
    for (auto &it_per_id : linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))
            continue;

        //std::cout<<"x:"<<x.rows() <<" "<<feature_index<<"\n";
        Vector4d line_orth = x.row(++feature_index);
        it_per_id.line_plucker = orth_to_plk(line_orth);// transfrom to camera frame

        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        /*
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
         */
    }
}
MatrixXd FeatureManager::getLineOrthVector(Vector3d Ps[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces)
{
    MatrixXd lineorth_vec(getLineFeatureCount(),4);
    int feature_index = -1;
    for (auto &it_per_id : linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))
            continue;

        int imu_i = it_per_id.start_frame;
/// roger line
  ///      ROS_ASSERT(NUM_OF_CAM == 1);
        double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);

        Eigen::Quaterniond t0x = cam_faces[face_id];
        Eigen::Vector3d twc_ = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc_ = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        Eigen::Vector3d twc = twc_;
        Eigen::Matrix3d Rwc = Rwc_ * t0x.toRotationMatrix();

        // todo::
        /// 很关键的一点，每个it_per_id.line_plucker都是表达在host frame坐标系的，而优化时的orth又是表达在世界body坐标系的，
        /// 所以it_per_id.line_plucker->orth时，要走line_triangulation(天然得到的plk_c,或者说三角化得到的plk天然就是表达在local camera坐标系下的) -> plk_c -> plk_w -> orth_w -> plk_w -> plk_c -> reproj_error这个数据流程
        /// 但是，在做多帧三角化而不是2帧三角化时，plk是表达在world系下的，要再转会camera系下（有world_body->world_cam->line_triangulation->plk_w->plk_c->it_per_id.line_plucker）才能赋给it_per_id.line_plucker
        Vector6d line_w = plk_to_pose(it_per_id.line_plucker, Rwc, twc);  // transfrom to world frame
        // line_w.normalize();
        lineorth_vec.row(++feature_index) = plk_to_orth(line_w);
        //lineorth_vec.row(++feature_index) = plk_to_orth(it_per_id.line_plucker);

    }
    return lineorth_vec;
}

void FeatureManager::setLineOrth(MatrixXd x,Vector3d P[], Matrix3d R[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces)
{
    int feature_index = -1;
    for (auto &it_per_id : linefeature)
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.is_triangulation))
            continue;

        Vector4d line_orth_w = x.row(++feature_index);
        Vector6d line_w = orth_to_plk(line_orth_w);

        int imu_i = it_per_id.start_frame;
        /// roger line
    ///    ROS_ASSERT(NUM_OF_CAM == 1);


        double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);

        Eigen::Quaterniond t0x = cam_faces[face_id];

        Eigen::Vector3d twc_ = P[imu_i] + R[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc_ = R[imu_i] * ric[0];               // Rwc = Rwi * Ric


        Eigen::Vector3d twc = twc_;
        Eigen::Matrix3d Rwc = Rwc_ * t0x.toRotationMatrix();

        it_per_id.line_plucker = plk_from_pose(line_w, Rwc, twc); // transfrom to camera frame
        //it_per_id.line_plucker = line_w; // transfrom to camera frame

        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        /*
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
         */
    }
}

double FeatureManager::reprojection_error( Vector4d obs, Matrix3d Rwc, Vector3d twc, Vector6d line_w, double &dlt_theta ) {

    double error = 0;

    Vector3d n_w, d_w;
    n_w = line_w.head(3);
    d_w = line_w.tail(3);

    Vector3d p1, p2;
    p1 << obs[0], obs[1], 1;
    p2 << obs[2], obs[3], 1;

    Vector3d line_dir = p1 - p2;
    double line_norm = line_dir.head(2).norm();
    Vector3d line_dir_norm = line_dir / line_norm;

    Vector6d line_c = plk_from_pose(line_w,Rwc,twc);
    /// 果然计算线重投影误差只用到了plk坐标中的法线部分，直线方向对直线在成像平面上的投影没影响
    Vector3d nc = line_c.head(3);
    /// nc的第3项归为1
    /// 本来应该线得投影方程应该是K*n得的，但这里内参为eye(3)，所以投影直线方程直接就是n了，（n要对前3维归一化n = [a b c]，且a*a + b*b = 1）
    double sql = nc.head(2).norm();
    nc /= sql;

double cos_theta1 = line_dir_norm.head(2).dot(nc.head(2));
    double theta1 = acos(cos_theta1) * ( 180.0 / M_PI );
    double cos_theta2 = line_dir_norm.head(2).dot(-nc.head(2));
    double theta2 = acos(cos_theta2) * ( 180.0 / M_PI );
#if 0
std::cout<<"line angle diff1 2 : "<<theta1<<", "<<theta2<<" deg ";
    std::cout<<"[p1: "<<p1.transpose()<<" p2: "<<p2.transpose()<<"], line_dir_norm: "<<line_dir_norm.transpose()<<", nc: "<<nc.transpose()<<" ...";
#endif

    // double dlt_theta;

if (theta1 < theta2){
    dlt_theta = 90 - theta1;
}
else
{
    dlt_theta = 90 - theta2;

}

    if (cos_theta1 > 0.998)
    {

    }

    /// 点线距作为衡量线重投影误差的指标
    error += fabs( nc.dot(p1) );
    error += fabs( nc.dot(p2) );

    return error / 2.0;
}
static void svd_impl(Eigen::MatrixXd A, Eigen::MatrixXd& U, Eigen::MatrixXd& S, Eigen::MatrixXd& V)
{
    JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    JacobiSVD<Eigen::MatrixXd> svd1(A, ComputeFullU | ComputeFullV);
    V = svd.matrixV();
    U = svd.matrixU();
    Eigen::MatrixXd V1 = svd1.matrixV();
    Eigen::MatrixXd U1 = svd1.matrixU();
#if 0
    std::cout << "V:\n" << V << std::endl;
    std::cout<<"U:\n"<<U<<std::endl;
    std::cout<<"U.inverse():\n"<<U.inverse()<<std::endl;
    std::cout<<"A:\n"<<A<<std::endl;
    std::cout<<"V.transpose().inverse():\n"<<V.transpose().inverse()<<std::endl;
#endif

    Eigen::MatrixXd S1 = U1.inverse() * A * V1.transpose().inverse(); // S = U^-1 * A * VT * -1
    S = S1.topRows(S1.cols());
    // Eigen::MatrixXd SS = S1.block<6,6>(0,0);

   // std::cout <<"S1:\n" << S1 << std::endl;
   //  std::cout <<"SS:\n" << SS << std::endl;

#if 0
    std::cout << "U :\n" << U << std::endl;
    std::cout << "S :\n" << S << std::endl;
    std::cout << "V :\n" << V << std::endl;

    std::cout << "A :\n" << A << std::endl;
    std::cout << "U * S * VT :\n" << U * S * V.transpose() << std::endl;
#endif
}
Vector6d FeatureManager::triangulateLine_svd(std::map<int, Eigen::Matrix3d> &Rwc_list, std::map<int, Eigen::Vector3d> &twc_list, std::map<int, Eigen::Vector4d> &obs_list, std::map<int, double> &cos_theta_list)
{
assert(Rwc_list.size() == cos_theta_list.size());
   int kObvNum = Rwc_list.size();
   Eigen::MatrixXd A_temp;
    A_temp.resize(2 * kObvNum, 6);

    int ind = 0;
    int min_imu_id = 10000;
    double min_cos_theta = 1.0;
    for (auto& per_frame : Rwc_list) {
        int imu_num = per_frame.first;

        if(imu_num < min_imu_id){
            min_imu_id = imu_num;
        }
        if(cos_theta_list[imu_num] < min_cos_theta){
            min_cos_theta = cos_theta_list[imu_num] ;
        }
        Eigen::Matrix<double, 3, 6> temp;

        temp.leftCols(3) = Rwc_list[imu_num].transpose();
        temp.rightCols(3) = skew_symmetric(-Rwc_list[imu_num].transpose()*twc_list[imu_num]) * Rwc_list[imu_num].transpose();

        Eigen::Vector3d ps, pe;
        ps.setOnes();
        pe.setOnes();
        ps.head(2) = obs_list[imu_num].head(2);
        pe.head(2) = obs_list[imu_num].tail(2);

        A_temp.row(ind * 2) = ps.transpose() * temp;
        A_temp.row(ind * 2 + 1) = pe.transpose() * temp;

        ind++;
    }

    // std::cout<<"A_temp:\n"<<A_temp<<std::endl;

    Eigen::MatrixXd U, S, V, U1, S1, V1, U_, S_, V_;
    svd_impl(A_temp, U, S, V);
    Eigen::VectorXd plk_w = V.rightCols(1);

    Eigen::Vector3d n = plk_w.head(3);
    Eigen::Vector3d v = plk_w.tail(3);
    Eigen::Matrix<double, 3, 2> C, C_;
    C.leftCols(1) = n;
    C.rightCols(1) = v;

    svd_impl(C, U1, S1, V1);

    Eigen::Matrix2d Z = S1*V1.transpose();
    Eigen::Vector2d Z1 = Z.leftCols(1);
    Eigen::Vector2d  Z2 = Z.rightCols(1);
    Eigen::Matrix2d T,TT,tmp;
    tmp<<0.0,  -1.0,  1.0,  0.0;
    TT.topRows(1) = Z2.transpose();
    TT.bottomRows(1) = Z1.transpose()*tmp;

    T << Z(2-1,1-1), Z(2-1,2-1), Z(1-1,2-1), -Z(1-1,1-1);

    svd_impl(T, U_,S_,V_);

    double V11 = V_(0,1);
    double V22 = V_(1,1);
    Eigen::Matrix2d VV, Diag;
    VV <<V11, -V22,V22, V11;
    Diag = VV.transpose()*S1*V1.transpose();
    Diag(0,1) = 0;
    Diag(1,0) = 0;
    C_ = U1*VV*Diag;

    Vector6d line_w;
    line_w.head(3) = C_.leftCols(1);
    line_w.tail(3) = C_.rightCols(1);
    Vector6d line_c = plk_from_pose(line_w,Rwc_list[min_imu_id],twc_list[min_imu_id]);

    Vector3d nn = line_c.head(3);
    Vector3d vv = line_c.tail(3);
    nn = nn / nn.norm();
    vv = vv / vv.norm();
    double err = nn.dot(vv);
    // assert(fabs(err) < 0.00001);
   assert(fabs(err) < 1.97758e-10);
 //   assert(fabs(err) < 1.97758e-7);
    //printf("orth err: %f \n", err);
// std::cout<<"orth err: "<<err<<std::endl;
    // /if (min_cos_theta > 0.998){}
    if (false){

        line_c(0) = 123456;
    }
    return line_c;
}
// void FeatureManager::triangulateLine2(Vector3d Ps[], Matrix3d Rs_[], Vector3d tic[], Matrix3d ric[], CamFaces cam_faces)
void FeatureManager::triangulateLine2(Vector3d Ps[], Matrix3d Rs_[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces)
{
    //std::cout<<"in triangulateLine() linefeature size: "<<linefeature.size()<<std::endl;
    printf("in triangulateLine() , LINE_MIN_OBS: %d, linefeature size: %d \n", LINE_MIN_OBS, linefeature.size());

    /*std::map<int, Eigen::Matrix3d> Rwc_list;
    std::map<int, Eigen::Vector3d> twc_list;*/

    for (auto &it_per_id : linefeature)        // 遍历每个特征，对新特征进行三角化
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();    // 已经有多少帧看到了这个特征
        // printf("line used_num: %d \n", it_per_id.used_num);



        if (!(it_per_id.used_num >= LINE_MIN_OBS &&
              it_per_id.start_frame < WINDOW_SIZE - 2))   // 看到的帧数少于2， 或者 这个特征最近倒数第二帧才看到， 那都不三角化
            continue;

        if (it_per_id.is_triangulation)       // 如果已经三角化了
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        /// roger line
        ///    ROS_ASSERT(NUM_OF_CAM == 1);

        // double face_id1 = it_per_id.linefeature_per_frame[imu_i].lineobs(4);
        double face_id1 = it_per_id.linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);

        Eigen::Quaterniond t0x = cam_faces[face_id];

        // std::cout<<"imu_i: "<<imu_i<<", face_id: "<<face_id<<", t0x: "<<t0x.toRotationMatrix()<<std::endl;


        Eigen::Vector3d t0_ = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d R0_ = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        Eigen::Vector3d t0 = t0_; //  + R0_ * t0x;  // twx = Rwc * tcx + twc
        Eigen::Matrix3d R0 = R0_ * t0x.toRotationMatrix();             // Rwx = Rwc * Rcx

       // Eigen::Matrix3d asdf = cam_faces[0]*Rs[imu_i];

        std::map<int, Eigen::Matrix3d> Rwc_list;
        std::map<int, Eigen::Vector3d> twc_list;
        std::map<int, Eigen::Vector4d> obs_list;
        std::map<int, Eigen::Vector3d> pi_norm_list;
        std::map<int, double> cos_theta_list;
        Rwc_list[imu_i] = (R0);
        twc_list[imu_i] = (t0);
        //cos_theta_list[imu_i] = 0.0;

        assert(std::fabs(Rs[0](0) - Rs_[0](0)) < 0.000001);
        assert(std::fabs(Rs[2](0) - Rs_[2](0)) < 0.000001);
       //  std::cout<<"Rs[0]: "<<Rs[0]<<std::endl;
       //  std::cout<<"Rs_[0]: "<<Rs_[0]<<std::endl;


        double d = 0, min_cos_theta = 1.0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obsi, obsj;  // obs from two frame are used to do triangulation

        // plane pi from ith obs in ith camera frame
        Eigen::Vector4d pii;
        Eigen::Vector3d ni;      // normal vector of plane
        int trace_num = it_per_id.linefeature_per_frame.size();

        for (auto &it_per_frame : it_per_id.linefeature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;
//Eigen::Vector4d obs_t_jemp;
            //obs_temp
            // obs_list[imu_j] = it_per_frame.lineobs;


            double face_id11 = it_per_frame.lineobs(4);
            int face_id_ = int(face_id11);
            assert(face_id_ == face_id);


            if (imu_j == imu_i)   // 第一个观测是start frame 上
            {
                obsi = it_per_frame.lineobs.head(4);
                Eigen::Vector3d p1(obsi(0), obsi(1), 1);
                Eigen::Vector3d p2(obsi(2), obsi(3), 1);
                pii = pi_from_ppp(p1, p2, Vector3d(0, 0, 0));
                ni = pii.head(3);
                ni.normalize();
                obs_list[imu_j] = it_per_frame.lineobs.head(4);
                // pi_norm_list[imu_j] = ni;
                cos_theta_list[imu_j] = 0;
                continue;
            }

//            double face_id11 = it_per_frame.lineobs(4);
//            int face_id_ = int(face_id11);
//assert(face_id_ == face_id);

            // Eigen::Quaterniond t0x = cam_faces[face_id];
#if 0
            if ((imu_j - imu_i) != (trace_num - 1) && (imu_i != imu_j)){
                continue;
            }
#endif
            // 非start frame(其他帧)上的观测
            /// 相机在世界坐标系下的pose
            Eigen::Vector3d t1_ = Ps[imu_j] + Rs[imu_j] * tic[0]; // twc = Rwi * tic + twi
            Eigen::Matrix3d R1_ = Rs[imu_j] * ric[0];             // Rwc = Rwi * Ric



            Eigen::Vector3d t1 = t1_; //  + R0_ * t0x;  // twx = Rwc * tcx + twc
            Eigen::Matrix3d R1 = R1_ * t0x.toRotationMatrix();             // Rwx = Rwc * Rcx

            Rwc_list[imu_j] = (R1);
            twc_list[imu_j] = (t1);






            Eigen::Vector3d t = R0.transpose() * (t1 - t0);   // tij
            Eigen::Matrix3d R = R0.transpose() * R1;          // Rij

            Eigen::Vector4d obsj_tmp = it_per_frame.lineobs.head(4);

            // plane pi from jth obs in ith camera frame
            Vector3d p3(obsj_tmp(0), obsj_tmp(1), 1);
            Vector3d p4(obsj_tmp(2), obsj_tmp(3), 1);
            p3 = R * p3 + t;
            p4 = R * p4 + t;
            Vector4d pij = pi_from_ppp(p3, p4, t);
            Eigen::Vector3d nj = pij.head(3);
            nj.normalize();


            double cos_theta = ni.dot(nj);
            obs_list[imu_j] = it_per_frame.lineobs.head(4);
            cos_theta_list[imu_j] = cos_theta;

            /// 角度阈值和距离阈值中，角度阈值的优先级更高一点? 所以先判断角度
            if (cos_theta < min_cos_theta) {
                min_cos_theta = cos_theta;
                tij = t;
                Rij = R;
                obsj = obsj_tmp;
                d = t.norm();
            }
            // if( d < t.norm() )  // 选择最远的那俩帧进行三角化
            // {
            //     d = t.norm();
            //     tij = t;
            //     Rij = R;
            //     obsj = it_per_frame.lineobs;      // 特征的图像坐标
            // }

        }

        // if the distance between two frame is lower than 0.1m or the parallax angle is lower than 15deg , do not triangulate.
        // if(d < 0.1 || min_cos_theta > 0.998)
#if 0
        if (min_cos_theta > 0.998){
        //if (min_cos_theta > 0.98) {
            // if( d < 0.2 )
            continue;
        }
#else
        // if the distance between two frame is lower than 0.3m or the parallax angle is lower than 10 deg , do not triangulate.
        if(d < 0.25 || min_cos_theta > 0.998){
            // if(d < 1.0 || min_cos_theta > 0.998){
            // if(d < 0.3 || min_cos_theta > 0.998){
            // if(d < 0.5 || min_cos_theta > 0.998){
            // if(d < 1.0 || min_cos_theta > 0.9962){
            // if(d < 0.5 || min_cos_theta > 0.9962){
            // if(d < 0.25 || min_cos_theta > 0.9962){
            // if(d < 0.2 || min_cos_theta > 0.99){
            /// if(d < 1 || min_cos_theta > 0.9848){
            // if(d < 0.3 || min_cos_theta > 0.9848){
            // if(d < 0.5 || min_cos_theta > 0.9848){
            // 20 deg
            // if(d < 0.3 || min_cos_theta > 0.9397){
            // 15 deg
            // if(d < 0.3 || min_cos_theta > 0.9659){

            continue;
        }

#endif

        Vector6d plk_cc = triangulateLine_svd(Rwc_list, twc_list, obs_list, cos_theta_list);

        // plane pi from jth obs in ith camera frame
        Vector3d p3( obsj(0), obsj(1), 1 );
        Vector3d p4( obsj(2), obsj(3), 1 );
        p3 = Rij * p3 + tij;
        p4 = Rij * p4 + tij;
        Vector4d pij = pi_from_ppp(p3, p4,tij);

        Vector6d plk = pipi_plk( pii, pij );


        Eigen::Matrix<double, 6,2> plks;
        double plk_cc_norm = plk_cc.norm();
        double plk_norm = plk.norm();
        plks.leftCols(1) = plk_cc / plk_cc_norm;
        plks.rightCols(1) = plk / plk_norm;
#if 1
        std::cout<<"plks:\n"<<plks.transpose()<<std::endl;
#endif


#if 1
        if(USE_MULTI_LINE_TRIANGULATION) {
            /// roger @ 20220217
            plk = plk_cc;
        }
#endif



        Vector3d n = plk.head(3);
        Vector3d v = plk.tail(3);


        /*
        Eigen::Matrix<double, 6,2> plks;
        double plk_cc_norm = plk_cc.norm();
        double plk_norm = plk.norm();
        plks.leftCols(1) = plk_cc / plk_cc_norm;
        plks.rightCols(1) = plk / plk_norm;
        std::cout<<"plks:\n"<<plks.transpose()<<std::endl;
        */

        //Vector3d cp = plucker_origin( n, v );
        //if ( cp(2) < 0 )
        {
            //  cp = - cp;
            //  continue;
        }

        //Vector6d line;
        //line.head(3) = cp;
        //line.tail(3) = v;
        //it_per_id.line_plucker = line;

        // plk.normalize();
#if 1
        it_per_id.line_plucker = plk;  // plk in camera frame
#else
        it_per_id.line_plucker = plk_cc;
#endif
        it_per_id.is_triangulation = true;

        //  used to debug
        Vector3d pc, nc, vc;
        nc = it_per_id.line_plucker.head(3);
        vc = it_per_id.line_plucker.tail(3);


        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs_startframe = it_per_id.linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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

        Vector3d pts_1(e1(0),e1(1),e1(2));
        Vector3d pts_2(e2(0),e2(1),e2(2));

        Vector3d w_pts_1 =  Rs[imu_i] * (ric[0] * pts_1 + tic[0]) + Ps[imu_i];
        Vector3d w_pts_2 =  Rs[imu_i] * (ric[0] * pts_2 + tic[0]) + Ps[imu_i];
        it_per_id.ptw1 = w_pts_1;
        it_per_id.ptw2 = w_pts_2;

        //if(isnan(cp(0)))
        {

            //it_per_id.is_triangulation = false;

            //std::cout <<"------------"<<std::endl;
            //std::cout << line << "\n\n";
            //std::cout << d <<"\n\n";
            //std::cout << Rij <<std::endl;
            //std::cout << tij <<"\n\n";
            //std::cout <<"obsj: "<< obsj <<"\n\n";
            //std::cout << "p3: " << p3 <<"\n\n";
            //std::cout << "p4: " << p4 <<"\n\n";
            //std::cout <<pi_from_ppp(p3, p4,tij)<<std::endl;
            //std::cout << pij <<"\n\n";

        }


    }

//    removeLineOutlier(Ps,tic,ric);
}
//
void FeatureManager::triangulateLine(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    //std::cout<<"in triangulateLine() linefeature size: "<<linefeature.size()<<std::endl;
    printf("in triangulateLine() , LINE_MIN_OBS: %d, linefeature size: %d \n", LINE_MIN_OBS, linefeature.size());
    for (auto &it_per_id : linefeature)        // 遍历每个特征，对新特征进行三角化
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();    // 已经有多少帧看到了这个特征
        // printf("line used_num: %d \n", it_per_id.used_num);
        if (!(it_per_id.used_num >= LINE_MIN_OBS &&
              it_per_id.start_frame < WINDOW_SIZE - 2))   // 看到的帧数少于2， 或者 这个特征最近倒数第二帧才看到， 那都不三角化
            continue;

        if (it_per_id.is_triangulation)       // 如果已经三角化了
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        /// roger line
        ///    ROS_ASSERT(NUM_OF_CAM == 1);

        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        double d = 0, min_cos_theta = 1.0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obsi, obsj;  // obs from two frame are used to do triangulation

        // plane pi from ith obs in ith camera frame
        Eigen::Vector4d pii;
        Eigen::Vector3d ni;      // normal vector of plane
        int trace_num = it_per_id.linefeature_per_frame.size();
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;

            if (imu_j == imu_i)   // 第一个观测是start frame 上
            {
                obsi = it_per_frame.lineobs.head(4);
                Eigen::Vector3d p1(obsi(0), obsi(1), 1);
                Eigen::Vector3d p2(obsi(2), obsi(3), 1);
                pii = pi_from_ppp(p1, p2, Vector3d(0, 0, 0));
                ni = pii.head(3);
                ni.normalize();
                continue;
            }
#if 0
            if ((imu_j - imu_i) != (trace_num - 1) && (imu_i != imu_j)){
                continue;
            }
#endif
            // 非start frame(其他帧)上的观测
            /// 相机在世界坐标系下的pose
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            Eigen::Vector3d t = R0.transpose() * (t1 - t0);   // tij
            Eigen::Matrix3d R = R0.transpose() * R1;          // Rij

            Eigen::Vector4d obsj_tmp = it_per_frame.lineobs.head(4);

            // plane pi from jth obs in ith camera frame
            Vector3d p3(obsj_tmp(0), obsj_tmp(1), 1);
            Vector3d p4(obsj_tmp(2), obsj_tmp(3), 1);
            p3 = R * p3 + t;
            p4 = R * p4 + t;
            Vector4d pij = pi_from_ppp(p3, p4, t);
            Eigen::Vector3d nj = pij.head(3);
            nj.normalize();

            double cos_theta = ni.dot(nj);
            /// 角度阈值和距离阈值中，角度阈值的优先级更高一点? 所以先判断角度
            if (cos_theta < min_cos_theta) {
                min_cos_theta = cos_theta;
                tij = t;
                Rij = R;
                obsj = obsj_tmp;
                d = t.norm();
            }
            // if( d < t.norm() )  // 选择最远的那俩帧进行三角化
            // {
            //     d = t.norm();
            //     tij = t;
            //     Rij = R;
            //     obsj = it_per_frame.lineobs;      // 特征的图像坐标
            // }

        }

        // if the distance between two frame is lower than 0.1m or the parallax angle is lower than 15deg , do not triangulate.
        // if(d < 0.1 || min_cos_theta > 0.998)
#if 0
         if (min_cos_theta > 0.998){
        //if (min_cos_theta > 0.98) {
            // if( d < 0.2 )
            continue;
        }
#else
        // if the distance between two frame is lower than 0.3m or the parallax angle is lower than 10 deg , do not triangulate.
        if(d < 0.25 || min_cos_theta > 0.998){
        // if(d < 1.0 || min_cos_theta > 0.998){
        // if(d < 0.3 || min_cos_theta > 0.998){
        // if(d < 0.5 || min_cos_theta > 0.998){
        // if(d < 1.0 || min_cos_theta > 0.9962){
        // if(d < 0.5 || min_cos_theta > 0.9962){
        // if(d < 0.25 || min_cos_theta > 0.9962){
        // if(d < 0.2 || min_cos_theta > 0.99){
        /// if(d < 1 || min_cos_theta > 0.9848){
        // if(d < 0.3 || min_cos_theta > 0.9848){
        // if(d < 0.5 || min_cos_theta > 0.9848){
                                     // 20 deg
        // if(d < 0.3 || min_cos_theta > 0.9397){
                                     // 15 deg
        // if(d < 0.3 || min_cos_theta > 0.9659){

            continue;
        }

#endif
        // plane pi from jth obs in ith camera frame
        Vector3d p3( obsj(0), obsj(1), 1 );
        Vector3d p4( obsj(2), obsj(3), 1 );
        p3 = Rij * p3 + tij;
        p4 = Rij * p4 + tij;
        Vector4d pij = pi_from_ppp(p3, p4,tij);

        Vector6d plk = pipi_plk( pii, pij );
        Vector3d n = plk.head(3);
        Vector3d v = plk.tail(3);

        //Vector3d cp = plucker_origin( n, v );
        //if ( cp(2) < 0 )
        {
            //  cp = - cp;
            //  continue;
        }

        //Vector6d line;
        //line.head(3) = cp;
        //line.tail(3) = v;
        //it_per_id.line_plucker = line;

        // plk.normalize();
        it_per_id.line_plucker = plk;  // plk in camera frame
        it_per_id.is_triangulation = true;

        //  used to debug
        Vector3d pc, nc, vc;
        nc = it_per_id.line_plucker.head(3);
        vc = it_per_id.line_plucker.tail(3);


        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs_startframe = it_per_id.linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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

        Vector3d pts_1(e1(0),e1(1),e1(2));
        Vector3d pts_2(e2(0),e2(1),e2(2));

        Vector3d w_pts_1 =  Rs[imu_i] * (ric[0] * pts_1 + tic[0]) + Ps[imu_i];
        Vector3d w_pts_2 =  Rs[imu_i] * (ric[0] * pts_2 + tic[0]) + Ps[imu_i];
        it_per_id.ptw1 = w_pts_1;
        it_per_id.ptw2 = w_pts_2;

        //if(isnan(cp(0)))
        {

            //it_per_id.is_triangulation = false;

            //std::cout <<"------------"<<std::endl;
            //std::cout << line << "\n\n";
            //std::cout << d <<"\n\n";
            //std::cout << Rij <<std::endl;
            //std::cout << tij <<"\n\n";
            //std::cout <<"obsj: "<< obsj <<"\n\n";
            //std::cout << "p3: " << p3 <<"\n\n";
            //std::cout << "p4: " << p4 <<"\n\n";
            //std::cout <<pi_from_ppp(p3, p4,tij)<<std::endl;
            //std::cout << pij <<"\n\n";

        }


    }

//    removeLineOutlier(Ps,tic,ric);
}

/**
 *  @brief  stereo line triangulate
 */
void FeatureManager::triangulateLine(double baseline)
{
    for (auto &it_per_id : linefeature)        // 遍历每个特征，对新特征进行三角化
    {
        it_per_id.used_num = it_per_id.linefeature_per_frame.size();    // 已经有多少帧看到了这个特征
        // TODO: 右目没看到
        if (it_per_id.is_triangulation || it_per_id.used_num < 2)  // 已经三角化了 或者 少于两帧看到 或者 右目没有看到
            continue;

        int imu_i = it_per_id.start_frame;

#if 0
        Vector4d lineobs_l,lineobs_r;
#else
        Vector5d lineobs_l,lineobs_r;
#endif
        lineFeaturePerFrame it_per_frame = it_per_id.linefeature_per_frame.front();
        lineobs_l = it_per_frame.lineobs.head(4);
        lineobs_r = it_per_frame.lineobs_R.head(4);

        // plane pi from ith left obs in ith left camera frame
        Vector3d p1( lineobs_l(0), lineobs_l(1), 1 );
        Vector3d p2( lineobs_l(2), lineobs_l(3), 1 );
        Vector4d pii = pi_from_ppp(p1, p2,Vector3d( 0, 0, 0 ));

        // plane pi from ith right obs in ith left camera frame
        Vector3d p3( lineobs_r(0) + baseline, lineobs_r(1), 1 );
        Vector3d p4( lineobs_r(2) + baseline, lineobs_r(3), 1 );
        Vector4d pij = pi_from_ppp(p3, p4,Vector3d(baseline, 0, 0));

        Vector6d plk = pipi_plk( pii, pij );
        Vector3d n = plk.head(3);
        Vector3d v = plk.tail(3);

        //Vector3d cp = plucker_origin( n, v );
        //if ( cp(2) < 0 )
        {
            //  cp = - cp;
            //  continue;
        }

        //Vector6d line;
        //line.head(3) = cp;
        //line.tail(3) = v;
        //it_per_id.line_plucker = line;

        // plk.normalize();
        it_per_id.line_plucker = plk;  // plk in camera frame
        it_per_id.is_triangulation = true;

        //if(isnan(cp(0)))
        {

            //it_per_id.is_triangulation = false;

            //std::cout <<"------------"<<std::endl;
            //std::cout << line << "\n\n";
            //std::cout << d <<"\n\n";
            //std::cout << Rij <<std::endl;
            //std::cout << tij <<"\n\n";
            //std::cout <<"obsj: "<< obsj <<"\n\n";
            //std::cout << "p3: " << p3 <<"\n\n";
            //std::cout << "p4: " << p4 <<"\n\n";
            //std::cout <<pi_from_ppp(p3, p4,tij)<<std::endl;
            //std::cout << pij <<"\n\n";

        }


    }

    removeLineOutlier();

}
//*/

/*
 // 此段代码用于仿真验证
void FeatureManager::triangulateLine(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    //std::cout<<"linefeature size: "<<linefeature.size()<<std::endl;
    // create two noise generator
    std::default_random_engine generator;
    std::default_random_engine generator_;
    std::normal_distribution<double> pixel_n_;

    std::normal_distribution<double> pixel_n(0.0, 1./500);
    std::normal_distribution<double> nt(0., 0.1);         // 10cm
    std::normal_distribution<double> nq(0., 1*M_PI/180);  // 2deg

    generator_ = generator;
    pixel_n_ = pixel_n;

    // 产生虚假观测
    // transform the landmark to world frame
    Eigen::Matrix4d Twl = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d Rwl;
    Rwl = Eigen::AngleAxisd(M_PI/5,Eigen::Vector3d::UnitZ())
          *Eigen::AngleAxisd(M_PI/5,Eigen::Vector3d::UnitY())
          *Eigen::AngleAxisd(-M_PI/5,Eigen::Vector3d::UnitX());
    Eigen::Vector3d twl(0.1, -0.1, 10.);
    //Eigen::Vector3d twl(15.0, 1.0, -1.);
    Twl.block(0, 0, 3, 3) = Rwl;
    Twl.block(0, 3, 3, 1) = twl;

    double cube_size = 6.0;
    Eigen::Vector4d pt0( 0.0, 0.0, 0.0, 1 );
    Eigen::Vector4d pt1( cube_size, 0.0, 0.0, 1 );          // line 1  = pt0 -->pt1
    Eigen::Vector4d pt2( 0.0, -cube_size, 0.0, 1);          // line 2  = pt0 -->pt2
    Eigen::Vector4d pt3( 0.0 , 0.0, cube_size, 1);    // line 3  = pt0 -->pt3
    pt0 = Twl * pt0;
    pt1 = Twl * pt1;
    pt2 = Twl * pt2;
    pt3 = Twl * pt3;


    int line_type = 0;
    for (auto &it_per_id : linefeature)        // 遍历每个特征，对新特征进行三角化
    {

        it_per_id.used_num = it_per_id.linefeature_per_frame.size();    // 已经有多少帧看到了这个特征
        if (!(it_per_id.used_num >= LINE_MIN_OBS && it_per_id.start_frame < WINDOW_SIZE - 2))   // 看到的帧数少于2， 或者 这个特征最近倒数第二帧才看到， 那都不三角化
            continue;

        // 挑选一条直线
        Eigen::Vector4d pc0 = pt0;
        Eigen::Vector4d pc1;
        switch(line_type)
        {
            case 0: {
                pc1 = pt1;
                line_type = 1;
                break;
            }
            case 1: {
                pc1 = pt2;
                line_type = 2;
                break;
            }
            case 2: {
                pc1 = pt3;
                line_type = 0;
                break;
            }
        }

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);

        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        // tranfrom line to camera
        Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();

        //Eigen::Vector3d tcw = - R0.transpose() * t0;
        //Eigen::Matrix3d Rcw = R0.transpose();
        //Tcw.block(0, 0, 3, 3) = Rcw;
        //Tcw.block(0, 3, 3, 1) = tcw;

        Eigen::Vector4d pc0i = Tcw * pc0;
        Eigen::Vector4d pc1i= Tcw * pc1;

        double d = 0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obsi, obsj, temp_obsj;
        for (auto &it_per_frame : it_per_id.linefeature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;

            Vector4d noise;
            noise << pixel_n_(generator_),pixel_n_(generator_),pixel_n_(generator_),pixel_n_(generator_);
            //noise.setZero();

            if(imu_j == imu_i)   // 第一个观测是start frame 上
            {
                obsi << pc0i(0) / pc0i(2), pc0i(1) / pc0i(2),
                        pc1i(0) / pc1i(2), pc1i(1) / pc1i(2);
                obsi = obsi + noise;
                it_per_frame.lineobs = obsi;
                //obsi = it_per_frame.lineobs;
                continue;
            }

            // std::cout<< "line tri: "<<imu_j<<std::endl;
            // 非start frame(其他帧)上的观测
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            Eigen::Vector3d t = R0.transpose() * (t1 - t0);   // tij
            Eigen::Matrix3d R = R0.transpose() * R1;          // Rij

            Eigen::Matrix4d Tji = Eigen::Matrix4d::Identity();
            Eigen::Vector3d tji = - R.transpose() * t;
            Eigen::Matrix3d Rji = R.transpose();
            Tji.block(0, 0, 3, 3) = Rji;
            Tji.block(0, 3, 3, 1) = tji;

            Eigen::Vector4d pc0j = Tji * pc0i;
            Eigen::Vector4d pc1j= Tji * pc1i;

            temp_obsj << pc0j(0) / pc0j(2), pc0j(1) / pc0j(2),
                    pc1j(0) / pc1j(2), pc1j(1) / pc1j(2);
            temp_obsj = temp_obsj + noise;
            it_per_frame.lineobs = temp_obsj;      // 特征的图像坐标
            if( d < t.norm() )  // 选择最远的那俩帧进行三角化
            {
                d = t.norm();
                tij = t;
                Rij = R;
                obsj = it_per_frame.lineobs;
            }

        }
        if(d < 0.15) // 如果小于15cm， 不三角化
            continue;

        // plane pi from ith obs in ith camera frame
        Vector3d p1( obsi(0), obsi(1), 1 );
        Vector3d p2( obsi(2), obsi(3), 1 );
        Vector4d pii = pi_from_ppp(p1, p2,Vector3d( 0, 0, 0 ));

        // plane pi from jth obs in ith camera frame
        Vector3d p3( obsj(0), obsj(1), 1 );
        Vector3d p4( obsj(2), obsj(3), 1 );
        p3 = Rij * p3 + tij;
        p4 = Rij * p4 + tij;
        Vector4d pij = pi_from_ppp(p3, p4,tij);

        Vector6d plk = pipi_plk( pii, pij );
        Vector3d n = plk.head(3);
        Vector3d v = plk.tail(3);
        Vector3d v1 = (pc0 - pc1).head(3);

        Vector6d line;
        line.head(3) = n;
        line.tail(3) = v;

        it_per_id.line_plucker = line;
        it_per_id.is_triangulation = true;

        it_per_id.line_plk_init = line;
        it_per_id.obs_init = obsi;

//-----------------------------------------------
        //  used to debug
        //std::cout <<"tri: "<< it_per_id.feature_id <<" " << it_per_id.line_plucker <<"\n";
        Vector3d pc, nc, vc;
        nc = it_per_id.line_plucker.head(3);
        vc = it_per_id.line_plucker.tail(3);

        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs_startframe = it_per_id.linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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

        Vector3d pts_1(e1(0),e1(1),e1(2));
        Vector3d pts_2(e2(0),e2(1),e2(2));

        Vector3d w_pts_1 =  Rs[imu_i] * (ric[0] * pts_1 + tic[0]) + Ps[imu_i];
        Vector3d w_pts_2 =  Rs[imu_i] * (ric[0] * pts_2 + tic[0]) + Ps[imu_i];
        it_per_id.ptw1 = w_pts_1;
        it_per_id.ptw2 = w_pts_2;
        it_per_id.Ri_ = Rs[imu_i];
        it_per_id.ti_ = Ps[imu_i];

        //std::cout<<"---------------------------\n";
        //std::cout << w_pts_1 <<"\n" << w_pts_2 <<"\n\n";

        //   -----
        imu_j = imu_i + 1;
        Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
        Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
        it_per_id.Rj_ = Rs[imu_j];
        it_per_id.tj_ = Ps[imu_j];
        it_per_id.obs_j = it_per_id.linefeature_per_frame[imu_j - imu_i].lineobs;

        Eigen::Vector3d t = R1.transpose() * (t0 - t1);   // Rcjw * (twci - twcj)
        Eigen::Matrix3d R = R1.transpose() * R0;          // Rij

        Vector6d plk_j = plk_to_pose(it_per_id.line_plucker, R, t);

        nc = plk_j.head(3);
        vc = plk_j.tail(3);

        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        obs_startframe = it_per_id.linefeature_per_frame[imu_j - imu_i].lineobs;   // 第一次观测到这帧
        p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
        ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        cam = Vector3d( 0, 0, 0 );

        pi1 = pi_from_ppp(cam, p11, p12);
        pi2 = pi_from_ppp(cam, p21, p22);

        e1 = Lc * pi1;
        e2 = Lc * pi2;
        e1 = e1/e1(3);
        e2 = e2/e2(3);

        pts_1 = Vector3d(e1(0),e1(1),e1(2));
        pts_2 = Vector3d(e2(0),e2(1),e2(2));

        w_pts_1 =  R1 * pts_1 + t1;
        w_pts_2 =  R1 * pts_2 + t1;

        //std::cout << w_pts_1 << "\n" <<w_pts_2 <<"\n";

    }
    removeLineOutlier(Ps,tic,ric);
}
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const FeatureFrame &image, const LineFeatureFrame &lineFeatureImage ,double td)
{
//    ROS_DEBUG("input feature: %d", (int)image.size());
  //  ROS_DEBUG("num of feature: %d", getFeatureCount());

    //printf("input current frame feature (all): %d \n", (int)image.size());
    //printf("!!!!! num of current frame feature (it.used_num >= 4 && it.good_for_solving): %d \n", getFeatureCount());
    printf("!!!!! [input_current_frame_feature_all,           features in sliding widow]: [%d %d] \n", (int)image.size(), feature.size());
    printf("!!!!! num of stable features in sliding window (it.used_num >= 4 && it.good_for_solving): %d \n", getFeatureCount());

    if(USE_LINE) {
        printf("!!!!! [input_current_frame_line_feature_all, line features in sliding widow]: [%d %d] \n",
               (int) lineFeatureImage.size(), linefeature.size());
        printf("!!!!! num of stable line features in sliding window (it.used_num >= %d && it.good_for_solving): %d \n",
               LINE_MIN_OBS, getLineFeatureCount());
    }
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    /// 遍历当前帧中的所有特征点, FeatureFrame以map形式存储
    int long_track_cnt = 0;
    int trace_size = 0;
    for (auto &id_pts : image)
    {    //trace_size = 0;
        //int trace_size = 0;
        /// id_pts.second是map的value值，这个value值是个vector类型的，而vector的每个元素都是个pair，
        /// vector[0].first一定是cam0（因为在arrange ff时，总是先emplace_back cam0的元素）
        /// setup_feature_frame 总是先emplace_back cam0，再emplace_back cam1

        /*
         * FeatureFrame BaseFisheyeFeatureTracker<CvMat>::setup_feature_frame() {
    FeatureFrame ff;
    ///                                              特征点id      [在cube_map上跟踪到的2d点坐标] [通过t0 t1 t2 t3 t4转化到单位球上的去畸变后的坐标] [单位球坐标下的速度]
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_top, cur_up_top_pts, cur_up_top_un_pts, up_top_vel, 0);
    BaseFeatureTracker::setup_feature_frame(ff, ids_up_side, cur_up_side_pts, cur_up_side_un_pts, up_side_vel, 0);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_top, cur_down_top_pts, cur_down_top_un_pts, down_top_vel, 1);
    BaseFeatureTracker::setup_feature_frame(ff, ids_down_side, cur_down_side_pts, cur_down_side_un_pts, down_side_vel, 1);

    return ff;
}
         */

        // TODO id_pts.first特征点id
        // TODO id_pts.second 是 vector<pair<camId, double<1x8>>>
        // TODO id_pts.second[0] 一定是 <cam0, double<1x8>
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        //In common stereo; the pts in left must in right
        ///But for stereo fisheye; this is not true due to the downward top view
        //We need to modified this to enable downward top view
        // assert(id_pts.second[0].first == 0);
        /// 默认时当前帧的特征点都是检测在camera 0中的
        if (id_pts.second[0].first != 0) {
            ///This point is right/down observation only, and not observed in left/up fisheye
            f_per_fra.camera = 1;
        }


        //std::cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$sss$$$$$$$$ id_pts.second.size(): "<< id_pts.second.size()<<std::endl;

        /// vector的size==2表示这是由up_side_img跟踪到down_side_img中的(用sgbm匹配到的，而不是光流)，其他情况vector的size恒等于1
        if(id_pts.second.size() == 2)
        {
            // ROS_INFO("Stereo feature %d", id_pts.first);
            /// vector<pair<int1, double8>>;
            /// 在调用函数rightObservation()时，is_stereo也被同时置为1了
            f_per_fra.rightObservation(id_pts.second[1].second);
            //printf("is_stereo: %d\n",f_per_fra.is_stereo);
             assert(id_pts.second[1].first == 1);
            if (id_pts.second[1].first != 1) {
                ROS_WARN("Bug occurs on pt, skip");
                continue;
            }
        }

        int feature_id = id_pts.first;

        //int trace_size = 0;
        if (feature.find(feature_id) == feature.end()) {
            //Insert
            /// 没找到，说明这个特征点id第一次出现，所以要新建一个特征点id
            FeaturePerId fre(feature_id, frame_count);
            /// 这个点可以只出现在down_top_img中，所以此时它的main_cam是1
            fre.main_cam = f_per_fra.camera;
            //std::cout << "feature id: "<<feature_id<< ", newly detected pix: "<<f_per_fra.uv.transpose()<<std::endl;
            ///往map里insert一个新featId
            feature.emplace(feature_id, fre);

            ///     精彩的写法，数据结构的妙用
            // TODO 精彩的写法，数据结构的妙用
            feature[feature_id].feature_per_frame.push_back(f_per_fra);
            //std::cout << "feature id: "<<feature_id<< ", newly detected pix2 : "<<feature[feature_id].feature_per_frame[0].uv.transpose()<<std::endl;
            new_feature_num++;
        } else {
            feature[feature_id].feature_per_frame.push_back(f_per_fra);
            //std::cout << "tracked feature's host pix: "<<feature[feature_id].feature_per_frame.begin()->uv.transpose()<<std::endl;
            //std::cout << "feature id: "<<feature_id<< ", tracked feature's host pix: "<<feature[feature_id].feature_per_frame[0].uv.transpose()<<std::endl;
            last_track_num++;
            trace_size = feature[feature_id].feature_per_frame.size();
            if( feature[feature_id].feature_per_frame.size() >= 4)
                long_track_num++;
        }
        //long_track_cnt = max(long_track_cnt, trace_size);
        if (trace_size >= WINDOW_SIZE){
            long_track_cnt++;
        }
    }


/// roger line
if(USE_LINE) {
    for (auto &id_line : lineFeatureImage)   //遍历当前帧上的特征
    {
        lineFeaturePerFrame f_per_fra(id_line.second[0].second);  // 观测

        int feature_id = id_line.first;
        //cout << "line id: "<< feature_id << "\n";
        auto it = find_if(linefeature.begin(), linefeature.end(), [feature_id](const lineFeaturePerId &it) {
            return it.feature_id == feature_id;    // 在feature里找id号为feature_id的特征
        });

        if (it == linefeature.end())  // 如果之前没存这个特征，说明是新的
        {
            linefeature.push_back(lineFeaturePerId(feature_id, frame_count));
            linefeature.back().linefeature_per_frame.push_back(f_per_fra);
        } else if (it->feature_id == feature_id) {
            it->linefeature_per_frame.push_back(f_per_fra);
            it->all_obs_cnt++;
        }
    }
}




    //trace_size=1;
    if (trace_size > KEYFRAME_LONGTRACK_THRES){

        //long_track_cnt++;
        //long_track_cnt = max(long_track_cnt, trace_size);
    }
    //long_track_cnt = max(long_track_cnt, trace_size);

    /// new_feature_num 当前帧新检测的特征点个数
    /// last_track_num 上一次持续跟踪的特征点个数
    /// long_track_num 上一次持续跟踪的特征点个数 中 跟踪了4帧以上的特征点个数

    double long_track_ratio = (double)long_track_num / (double)last_track_num;
    //double long_last_ratio = double(long_track_num) / (double)last_track_num;



    //if (frame_count < 2 || last_track_num < 20 || long_track_num < KEYFRAME_LONGTRACK_THRES || new_feature_num > 0.5 * last_track_num) {
    //if (frame_count < 2 || last_track_num < 20 || long_track_num < KEYFRAME_LONGTRACK_THRES || new_feature_num > 0.85 * last_track_num) {
    //if (frame_count < 2 || last_track_num < KEYFRAME_LONGTRACK_THRES + 20 || long_track_num < KEYFRAME_LONGTRACK_THRES || new_feature_num > 0.5 * last_track_num) {
    //if (frame_count < 2 || last_track_num < KEYFRAME_LONGTRACK_THRES + 10 || long_track_num < KEYFRAME_LONGTRACK_THRES || new_feature_num > 0.85 * last_track_num) {
    //if (frame_count < 2 || last_track_num < KEYFRAME_LONGTRACK_THRES + 10 || ((long_track_num < KEYFRAME_LONGTRACK_THRES) && (long_track_ratio < LONG_TRACK_RATIO))|| new_feature_num > 0.85 * last_track_num) {


     if (frame_count < 2 || last_track_num < KEYFRAME_LONGTRACK_THRES + 10 || ((long_track_num < KEYFRAME_LONGTRACK_THRES) || (long_track_ratio < LONG_TRACK_RATIO)) || new_feature_num > 0.85 * last_track_num) {
/// roger @ 20220202
 ///  if (frame_count < 2 || last_track_num < KEYFRAME_LONGTRACK_THRES + 10 || ((long_track_num < KEYFRAME_LONGTRACK_THRES) || (long_track_ratio < LONG_TRACK_RATIO)) || ((-new_feature_num) > 0.85 * last_track_num)) {
        ROS_DEBUG("Add KF LAST %d LONG %d new %d", last_track_num, long_track_num, new_feature_num);
        printf("##### @@@@@ ##### @@@@@ Add KF LAST %d LONG %d new %d, LONG_TRACK_RATIO: %0.2f, long_track_ratio: %0.2f\n", last_track_num, long_track_num, new_feature_num, LONG_TRACK_RATIO, long_track_ratio);
        return true;
    }
///  遍历feature_manager中的所有feature id
/// 每个feature都是个 map<int, FeaturePerId>   feature_per_id
    for (auto &_it : feature)
    {
        auto & it_per_id = _it.second;
        if (///frame_count代表当前帧号，这个判断表示start_frame是在2帧之前，即这个feature id已经被跟踪超过2帧
                it_per_id.start_frame <= frame_count - 2
            &&
            /// 表示这个feature id从start_frame开始一直被跟踪到当前帧（即一直到最新帧仍没跟丢）
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            /// 这个feature id在次次新帧，和次新帧处的像素点位移值
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        ROS_DEBUG("Add KF: Parallax num ==0");
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);

        ///  * FOCAL_LENGTH 表示将量纲转为像素
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        printf("============================================ ##### ##### @@@@@ KEYFRAME_LONGTRACK_THRES: %d, last_track_num: %d, long_track_num: %d, new_feature_num: %d, LONG_TRACK_RATIO: %0.2f, long_track_ratio: %0.2f\n", KEYFRAME_LONGTRACK_THRES, last_track_num, long_track_num, new_feature_num,LONG_TRACK_RATIO, long_track_ratio);
        std::cout<< "================================================================================= FOCAL_LENGTH: "<<FOCAL_LENGTH<<", MIN_PARALLAX: "<<MIN_PARALLAX*FOCAL_LENGTH<<", last_average_parallax: "<<last_average_parallax<<std::endl; //", tracked pts longer than ["<<KEYFRAME_LONGTRACK_THRES<<"] frames: "<<long_track_cnt<<std::endl;
        bool is_kf = parallax_sum / parallax_num >= MIN_PARALLAX;
        if (is_kf) {
            ROS_DEBUG("Add KF: Parallax is bigger than required == 0 parallax_sum %f parallax_num %d avg %f MIN_PARALLAX %f", 
                parallax_sum, parallax_num, parallax_sum / parallax_num, MIN_PARALLAX
            );
        }

        //std::cout<< "========================================================================== is_kf: "<<is_kf<<std::endl;
        return is_kf;
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &_it : feature)
    {
        auto & it = _it.second;
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(std::map<int, double> deps)
{
    for (auto &it : deps)
    {
        int _id = it.first;
        double depth = it.second;
        auto & it_per_id = feature[_id];

        it_per_id.used_num = it_per_id.feature_per_frame.size();

        it_per_id.estimated_depth = 1.0 / depth;
        it_per_id.need_triangulation = false;
        it_per_id.depth_inited = true;
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
            it_per_id.good_for_solving = false;
            it_per_id.depth_inited = false;
        }
        else {
            it_per_id.solve_flag = 1;
        }
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        auto & _it = it->second;
        it_next++;
        if (_it.solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        it_per_id.estimated_depth = -1;
        it_per_id.depth_inited = false;
        it_per_id.good_for_solving = false;
    }
}

std::map<int, double> FeatureManager::getDepthVector()
{
    //This function gives actually points for solving; We only use oldest max_solve_cnt point, oldest pts has good track
    //As for some feature point not solve all the time; we do re triangulate on it
    /// map<int, FeaturePerId> feature
    /// map是有序的，feature id越小，表示feature越老，feature越老表示跟踪越稳定，所以逻辑上是优先优化较老的feature
    std::map<int, double> dep_vec;
    /// 对于is_stereo的feature id，哪怕只跟踪了2帧，也可以参加优化（绿色），对于!is_stereo的feature id，只有跟踪了大于4帧的feature才可以参与优化（绿色）
    int outlierNum = 0;
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        bool id_in_outouliers = outlier_features.find(it_per_id.feature_id) != outlier_features.end();
        outlierNum = outlierNum + id_in_outouliers;
        /// roger @ 20220126, to make full use of static-stereo points
        if(it_per_id.is_stereo && it_per_id.used_num >= 2 && dep_vec.size() < MAX_SOLVE_CNT &&  it_per_id.good_for_solving && !id_in_outouliers) {
        /// if(it_per_id.is_stereo && it_per_id.used_num >= 1 && dep_vec.size() < MAX_SOLVE_CNT &&  it_per_id.good_for_solving && !id_in_outouliers && it_per_id.estimated_depth > 0 && it_per_id.estimated_depth != INIT_DEPTH) {
            dep_vec[it_per_id.feature_id] = 1. / it_per_id.estimated_depth;
            ft->setFeatureStatus(it_per_id.feature_id, 3);
        } else {
            it_per_id.need_triangulation = true;
        }
    }
    std::cout<<"------------------------------------------------------------------------------------------------------------------ dep_vec size 1 : "<< dep_vec.size()<<", outlier num 1 : "<<outlierNum<<std::endl;
    /// 对于is_stereo的feature id，哪怕只跟踪了2帧，也可以参加优化（绿色），对于!is_stereo的feature id，只有跟踪了大于4帧的feature才可以参与优化（绿色）
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;


        // TODO
        //std::cout<<"+++++++++++ delta used_num: "<< it_per_id.used_num - it_per_id.feature_per_frame.size()<<std::endl;
        //std::cout<<"+++++++++++ feature id start_frame: "<< it_per_id.start_frame<<std::endl;

        it_per_id.used_num = it_per_id.feature_per_frame.size();
        bool id_in_outouliers = outlier_features.find(it_per_id.feature_id) != outlier_features.end();
        outlierNum = outlierNum + id_in_outouliers;
        if(dep_vec.size() < MAX_SOLVE_CNT && it_per_id.used_num >= 4 
            && it_per_id.good_for_solving && !id_in_outouliers) {
            dep_vec[it_per_id.feature_id] = 1. / it_per_id.estimated_depth;
            ft->setFeatureStatus(it_per_id.feature_id, 3);
        } else {
            it_per_id.need_triangulation = true;
        }
    }
    std::cout<<"------------------------------------------------------------------------------------------------------------------ dep_vec size 2 : "<< dep_vec.size()<<", outlier num 2 : "<<outlierNum<<std::endl;
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


void FeatureManager::triangulatePoint3DPts(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector3d &point0, Eigen::Vector3d &point1, Eigen::Vector3d &point_3d)
{
    //TODO:Rewrite this for 3d point
    
    double p0x = point0[0];
    double p0y = point0[1];
    double p0z = point0[2];

    double p1x = point1[0];
    double p1y = point1[1];
    double p1z = point1[2];

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = p0x * Pose0.row(2) - p0z*Pose0.row(0);
    design_matrix.row(1) = p0y * Pose0.row(2) - p0z*Pose0.row(1);
    design_matrix.row(2) = p1x * Pose1.row(2) - p1z*Pose1.row(0);
    design_matrix.row(3) = p1y * Pose1.row(2) - p1z*Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

double FeatureManager::triangulatePoint3DPts(vector<Eigen::Matrix<double, 3, 4>> &poses, vector<Eigen::Vector3d> &points, Eigen::Vector3d &point_3d)
{
    //TODO:Rewrite this for 3d point
    Eigen::MatrixXd design_matrix(poses.size()*2, 4);
    assert(poses.size() > 0 && poses.size() == points.size() && "We at least have 2 poses and number of pts and poses must equal");
    for (unsigned int i = 0; i < poses.size(); i ++) {
        double p0x = points[i][0];
        double p0y = points[i][1];
        double p0z = points[i][2];
        design_matrix.row(i*2) = p0x * poses[i].row(2) - p0z*poses[i].row(0);
        design_matrix.row(i*2+1) = p0y * poses[i].row(2) - p0z*poses[i].row(1);

    }
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    Eigen::MatrixXd pts(4, 1);
    pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
    Eigen::MatrixXd errs = design_matrix*pts;
    // std::cout << "ERR" << errs.sum() << std::endl;
    return errs.norm()/ errs.rows(); 
}



bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    cv::Mat inliers;
    double foc = 460;
    double f_thr;
    if (!enable_down_side & !enable_down_top) {
        f_thr = 6.0; // 16; //8; //16.0; //10.0; //5.0;
    }
     else {
        f_thr = 3.0; // 8; //4;//8.0; //4.0; //2.0;
    }
#if 0
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
#else
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);
    printf("f_thr set to %0.3f\n", f_thr);
    pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, f_thr / foc, 0.99, inliers);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, false, 100, f_thr / foc, 0.99, inliers);
    std::cout<<"solvePnPRansac inliers: "<<inliers.t()<<std::endl;
    cv::Scalar ss = sum(inliers);
    //printf("solvePnPRansac inlier ratio: %d/%d \n", ss[0],inliers.size);
    printf("solvePnPRansac inlier ratio: %d/%d \n", inliers.rows, pts2D.size());
    #endif
    if(!pnp_succ | isnan(rvec.at<double>(0,0)) | isnan(t.at<double>(0,0)))
    {
        printf("pnp failed ! \n");
        return false;
    }
    //std::cout<<"########## ++++++++++ initialized with:\n "<<pts2D<<std::endl;
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{

    bool log;
    if(frameCnt > 0)
    {

        printf("########## ++++++++++ initializing with %d features in sliding window...\n", feature.size());

        int used_num_thr;
#if 1
        if (!enable_down_side & !enable_down_top) {
            used_num_thr = 0;
            printf("mono mode, used_num >= %d.\n", used_num_thr);
        }
        else {
            used_num_thr = 4;
            printf("stereo mode, used_num >= %d.\n", used_num_thr);
        }
#else
        if (frameCnt <= WINDOW_SIZE) {
            used_num_thr = 0;
            printf("initializing, used_num >= %d.\n", used_num_thr);
        }
        else {
            used_num_thr = 4;
            printf("initialized, used_num >= %d.\n", used_num_thr);
        }


#endif
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        //printf("########## ++++++++++ initializing with %d features...\n", feature.size());
        for (auto &_it : feature) {
            auto & it_per_id = _it.second;
            int index1 = frameCnt - it_per_id.start_frame;
            //printf("index1: ")
   //         printf("((int)it_per_id.feature_per_frame.size() >= index1 + 1) = %d, | it_per_id.depth_inited = %d, | it_per_id.good_for_solving= %d, | it_per_id.main_cam = %d, | it_per_id.used_num = %d \n",
     //              (int)it_per_id.feature_per_frame.size() >= index1 + 1, it_per_id.depth_inited, it_per_id.good_for_solving, it_per_id.main_cam, it_per_id.used_num);

            //printf("((int)it_per_id.feature_per_frame.size() >= index1 + 1) = %d, | it_per_id.depth_inited = %d, | it_per_id.good_for_solving= %d, | it_per_id.main_cam = %d, | it_per_id.used_num = %d, | it_per_id.is_stereo %d \n",
             //      (int)it_per_id.feature_per_frame.size() >= index1 + 1, it_per_id.depth_inited, it_per_id.good_for_solving, it_per_id.main_cam, it_per_id.used_num, it_per_id.is_stereo);
            //if (it_per_id.depth_inited && it_per_id.good_for_solving && it_per_id.main_cam == 0 && it_per_id.used_num >= 4)
            //if (it_per_id.is_stereo && it_per_id.main_cam == 0 ) //&& it_per_id.used_num >= 4)
           // printf("it_per_id.estimated_depth = %0.3f, | it_per_id.depth_inited = %d, | it_per_id.good_for_solving= %d, | it_per_id.main_cam = %d, | it_per_id.used_num = %d, | it_per_id.is_stereo %d \n",
             //      it_per_id.estimated_depth, it_per_id.depth_inited, it_per_id.good_for_solving, it_per_id.main_cam, it_per_id.used_num, it_per_id.is_stereo);

            //if (it_per_id.estimated_depth > 0 && it_per_id.main_cam == 0 && it_per_id.good_for_solving&& it_per_id.used_num >= 4)
            bool judge;


//            if (!enable_down_side & !enable_down_top) {
//                judge = it_per_id.depth_inited && it_per_id.good_for_solving && it_per_id.main_cam == 0 &&
//                        it_per_id.used_num >= 0;
//                printf("mono mode, used_num >=0.\n");
//            }
//            else {
//                judge = it_per_id.depth_inited && it_per_id.good_for_solving && it_per_id.main_cam == 0 &&
//                        it_per_id.used_num >= 4;
//                printf("stereo mode, used_num >=4.\n");
//            }


            judge = it_per_id.depth_inited && it_per_id.good_for_solving && it_per_id.main_cam == 0 &&
                    it_per_id.used_num >= used_num_thr;
#if 0
            /// hack here to make fisheye_vins_2020-01-30-10-38-14.bag runnable
            judge = false;
#endif
            if (judge)
                {
                int index = frameCnt - it_per_id.start_frame;
          //      printf("index(frameCnt - it_per_id.start_frame) = %d, it_per_id.feature_per_frame.size() = %d \n", index, it_per_id.feature_per_frame.size());


                   // printf("feat id = %d, | it_per_id.estimated_depth = %0.3f, | it_per_id.depth_inited = %d, | it_per_id.good_for_solving= %d, | it_per_id.main_cam = %d, | it_per_id.used_num = %d, | it_per_id.is_stereo %d \n",
                     //      it_per_id.feature_id, it_per_id.estimated_depth, it_per_id.depth_inited, it_per_id.good_for_solving, it_per_id.main_cam, it_per_id.used_num, it_per_id.is_stereo);

                    // printf("######### ++++++++++ (int)it_per_id.feature_per_frame.size() >= index + 1 = %d\n",(int)it_per_id.feature_per_frame.size() >= index + 1);

                    /// 加这个判断条件本质是要保证三角化的ref坐标系为滑窗中最老的一帧，所以used_num>=index+1才行，（其实是要used_num = index + 1，>=1 条件有点冗余了)(scratch that)
///还是要仔细看代码，不能想当然，恒置成true的用意是把能用的观测尽量用上，先不考虑计算量
                    /// 虽然三角化的ref需要定在滑窗的最老一帧，但depth的host帧不一定要是滑窗中的最老一帧，可以通过滑窗的pose转移到滑窗中最老帧坐标系（草率了）
                if((int)it_per_id.feature_per_frame.size() >= index + 1)
                    //if((int)it_per_id.feature_per_frame.size() >= index + 1 | it_per_id.start_frame == 0 | index == WINDOW_SIZE | (int)it_per_id.feature_per_frame.size() == index)
                    //if((int)it_per_id.feature_per_frame.size() >= index + 1 | it_per_id.start_frame == 0 | index == WINDOW_SIZE) // | (int)it_per_id.feature_per_frame.size() == index)
                    //if(true)
                    {


                    assert(it_per_id.used_num == index);
#if 0
                        assert ((int)((int)it_per_id.feature_per_frame.size() == index) == (it_per_id.is_stereo));
#else
//                        if ((int)((int)it_per_id.feature_per_frame.size() == index) == (it_per_id.is_stereo)) {
//                            log = true; //false;
//                        }
//                        else {
//                            log = true;
//                        }
#endif
                    //if(it_per_id.feature_per_frame.begin()->uv.x() == round(it_per_id.feature_per_frame.begin()->uv.x()))
                    if(true)
                    {
                        if(PRINT_LOG) {
                        //if(true) {
                        //if(log) {
#if 0
                            printf("feat id = %d, | feat coord: [%0.2f %0.2f], | estimated_depth = %0.3f, | depth_inited = %d, | good_for_solving= %d, | used_num = %d, | tracked_num = %d, | index = %d, | main_cam = %d, | is_stereo %d \n",
                                   it_per_id.feature_id, it_per_id.feature_per_frame.begin()->uv.x(),
                                   it_per_id.feature_per_frame.begin()->uv.y(), it_per_id.estimated_depth,
                                   it_per_id.depth_inited, it_per_id.good_for_solving, it_per_id.used_num,
                                   it_per_id.feature_per_frame.size(), index, it_per_id.main_cam, it_per_id.is_stereo);
#else
                            printf("feat id = %d, | feat coord: [%0.2f %0.2f], | estimated_depth = %0.3f, | used_num = %d, | tracked_num = %d, | index = %d, | main_cam = %d, | is_stereo %d \n",
                                   it_per_id.feature_id, it_per_id.feature_per_frame.begin()->uv.x(),
                                   it_per_id.feature_per_frame.begin()->uv.y(), it_per_id.estimated_depth,
                                   it_per_id.used_num,
                                   it_per_id.feature_per_frame.size(), index, it_per_id.main_cam, it_per_id.is_stereo);
#endif
                        }
                    }
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0];
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    //Because PnP require 2d points; We hack here to use only z > 1 unit sphere point to init pnp
                    // TODO use opengv::gp3p instead, and change error matrics, not reprojection error, but reprojection angle
                    Eigen::Vector3d pt = it_per_id.feature_per_frame[index].point;
                    //printf("adding feature coord: [%0.3f %0.3f %0.3f]",pt(0),pt(1),pt(2));
#if 0
                    std::cout<<"########## ++++++++++ adding feature undist: "<<pt.transpose()<<std::endl;
#endif
                    cv::Point2f point2d(pt.x()/pt.z(), pt.y()/pt.z());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d);
                }
            }
        }
        Eigen::Matrix3d RCam, RCam_;
        Eigen::Vector3d PCam, PCam_;
        // trans to w_T_cam
        printf("processing initFramePoseByPnP on frame count: %d \n",frameCnt);
        if (frameCnt <= WINDOW_SIZE) {
        //if (true) {
#if 0
            RCam = Rs[frameCnt - 1] * ric[0];
            PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
            std::cout<<"[WINDOW_SIZE] - 1 rot and trans: \n"<<RCam<<"\n"<< PCam<<std::endl;
            std::cout<<"[WINDOW_SIZE] - 0 rot and trans: \n"<< Rs[frameCnt - 0] * ric[0]<<"\n"<< Rs[frameCnt - 0] * tic[0] + Ps[frameCnt - 0]<<std::endl;
#else
            RCam_ = Rs[frameCnt - 1] * ric[0];
            PCam_ = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

            RCam = Rs[WINDOW_SIZE] * ric[0];
            PCam = Rs[WINDOW_SIZE] * tic[0] + Ps[WINDOW_SIZE];
            std::cout<<"[WINDOW_SIZE] - 1 body rot and trans: \n"<< Rs[frameCnt - 1]<<"\n"<<Ps[frameCnt - 1]<<std::endl;
            std::cout<<"[WINDOW_SIZE] - 1 rot and trans: \n"<< RCam_<<"\n"<<PCam_<<std::endl;
            std::cout<<"[WINDOW_SIZE] - 0 rot and trans: \n"<< Rs[frameCnt - 0] * ric[0]<<"\n"<<Rs[frameCnt - 0] * tic[0] + Ps[frameCnt - 0]<<std::endl;
#endif
        }
        else{
            RCam = Rs[WINDOW_SIZE] * ric[0];
            PCam = Rs[WINDOW_SIZE] * tic[0] + Ps[WINDOW_SIZE];

        }
        //std::cout<<"pts2D:\n"<<pts2D<<std::endl;
        //std::cout<<"pts3D:\n"<<pts3D<<std::endl;
        bool suc = solvePoseByPnP(RCam_, PCam_, pts2D, pts3D);
        std::cout<<"[frameCnt] - 0 rot and trans after pnp: \n"<< RCam_<<"\n"<<PCam_<<std::endl;

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            /// 这是在预测最新帧的pose，搞半天原来是在做这个事，预测滑窗中最新帧的pose，这里可能只要一个差不多的初值就行
            /// 预测次新帧到最新帧的pose(scratch that)
            /// 应该是预测滑窗中最新帧在word下的pose，所以所有的 [3d点] 一定是要被 [最新帧] 观测到的，难怪啊！！搞了这么久，就是因为不愿意沉下心看代码，主观脑补了太多东西

            if (frameCnt <= WINDOW_SIZE) {
                Rs[frameCnt] = RCam * ric[0].transpose();
                Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;
            }
            else {
                Rs[WINDOW_SIZE] = RCam * ric[0].transpose();
                Ps[WINDOW_SIZE] = -RCam * ric[0].transpose() * tic[0] + PCam;
            }
            Eigen::Quaterniond Q(Rs[frameCnt]);
            //std::cout<<"[WINDOW_SIZE] - 1 rot and trans after pnp: \n"<< RCam_<<"\n"<<PCam_<<std::endl;
            std::cout<<"[frameCnt] - 0 rot and trans after pnp: \n"<< Rs[frameCnt - 0] * ric[0]<<"\n"<<Rs[frameCnt - 0] * tic[0] + Ps[frameCnt - 0]<<std::endl;

            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            // cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
        else {

        }
    }
}

void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    int badCnt = 0;
    int newlyTriangulatedGoodCnt = 0;
    int triangulatedCnt = 0;
    Eigen::VectorXi newlyTriangulatedIds;
    Eigen::MatrixXd newlyTriangulatedInfo;
    //vector<int> newlyTriangulatedIds;
    for (auto &_it : feature) {
        auto & it_per_id = _it.second;
        //Only solving point dnot re-triangulate
        if (!it_per_id.need_triangulation) {
            triangulatedCnt++;
            continue;
        }
        if (outlier_features.find(it_per_id.feature_id) != outlier_features.end()) {
            //Is in outliers
            ft->setFeatureStatus(it_per_id.feature_id, -1);
            continue;
        }

        int main_cam_id = it_per_id.main_cam;

        std::vector<Eigen::Matrix<double, 3, 4>> poses;
        std::vector<Eigen::Vector3d> ptss;
        /// 做三角化时的reference pose，或者说时base pose（表达在世界坐标系下）
        Eigen::Matrix<double, 3, 4> origin_pose;
        auto t0 = Ps[it_per_id.start_frame] + Rs[it_per_id.start_frame] * tic[main_cam_id];
        auto R0 = Rs[it_per_id.start_frame] * ric[main_cam_id];
        /// 做三角化时的reference pose，或者说时base pose（表达在start_frame坐标系下）
        origin_pose.leftCols<3>() = R0.transpose();
        origin_pose.rightCols<1>() = -R0.transpose() * t0;
        bool has_stereo = false;

        Eigen::Vector3d _min = t0;
        Eigen::Vector3d _max = t0;

        for (unsigned int frame = 0; frame < it_per_id.feature_per_frame.size(); frame ++) {
            int imu_i = it_per_id.start_frame + frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[main_cam_id];
            /// 相机光心移动的模长，为什么取max？因为相机可能在滑窗内做往复运动，不一定是单向运动
            _max.x() = max(t0.x(), _max.x());
            _max.y() = max(t0.y(), _max.y());
            _max.z() = max(t0.z(), _max.z());
            
            _min.x() = min(t0.x(), _min.x());
            _min.y() = min(t0.y(), _min.y());
            _min.z() = min(t0.z(), _min.z());

            Eigen::Matrix3d R0 = Rs[imu_i] * ric[main_cam_id];
            /// 表达在左相机坐标系下
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            Eigen::Vector3d point0 = it_per_id.feature_per_frame[frame].point;

            poses.push_back(leftPose);
            ptss.push_back(point0);

            if(STEREO && it_per_id.feature_per_frame[frame].is_stereo) {
                //Secondary cam must be 1 now
                has_stereo = true;
                //std::cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"<<std::endl;
                Eigen::Matrix<double, 3, 4> rightPose;
                Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
                Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
                /// 表达在右相机坐标系下
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;
                auto point1 = it_per_id.feature_per_frame[frame].pointRight;
                poses.push_back(rightPose);
                ptss.push_back(point1);

                _max.x() = max(t1.x(), _max.x());
                _max.y() = max(t1.y(), _max.y());
                _max.z() = max(t1.z(), _max.z());
                
                _min.x() = min(t1.x(), _min.x());
                _min.y() = min(t1.y(), _min.y());
                _min.z() = min(t1.z(), _min.z());
            }
        }

        if (!has_stereo) {
            //We need calculate baseline
            it_per_id.is_stereo = false;
            if ((_max - _min).norm() < depth_estimate_baseline) {
                continue;
            }
        } else {
            it_per_id.is_stereo = true;
        }

        if (poses.size() < 2) {
            //No enough information
            continue;
        }

        Eigen::Vector3d point3d;
        double err = triangulatePoint3DPts(poses, ptss, point3d)*FOCAL_LENGTH;
        Eigen::Vector3d localPoint = origin_pose.leftCols<3>() * point3d + origin_pose.rightCols<1>();

        if (err > triangulate_max_err) {
            // ROS_WARN("Feature ID %d CAM %d IS stereo %d poses %ld dep %d %f AVG ERR: %f", 
            //     it_per_id.feature_id, 
            //     it_per_id.main_cam,
            //     it_per_id.feature_per_frame[0].is_stereo,
            //     poses.size(),
            //     it_per_id.depth_inited, it_per_id.estimated_depth, err);
            /// red
            if(PRINT_LOG) {
                printf("triangulate_max_err set to %0.3f, FOCAL_LENGTH set to: %0.3f, Pendding feature %d on cam %d...  error * FOCAL_LENGTH: %0.3f\n",
                       triangulate_max_err, FOCAL_LENGTH, it_per_id.feature_id, it_per_id.main_cam, err);
            }
            ft->setFeatureStatus(it_per_id.feature_id, 2);
            it_per_id.good_for_solving = false;
            it_per_id.depth_inited = false;
            it_per_id.need_triangulation = true;
            //it_per_id.estimated_depth = localPoint.norm();
            badCnt++;
        } else {
            if (it_per_id.feature_per_frame.size() >= 4) {
                /// blue
                ft->setFeatureStatus(it_per_id.feature_id, 1);            
            }
            it_per_id.depth_inited = true;
            it_per_id.good_for_solving = true;
            it_per_id.estimated_depth = localPoint.norm();
            /// 当pose位移太小，而重投影误差又不太大，保险起见，还是不相信三角化的结果，赋一个INIT_DEPTH
            if (!has_stereo && (_max - _min).norm() < depth_estimate_baseline) {
                it_per_id.estimated_depth = INIT_DEPTH;
            }
            //printf("newly triangulated feature id:");
            //newlyTriangulatedIds.push_back(it_per_id.feature_id);

            int index = frameCnt - it_per_id.start_frame;
            int used_num_ = it_per_id.feature_per_frame.size();


            if(newlyTriangulatedIds.size() == 0) {
                newlyTriangulatedIds.resize(1);
                newlyTriangulatedIds(0) = it_per_id.feature_id;

                newlyTriangulatedInfo.resize(1,15);
                newlyTriangulatedInfo.setZero();

                newlyTriangulatedInfo(0,0) = it_per_id.feature_id;
                newlyTriangulatedInfo(0,1) = used_num_;
                newlyTriangulatedInfo(0,2) = index;
                newlyTriangulatedInfo(0,3) = it_per_id.main_cam;
                newlyTriangulatedInfo(0,4) = it_per_id.is_stereo;
                newlyTriangulatedInfo(0,5) = it_per_id.feature_per_frame[0].uv(0);
                newlyTriangulatedInfo(0,6) = it_per_id.feature_per_frame[0].uv(1);
                newlyTriangulatedInfo(0,11) = it_per_id.feature_per_frame.back().uv(0);
                newlyTriangulatedInfo(0,12) = it_per_id.feature_per_frame.back().uv(1);

                newlyTriangulatedInfo(0,14) = it_per_id.estimated_depth;
                if(it_per_id.feature_per_frame.size() > 1) {
                    //newlyTriangulatedInfo(0, 5) = it_per_id.main_cam;
                    newlyTriangulatedInfo(0, 7) = it_per_id.feature_per_frame[1].uv(0);
                    newlyTriangulatedInfo(0, 8) = it_per_id.feature_per_frame[1].uv(1);
                    newlyTriangulatedInfo(0,9) = it_per_id.feature_per_frame[it_per_id.feature_per_frame.size()-2].uv(0);
                    newlyTriangulatedInfo(0,10) = it_per_id.feature_per_frame[it_per_id.feature_per_frame.size()-2].uv(1);
                }
            }
            else
            {
                // e.conservativeResize(e.size() + ekk.size());
                // e.tail(ekk.size()) = ekk;


                newlyTriangulatedIds.conservativeResize(newlyTriangulatedIds.size() + 1);
                newlyTriangulatedIds.tail(1).array() = it_per_id.feature_id;

                newlyTriangulatedInfo.conservativeResize(newlyTriangulatedInfo.rows()+1,15);

//                Eigen::MatrixXd tp;
//                tp.resize(1,12);
//                tp.setZero();
//                newlyTriangulatedInfo.bottomRows(1)= tp;

                newlyTriangulatedInfo.bottomRows(1).setZero();

                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,0) = it_per_id.feature_id;
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,1) =used_num_;
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,2) = index;
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,3) = it_per_id.main_cam;
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,4) = it_per_id.is_stereo;
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,5) = it_per_id.feature_per_frame[0].uv(0);
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,6) = it_per_id.feature_per_frame[0].uv(1);
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,11) = it_per_id.feature_per_frame.back().uv(0);
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,12) = it_per_id.feature_per_frame.back().uv(1);
                newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,14) = it_per_id.estimated_depth;
                if(it_per_id.feature_per_frame.size() > 1) {
                    //newlyTriangulatedInfo(0, 5) = it_per_id.main_cam;
                    newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1, 7) = it_per_id.feature_per_frame[1].uv(0);
                    newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1, 8) = it_per_id.feature_per_frame[1].uv(1);
                    newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,9) = it_per_id.feature_per_frame[it_per_id.feature_per_frame.size()-2].uv(0);
                    newlyTriangulatedInfo(newlyTriangulatedInfo.rows()-1,10) = it_per_id.feature_per_frame[it_per_id.feature_per_frame.size()-2].uv(1);
                }
            }
            newlyTriangulatedGoodCnt++;
        }
        // ROS_INFO("Pt3d %f %f %f LocalPt %f %f %f", point3d.x(), point3d.y(), point3d.z(), localPoint.x(), localPoint.y(), localPoint.z());
    }
    printf("&&&&&&&&&&&&&&&&&&&&&&&& [triangulated_point_cnt / new_good_point_cnt( < triangulate_max_err) / bad_point_cnt]: [%d / %d / %d] \n", triangulatedCnt,newlyTriangulatedGoodCnt, badCnt);
    //std::cout<<"newly triangulated feat id in current frame: \n"<<newlyTriangulatedIds.transpose()<<std::endl;
    //std::cout<<"newly triangulated feat info in current frame: \n"<<newlyTriangulatedInfo<<std::endl;
    std::cout<<"all triangulated *GOOD(blue or green)* feature info in sliding window: "<<newlyTriangulatedInfo.rows()<<std::endl;
    if (newlyTriangulatedInfo.rows()>=5) {
        std::cout << "[feat_id used_num index main_cam is_stereo track1 track2 trackend-1 trackend estimated_depth]: \n" << newlyTriangulatedInfo.topRows(5)<<std::endl;
        //std::cout << "[feat_id used_num index main_cam is_stereo track1 track2 trackend-1 trackend estimated_depth]: \n" << newlyTriangulatedInfo
        //          << std::endl;
    }
    else {
        std::cout << "[feat_id used_num index main_cam track1 track2]: \n" << "none. \n"
                  << std::endl;
    }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->first;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            ft->setFeatureStatus(it->second.feature_id, -1);
            feature.erase(it);
            outlier_features.insert(it->second.feature_id);
            //printf("remove outlier %d \n", index);
        }
    }
}
void FeatureManager::removeLineOutlier()
{

    for (auto it_per_id = linefeature.begin(), it_next = linefeature.begin();
         it_per_id != linefeature.end(); it_per_id = it_next)
    {
        it_next++;
        it_per_id->used_num = it_per_id->linefeature_per_frame.size();
        // TODO: 右目没看到
        if (it_per_id->is_triangulation || it_per_id->used_num < 2)  // 已经三角化了 或者 少于两帧看到 或者 右目没有看到
            continue;

        int imu_i = it_per_id->start_frame, imu_j = imu_i -1;

        // 计算初始帧上线段对应的3d端点
        Vector3d pc, nc, vc;
        nc = it_per_id->line_plucker.head(3);
        vc = it_per_id->line_plucker.tail(3);

        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs_startframe = it_per_id->linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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

    //    std::cout << "line endpoint 1 : "<<e1 << "\n "<< e2<<"\n";
#if 1
        if(e1(2) < 0 || e2(2) < 0)
        {
            linefeature.erase(it_per_id);
            continue;
        }
#endif
#if 1
        /// roger @ 20220213
        // if((e1-e2).norm() > 5)
        if((e1-e2).norm() > 10)
        {
            linefeature.erase(it_per_id);
            continue;
        }
#endif
/*
        // 点到直线的距离不能太远啊
        Vector3d Q = plucker_origin(nc,vc);
        if(Q.norm() > 5.0)
        {
            linefeature.erase(it_per_id);
            continue;
        }
*/

    }
}

void FeatureManager::removeLineOutlier(Vector3d Ps[], Vector3d tic[], Matrix3d ric[], vector<Eigen::Quaterniond> cam_faces)
{
int whole_line_num = 0;
int good_line_num = 0;
    for (auto it_per_id = linefeature.begin(), it_next = linefeature.begin();
         it_per_id != linefeature.end(); it_per_id = it_next)
    {
        it_next++;
        it_per_id->used_num = it_per_id->linefeature_per_frame.size();
        if (!(it_per_id->used_num >= LINE_MIN_OBS && it_per_id->start_frame < WINDOW_SIZE - 2 && it_per_id->is_triangulation))
            continue;

        int imu_i = it_per_id->start_frame, imu_j = imu_i -1;

        /// roger line
        /// ROS_ASSERT(NUM_OF_CAM == 1);


        double face_id1 = it_per_id->linefeature_per_frame[0].lineobs(4);
        int face_id = int(face_id1);

        Eigen::Quaterniond t0x = cam_faces[face_id];
#if 0
std::cout<<"face_id: "<<face_id<<"\nt0x:\n"<<t0x.toRotationMatrix()<<std::endl;
#endif
        Eigen::Vector3d twc_ = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc_ = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric


        Eigen::Vector3d twc = twc_; //  + R0_ * t0x;  // twx = Rwc * tcx + twc
        Eigen::Matrix3d Rwc = Rwc_ * t0x.toRotationMatrix();

        // 计算初始帧上线段对应的3d端点
        Vector3d pc, nc, vc;
        nc = it_per_id->line_plucker.head(3);
        vc = it_per_id->line_plucker.tail(3);

        //       double  d = nc.norm()/vc.norm();
        //       if (d > 5.0)
        {
            //           std::cerr <<"remove a large distant line \n";
            //           linefeature.erase(it_per_id);
            //           continue;
        }

        Matrix4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vector4d obs_startframe = it_per_id->linefeature_per_frame[0].lineobs.head(4);   // 第一次观测到这帧
        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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



    //     std::cout<<"e1:\n "<<e1.transpose()<<"\ne2:\n"<<e2.transpose()<<std::endl;
       //  std::cout << "line endpoint: "<<e1 << "\n "<< e2<<"\n";
#if 1
        if(e1(2) < 0 || e2(2) < 0)
        {
            linefeature.erase(it_per_id);
            continue;
        }
#endif
#if 1
        /// roger @ 20220213
        if((e1-e2).norm() > 5){
        // if((e1-e2).norm() > 10){
        //{
            linefeature.erase(it_per_id);
            continue;
        }
#endif
        whole_line_num++;
/*
        // 点到直线的距离不能太远啊
        Vector3d Q = plucker_origin(nc,vc);
        if(Q.norm() > 5.0)
        {
            linefeature.erase(it_per_id);
            continue;
        }
*/
        // 并且平均投影误差不能太大啊
        Vector6d line_w = plk_to_pose(it_per_id->line_plucker, Rwc, twc);  // transfrom to world frame

        int i = 0;
        double allerr = 0;
        double allerr_ang = 0;
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        Eigen::Vector4d obs;
       /// if (it_per_id->linefeature_per_frame.size() == 2) {
       ///     std::cout << "FOCAL_LENGTH: " << FOCAL_LENGTH << ", [reprojection_error dlt_theta]: ";
      ///  }
      Eigen::MatrixXd reproj_info;
      reproj_info.resize(it_per_id->linefeature_per_frame.size(), 2);
      int start_ind = 0;
        for (auto &it_per_frame : it_per_id->linefeature_per_frame)   // 遍历所有的观测， 注意 start_frame 也会被遍历
        {
            imu_j++;

            obs = it_per_frame.lineobs.head(4);
            Eigen::Vector3d t1_ = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1_ = Rs[imu_j] * ric[0];


            Eigen::Vector3d t1 = t1_; //  + R0_ * t0x;  // twx = Rwc * tcx + twc
            Eigen::Matrix3d R1 = R1_ * t0x.toRotationMatrix();

            double dlt_theta;
            double err =  reprojection_error(obs, R1, t1, line_w, dlt_theta);
            reproj_info(start_ind,0) = err * FOCAL_LENGTH;
            reproj_info(start_ind,1) = dlt_theta;
            start_ind++;
//            if(err > 0.0000001)
//                i++;
//            allerr += err;    // 计算平均投影误差
///if (it_per_id->linefeature_per_frame.size() == 2) {
 ///   std::cout << "<" << err * FOCAL_LENGTH << ", " << dlt_theta << ">, ";
///}
            if(allerr < err)    // 记录最大投影误差，如果最大的投影误差比较大，那就说明有outlier
                allerr = err;
            if(allerr_ang < dlt_theta) {
                allerr_ang = dlt_theta;
            }
        }
#if 0
        std::cout<<"reproj_info: \n"<<reproj_info.transpose()<<std::endl;
#endif
///std::cout<<std::endl;
        // std::cout<<

//        allerr = allerr / i;
        ///if (allerr > 3.0 / 500.0)
        // if (allerr > 3.0 / FOCAL_LENGTH)
        // if (allerr > 2.0 / FOCAL_LENGTH)
        // if (allerr > 0.6 / FOCAL_LENGTH)
        // if ((allerr > 1.0 / FOCAL_LENGTH) | (allerr_ang > 0.35) )
        // if ((allerr > 0.4 / FOCAL_LENGTH) | (allerr_ang > 0.25) )
        // if ((allerr > 0.5 / FOCAL_LENGTH) | (allerr_ang > 0.35) )

        // if ((allerr > 1.0 / FOCAL_LENGTH) | (allerr_ang > 0.35) )
        // if ((allerr > 1.0 / FOCAL_LENGTH) | (allerr_ang > 0.5) )
        // if ((allerr > 2.0 / FOCAL_LENGTH) | (allerr_ang > 1.0) )
        // if ((allerr > 2.0 / FOCAL_LENGTH) | (allerr_ang > 0.5) )
        // if ((allerr > 1.5 / FOCAL_LENGTH) | (allerr_ang > 0.5) )
        // if ((allerr > 2.0 / FOCAL_LENGTH) | (allerr_ang > 0.4) )
        if ((allerr > LINE_PIXEL_THRES / FOCAL_LENGTH) | (allerr_ang > LINE_ANGLE_THRES) )
        /// if (allerr > 5.11 / FOCAL_LENGTH)
        {
//            std::cout<<"remove a large error\n";
            linefeature.erase(it_per_id);
        }
        else
        {
            good_line_num++;
        }
         // std::cout<<"reproj_info: \n"<<reproj_info.transpose()<<std::endl;
    }
    printf("[good_line_num / whole_line_num]: [%d / %d]...\n", good_line_num, whole_line_num);
}
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P, vector<Eigen::Quaterniond> cam_faces)
{
    for (auto _it = feature.begin(), it_next = feature.begin();
         _it != feature.end(); _it = it_next)
    {
        auto & it = _it->second; 
        it_next++;

        if (it.start_frame != 0)
            it.start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it.feature_per_frame[0].point;  
            it.feature_per_frame.erase(it.feature_per_frame.begin());
            if (it.feature_per_frame.size() < 2)
            {
                feature.erase(_it);
                ft->setFeatureStatus(_it->second.feature_id, -1);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it.estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j.norm();

                it.depth_inited = true;
                //We retriangulate this point
                it.need_triangulation = true;
                if (FISHEYE) {
                    it.estimated_depth = pts_j.norm();
                } else {
                    if (dep_j > 0)
                        it.estimated_depth = dep_j;
                    else
                        it.estimated_depth = INIT_DEPTH;
                }
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }

    /// roger line
    if(USE_LINE) {
        for (auto it = linefeature.begin(), it_next = linefeature.begin();
             it != linefeature.end(); it = it_next) {
            it_next++;

            if (it->start_frame != 0)    // 如果特征不是在这帧上初始化的，那就不用管，只要管id--
            {
                it->start_frame--;
            } else {
/*
            //  used to debug
            Vector3d pc, nc, vc;
            nc = it->line_plucker.head(3);
            vc = it->line_plucker.tail(3);

            Matrix4d Lc;
            Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

            Vector4d obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
            Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
            Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
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

            Vector3d pts_1(e1(0),e1(1),e1(2));
            Vector3d pts_2(e2(0),e2(1),e2(2));

            Vector3d w_pts_1 =  marg_R * pts_1 + marg_P;
            Vector3d w_pts_2 =  marg_R * pts_2 + marg_P;

            std::cout<<"-------------------------------\n";
            std::cout << w_pts_1 << "\n" <<w_pts_2 <<"\n\n";
            Vector4d obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
            */
//-----------------
                double face_id1 = it->linefeature_per_frame[0].lineobs(4);
                int face_id = int(face_id1);

                Eigen::Quaterniond t0x = cam_faces[face_id];
                Eigen::Matrix4d T0x;
                T0x.setIdentity();
                T0x.topLeftCorner<3,3>() = t0x.toRotationMatrix();

                it->linefeature_per_frame.erase(it->linefeature_per_frame.begin());  // 移除观测
                if (it->linefeature_per_frame.size() < 2)                     // 如果观测到这个帧的图像少于两帧，那这个特征不要了
                {
                    linefeature.erase(it);
                    continue;
                } else  // 如果还有很多帧看到它，而我们又把这个特征的初始化帧给marg掉了，那就得把这个特征转挂到下一帧上去, 这里 marg_R, new_R 都是相应时刻的相机坐标系到世界坐标系的变换
                {
                    it->removed_cnt++;
                    // transpose this line to the new pose
                    Matrix3d new_R_, marg_R_;
                    Vector3d new_P_, marg_P_;
                    {
                      //  Eigen::Vector3d twc = twc_; //  + R0_ * t0x;  // twx = Rwc * tcx + twc
                      //  Eigen::Matrix3d Rwc = Rwc_ * t0x.toRotationMatrix();
                        new_R_ = new_R * t0x.toRotationMatrix();
                        marg_R_ = marg_R * t0x.toRotationMatrix();

                        new_P_ = new_P;
                        marg_P_ = marg_P;
                    }

#if 0
                    //new_R_ = new_R*T0x.topLeftCorner<3,3>();
                    Matrix3d Rji = new_R.transpose() * marg_R;     // Rcjw * Rwci
                    Vector3d tji = new_R.transpose() * (marg_P - new_P);
#else
                    //new_R_ = new_R*T0x.topLeftCorner<3,3>();
                    Matrix3d Rji = new_R_.transpose() * marg_R_;     // Rcjw * Rwci
                    Vector3d tji = new_R_.transpose() * (marg_P_ - new_P_);
#endif

                    Vector6d plk_j = plk_to_pose(it->line_plucker, Rji, tji);
                    it->line_plucker = plk_j;
                }
//-----------------------
/*
            //  used to debug
            nc = it->line_plucker.head(3);
            vc = it->line_plucker.tail(3);

            Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

            obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
            p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
            p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
            ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
            ln = ln / ln.norm();

            p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
            p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
            cam = Vector3d( 0, 0, 0 );

            pi1 = pi_from_ppp(cam, p11, p12);
            pi2 = pi_from_ppp(cam, p21, p22);

            e1 = Lc * pi1;
            e2 = Lc * pi2;
            e1 = e1/e1(3);
            e2 = e2/e2(3);

            pts_1 = Vector3d(e1(0),e1(1),e1(2));
            pts_2 = Vector3d(e2(0),e2(1),e2(2));

            w_pts_1 =  new_R * pts_1 + new_P;
            w_pts_2 =  new_R * pts_2 + new_P;

            std::cout << w_pts_1 << "\n" <<w_pts_2 <<"\n";
*/
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->second.start_frame != 0)
            it->second.start_frame--;
        else
        {
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin());
            if (it->second.feature_per_frame.size() == 0) {
                ft->setFeatureStatus(it->second.feature_id, -1);
                feature.erase(it);
            }
        }
    }

    /// roger line
    if(USE_LINE){
            for (auto it = linefeature.begin(), it_next = linefeature.begin();
                 it != linefeature.end(); it = it_next) {
                it_next++;

                // 如果这个特征不是在窗口里最老关键帧上观测到的，由于窗口里移除掉了一个帧，所有其他特征对应的初始化帧id都要减1左移
                // 例如： 窗口里有 0,1,2,3,4 一共5个关键帧，特征f2在第2帧上三角化的， 移除掉第0帧以后， 第2帧在窗口里的id就左移变成了第1帧，这是很f2的start_frame对应减1
                if (it->start_frame != 0)
                    it->start_frame--;
                else {
                    it->linefeature_per_frame.erase(it->linefeature_per_frame.begin());  // 删掉特征ft在这个图像帧上的观测量
                    if (it->linefeature_per_frame.size() == 0)                       // 如果没有其他图像帧能看到这个特征ft了，那就直接删掉它
                        linefeature.erase(it);
                }
            }
        }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->second.start_frame == frame_count)
        {
            it->second.start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->second.start_frame;
            if (it->second.endFrame() < frame_count - 1)
                continue;
            it->second.feature_per_frame.erase(it->second.feature_per_frame.begin() + j);
            if (it->second.feature_per_frame.size() == 0) {
                ft->setFeatureStatus(it->second.feature_id, -1);
                feature.erase(it);
            }
        }
    }

    /// roger line
    if(USE_LINE) {
        for (auto it = linefeature.begin(), it_next = linefeature.begin(); it != linefeature.end(); it = it_next) {
            it_next++;

            if (it->start_frame == frame_count)  // 由于要删去的是第frame_count-1帧，最新这一帧frame_count的id就变成了i-1
            {
                it->start_frame--;
            } else {
                int j = WINDOW_SIZE - 1 - it->start_frame;    // j指向第i-1帧
                if (it->endFrame() < frame_count - 1)
                    continue;
                it->linefeature_per_frame.erase(it->linefeature_per_frame.begin() + j);   // 删掉特征ft在这个图像帧上的观测量
                if (it->linefeature_per_frame.size() == 0)                            // 如果没有其他图像帧能看到这个特征ft了，那就直接删掉它
                    linefeature.erase(it);
            }
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    /// check the second last frame is keyframe or not
    ///parallax betwwen seconde last frame and third last frame\

    /// i：vector中倒数第二个，次次新帧的tracking点像素坐标
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    /// j：vector中倒数第一个，次新帧的tracking点像素坐标
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

#ifdef UNIT_SPHERE_ERROR
    return (frame_i.point - frame_j.point).norm();
#else
    if (FISHEYE) {
        return (frame_i.point - frame_j.point).norm();
    } else {

        double ans = 0;
        Vector3d p_j = frame_j.point;

        double u_j = p_j(0)/p_j(2);
        double v_j = p_j(1)/p_j(2);

        Vector3d p_i = frame_i.point;

        //int r_i = frame_count - 2;
        //int r_j = frame_count - 1;
        //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;
        double du = u_i - u_j, dv = v_i - v_j;

        ans = sqrt(du * du + dv * dv);
        return ans;
    }
    return -1;
#endif;

}
