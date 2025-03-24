#include<vector>
#include<Eigen/Dense>
#include<fstream>
#include<string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h> // icp算法
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <boost/thread/thread.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/search/flann_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include<iomanip>
#include<omp.h>
#include <pcl/filters/bilateral.h>  
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>

using namespace std;

struct rt
{
    double time;
    cv::Matx33f R;
    cv::Vec3f T;
    double angle;
    Eigen::Quaterniond Q;
};
struct calibrt
{
    double angle;
    Eigen::Vector3d motoraxis;
    Eigen::Vector3d trans;
};

cv::Matx31f to_Rvec(cv::Matx33f Rmatrix)
{
    cv::Matx31f dst;
    cv::Rodrigues(Rmatrix, dst, cv::noArray());
    return dst;
}

cv::Matx33f to_Rmatrix(cv::Matx31f Rvec)
{
    cv::Matx33f dst;
    cv::Rodrigues(Rvec, dst, cv::noArray());
    return dst;
}
double GetRad(double angle)
{
    return angle * 3.14159265358 / 180;
}

void rotcloud(pcl::PointCloud<pcl::PointXYZI>& cloud, rt RT)
{
    for (int i = 0; i < cloud.points.size(); i++)
    {
        cv::Matx31f p;
        p(0) = cloud.points[i].x; p(1) = cloud.points[i].y; p(2) = cloud.points[i].z;
        p = RT.R * p + RT.T;
        cloud.points[i].x = p(0); cloud.points[i].y = p(1); cloud.points[i].z = p(2);
    }
}
void filtnear(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered,double distance)
{
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);

    // 过滤X轴
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-distance, distance);
    pass.setNegative(true);
    pass.filter(*cloud_filtered);

    // 过滤Y轴
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-distance, distance);
    pass.setNegative(true);
    pass.filter(*cloud_filtered);

    // 过滤Z轴
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-distance, distance);
    pass.setNegative(true);
    pass.filter(*cloud_filtered);

}
void getLid2IMUTransRefine(const double& motorAngle, Eigen::Matrix3d& Ril, Eigen::Vector3d& til, Eigen::Vector3d motoraxis, Eigen::Vector3d trans)
{
    Eigen::Matrix3d rotMatrix;
    Eigen::Vector3d vectorBefore(0, 1, 0);
    //Eigen::Vector3d vectorAfter(0.000718973, 0.99999, 0.00379432);
    rotMatrix = Eigen::Quaterniond::FromTwoVectors(vectorBefore, motoraxis).toRotationMatrix();
    Eigen::Affine3d _fixPart_new = Eigen::Affine3d::Identity();

    _fixPart_new.rotate(rotMatrix.inverse());
    //_fixPart_new.translation() << 0.00213961, 0, 0.0176996;
    _fixPart_new.translation() << trans(0, 0), 0, trans(2, 0);

    Eigen::Affine3d Til = _fixPart_new.inverse() *
        Eigen::AngleAxisd(-motorAngle, Eigen::Vector3d::UnitY()) *
        _fixPart_new;

    Ril = Til.linear();
    til = Til.translation();
}

void cloud_with_normal(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_normals)
{
    //-----------------拼接点云数据与法线信息---------------------
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> n;//OMP加速
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    //建立kdtree来进行近邻点集搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    //n.setViewPoint(0,0,0);//设置视点，默认为（0，0，0）
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(20);//点云法向计算时，需要所搜的近邻点大小
    //n.setRadiusSearch(0.03);//半径搜素
    n.compute(*normals);//开始进行法向计
    //将点云数据与法向信息拼接
    pcl::concatenateFields(*cloud, *normals, *cloud_normals);
}


void p2picp(pcl::PointCloud<pcl::PointXYZI>::Ptr source, pcl::PointCloud<pcl::PointXYZI>::Ptr target,/*pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_with_normals,*/ pcl::PointCloud<pcl::PointXYZI>::Ptr & final, Eigen::Matrix4f& T, double distance)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
    cloud_with_normal(source, source_with_normals);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
    cloud_with_normal(target, target_with_normals);
    //--------------------点到面的icp（非线性优化）-----------------------
    pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal>lm_icp;
    pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZINormal, pcl::PointXYZINormal>::Ptr PointToPlane
    (new pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZINormal, pcl::PointXYZINormal>);
    lm_icp.setTransformationEstimation(PointToPlane);
    lm_icp.setInputSource(source_with_normals);
    lm_icp.setInputTarget(target_with_normals);
    lm_icp.setTransformationEpsilon(1e-20);   // 为终止条件设置最小转换差异
    lm_icp.setMaxCorrespondenceDistance(distance);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
    lm_icp.setEuclideanFitnessEpsilon(0.000001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
    lm_icp.setMaximumIterations(20);           // 最大迭代次数
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr lm_icp_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    lm_icp.align(*lm_icp_cloud);
    //cout << "\nICP has converged, score is " << lm_icp.getFitnessScore() << endl;
    //cout << "变换矩阵：\n" << lm_icp.getFinalTransformation() << endl;
    // 使用创建的变换对为输入的源点云进行变换
    pcl::transformPointCloud(*source, *final, lm_icp.getFinalTransformation());
    T = lm_icp.getFinalTransformation();
}
void getLid2IMUTrans(const double& motorAngle, Eigen::Matrix3d& Ril, Eigen::Vector3d& til)
{
    Eigen::Affine3d Til =
        Eigen::Translation3d(0, 0, -0.0184) *
        Eigen::AngleAxisd(-motorAngle, Eigen::Vector3d::UnitY()) *
        Eigen::Translation3d(0, 0, 0.0184);

    Ril = Til.linear();
    til = Til.translation();
}
void calibmotor(std::string filepath, double z_shift, double distance,int iternum)
{
    Eigen::Vector3d motoraxis;
    Eigen::Vector3d trans;
    motoraxis(0) = 0; motoraxis(1) = 1; motoraxis(2) = 0;
    trans(0) = 0; trans(1) = 0; trans(2) = 0.0184;
    trans(2) = z_shift;
    ofstream fout("result/RefineParam.txt", ios::out);
    string path = filepath + "result";
    mkdir("resultpcd",0755);
    pcl::PointCloud<pcl::PointXYZI>::Ptr mergecloudzg(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < 36; i++)
    {
        string file = filepath + to_string(i + 1) + ".pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::io::loadPCDFile<pcl::PointXYZI>(file, *cloud);
        Eigen::Matrix3d R; Eigen::Vector3d T; Eigen::Matrix4f RT; Eigen::Matrix3d R_c; Eigen::Vector3d T_c;
        getLid2IMUTrans(GetRad(i * 10), R, T);
        rt temp; temp.R(0, 0) = R(0, 0); temp.R(0, 1) = R(0, 1); temp.R(0, 2) = R(0, 2);
        temp.R(1, 0) = R(1, 0); temp.R(1, 1) = R(1, 1); temp.R(1, 2) = R(1, 2);
        temp.R(2, 0) = R(2, 0); temp.R(2, 1) = R(2, 1); temp.R(2, 2) = R(2, 2);
        temp.T(0) = T(0); temp.T(1) = T(1); temp.T(2) = T(2);
        rotcloud(*cloud, temp);
        *mergecloudzg = *cloud + *mergecloudzg;
    }
    pcl::io::savePCDFile<pcl::PointXYZI>(filepath + "result/orign.pcd", *mergecloudzg);
    for (int j = 0; j < iternum; j++)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr mergecloud(new pcl::PointCloud<pcl::PointXYZI>);
        vector<calibrt>resultRT;
        for (int i = 0; i < 36; i++)
        {
            string file = filepath + to_string(i + 1) + ".pcd";
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::io::loadPCDFile<pcl::PointXYZI>(file, *cloud);
            //filtnear(cloudbefore, cloud, 0.1);
            //pcl::io::savePCDFile<pcl::PointXYZI>(filepath + "result//test.pcd", *cloud);
            Eigen::Matrix3d R; Eigen::Vector3d T; Eigen::Matrix4f RT; Eigen::Matrix3d R_c; Eigen::Vector3d T_c;
            getLid2IMUTransRefine(GetRad(i * 10), R, T, motoraxis, trans);
            rt temp; temp.R(0, 0) = R(0, 0); temp.R(0, 1) = R(0, 1); temp.R(0, 2) = R(0, 2);
            temp.R(1, 0) = R(1, 0); temp.R(1, 1) = R(1, 1); temp.R(1, 2) = R(1, 2);
            temp.R(2, 0) = R(2, 0); temp.R(2, 1) = R(2, 1); temp.R(2, 2) = R(2, 2);
            temp.T(0) = T(0); temp.T(1) = T(1); temp.T(2) = T(2);
            rotcloud(*cloud, temp);
            //cout << i << endl;
            if (mergecloud->points.size() != 0) {
                cout << "iter: " <<j<< " 正在处理: 第"<<i<<"帧点云" << endl;
                p2picp(cloud, mergecloud, cloud, RT, distance);
                //gicp(cloud, mergecloud, cloud, RT);
                R_c(0, 0) = RT(0, 0); R_c(0, 1) = RT(0, 1); R_c(0, 2) = RT(0, 2); T_c(0, 0) = RT(0, 3);
                R_c(1, 0) = RT(1, 0); R_c(1, 1) = RT(1, 1); R_c(1, 2) = RT(1, 2); T_c(1, 0) = RT(1, 3);
                R_c(2, 0) = RT(2, 0); R_c(2, 1) = RT(2, 1); R_c(2, 2) = RT(2, 2); T_c(2, 0) = RT(2, 3);
                R = R_c * R;
                T = R_c * T + T_c;

                cv::Matx33f Rmatrix;
                Rmatrix(0, 0) = R(0, 0); Rmatrix(0, 1) = R(0, 1); Rmatrix(0, 2) = R(0, 2);
                Rmatrix(1, 0) = R(1, 0); Rmatrix(1, 1) = R(1, 1); Rmatrix(1, 2) = R(1, 2);
                Rmatrix(2, 0) = R(2, 0); Rmatrix(2, 1) = R(2, 1); Rmatrix(2, 2) = R(2, 2);
                cv::Matx31f rvec = to_Rvec(Rmatrix);
                //cout << (180*sqrt(rvec(0, 0) * rvec(0, 0) + rvec(1, 0) * rvec(1, 0) + rvec(2, 0) * rvec(2, 0)))/3.141592653 << endl;
                float norm = sqrt(rvec(0, 0) * rvec(0, 0) + rvec(1, 0) * rvec(1, 0) + rvec(2, 0) * rvec(2, 0));
                rvec(0, 0) /= norm;
                rvec(1, 0) /= norm;
                rvec(2, 0) /= norm;
                calibrt temprt;
                temprt.angle = i * 10;
                if (rvec(1, 0) < 0)
                {
                    temprt.motoraxis(0, 0) = -rvec(0, 0); temprt.motoraxis(1, 0) = -rvec(1, 0); temprt.motoraxis(2, 0) = -rvec(2, 0);
                }
                else
                {
                    temprt.motoraxis(0, 0) = rvec(0, 0); temprt.motoraxis(1, 0) = rvec(1, 0); temprt.motoraxis(2, 0) = rvec(2, 0);
                }
                Eigen::Matrix3d rotMatrix;
                Eigen::Vector3d vectorBefore(0, 1, 0);
                rotMatrix = Eigen::Quaterniond::FromTwoVectors(vectorBefore, temprt.motoraxis).toRotationMatrix();
                Eigen::Matrix3d r1invr2;
                rotMatrix = rotMatrix.inverse();
                r1invr2 = R * rotMatrix.inverse();
                temprt.trans = (r1invr2 - rotMatrix.inverse()).inverse() * T;
                resultRT.push_back(temprt);
            }

            *mergecloud = *cloud + *mergecloud;
        }
        
        pcl::io::savePCDFile<pcl::PointXYZI>(filepath + "result//refine-" + to_string(j) + ".pcd", *mergecloud);
        motoraxis.setZero(); trans.setZero();
        for (int i = 0; i < resultRT.size(); i++)
        {
            motoraxis += resultRT[i].motoraxis;
            trans += resultRT[i].trans;
        }
        motoraxis /= resultRT.size();
        trans /= resultRT.size();
        cout << "iter: " << j << endl;
        cout << "最优旋转轴：" << motoraxis << endl;
        cout << "最优平移：" << trans << endl;
        fout << "iter: " << j << "  " << motoraxis(0) << " " << motoraxis(1) << " " << motoraxis(2) << " " << trans(0) << " " << trans(1) << " " << trans(2) << endl;
    }
}


int main()
{
    ifstream fin;
    string calib_config_file("param/Function_parameters.yaml");
    cv::FileStorage fSettings(calib_config_file, cv::FileStorage::READ);
    string filepath; double z_shift; double distance; int iternum;
    filepath = fSettings["pcdpath"].string();       //canny param 10
    z_shift = fSettings["z_shift"];            //edge in image minlength 200
    distance = fSettings["max_distance"];
    iternum = fSettings["iter_num"];

    calibmotor(filepath, z_shift, distance,iternum);
}