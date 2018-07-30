#include <iostream>
#include <opencv2/opencv.hpp>
#include "RoadPlane_Estimation.h"
#include <sys/timeb.h>

using namespace std;
using namespace cv;

int myget_time()
{
    time_t t_sec0;
    struct timeb tb_start;
    ftime(&tb_start);
    time(&t_sec0);
    return (int)(t_sec0*1000+tb_start.millitm);
}

int main()
{

    rm::RoadPlane_Estimation road_estimator;

    rm::Stereo_CamPara scp;

    //camera intrinsic matrix
    double K[]={718.856,0.0,607.1928,
                0.0,718.856,185.2157,
                0.0,0.0,1.0};
    //transform matrix of right image coordinate to left image coordinate
    double r2l_trans[]={1.0,0.0,0.0,1.0,
                        0.0,1.0,0.0,0.0,
                        0.0,0.0,1.0,0.0,
                        0.0,0.0,0.0,1.0};//因为目前读取的图像已经进行校正过,故该矩阵设为1,即:右图像到左图像不用进行旋转和平移

    cv::Mat left_K(3,3,CV_64FC1,K);//intrinsic
    cv::Mat right_K(3,3,CV_64FC1,K);//intrinsic
    cv::Mat distor_M=cv::Mat::zeros(1,5,CV_64FC1);//distortion matrix of camera
    cv::Mat trans_r2l_M(4,4,CV_64FC1,r2l_trans);//r2l

    scp.left_K=left_K; scp.right_K=right_K;
    scp.left_distor=distor_M; scp.right_distor=distor_M;
    scp.r2l_Tran=trans_r2l_M;
    scp.init_cam_height=1.50;
    scp.init_cam_pitch=0.0;
    scp.image_size = cv::Size(1242,375);
    scp.min_range=4;
    scp.max_range=50;

    road_estimator.init(scp);

    cv::VideoCapture left_reader,right_reader;
    if(!left_reader.open("/home/nvidia/Pictures/03/left/%06d.png")){cerr<<"error in opening video_1!"<<endl;return -1;}
    if(!right_reader.open("/home/nvidia/Pictures/03/right/%06d.png")){cerr<<"error in opening video_2!"<<endl;return -1;}

    while(1)
    {
        Mat img1,img2;
        if(!left_reader.read(img1)){cerr<<"error in reading video_1!"<<endl;break;}
        if(!right_reader.read(img2)){cerr<<"error in reading video_2!"<<endl;break;}

        int tstart=myget_time();

        road_estimator.run_road_plane_estimation(img1,img2);//路面估计

        cout<<"total:"<<myget_time()- tstart << " ms" << std::endl;

        int c=cv::waitKey(1);
        if(c==27){ break;}
    }//while

    return 0;
}