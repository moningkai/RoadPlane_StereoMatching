//
// Created by nvidia on 7/17/18.
//

#ifndef ROADPLANE_ESTIMATION_H
#define ROADPLANE_ESTIMATION_H


#include "StereoMeasurement.h"
#include <opencv2/opencv.hpp>
#include <sys/timeb.h>

namespace rm
{
    using cv::Mat;
    using std::cout;using std::cin;using std::vector;

    struct Stereo_CamPara
    {
        cv::Size image_size;
        cv::Mat left_K, right_K;//左右相机内参矩阵
        cv::Mat left_distor, right_distor;//左右相机畸变参数
        cv::Mat r2l_Tran;//右相机到左相机的变换矩阵
        double init_cam_height,init_cam_pitch;//相机初始高度，俯仰角
        double min_range, max_range;//min/max range of detection
        Stereo_CamPara():init_cam_height(0.0),init_cam_pitch(0.0),min_range(0.0),max_range(30.0){};
    };

    class RoadPlane_Estimation {

    public:
        RoadPlane_Estimation();

        virtual ~RoadPlane_Estimation();

        //stereo matching implementation
        StereoMeasurement m_stereoMeasurer;

        //initial camera height and pitch
        //相机高度以及相机俯仰角
        double m_camHeight,m_camPitch;

        //undistortion image after preprocessing
        Mat m_undistorLimg,m_undistorRimg;

        //segment resuilt has a same size with undistorL/Rimg
        Mat m_l_segImg,m_r_segImg;

        //road roi mask:only 255 represent road
        Mat m_l_roadROI,m_r_roadROI;

        //key points in road roi
        vector<cv::KeyPoint> m_l_roadKPts,m_r_roadKPts;

        //left and right match key point ID
        vector<cv::DMatch> m_roadKPts_matchID;

        //homography matrix of object plane
        Mat m_road_HMatrix;

        //Object boundry
        vector<cv::Point2i> m_roadBoundry;

    public:
        //initializa stereo camera system and road model
        void init(Stereo_CamPara& in_stereo_camPara);

        void run_road_plane_estimation(Mat in_leftImg,Mat in_rightImg);

    private:
        //去除图像畸变
        void preprocess_stereo_images(Mat in_leftImg,Mat in_rightImg,Mat &out_leftImg,Mat &out_rightImg);

        //out_segImg contains several kinds of color representing object classes
        void segment_images(Mat in_Img,Mat &out_segImg);

        //--------for-test--------//
        int m_cur_frameID = 0;//for test
        void segment_i_images(Mat in_Img,int left_or_right,Mat &out_segImg);
        //--------for-test--------//

        void get_objectROI_from_segImg(Mat in_segImg,cv::Scalar obj_classColor,Mat &out_objectROI);

        void calculate_homography_matrix(vector<cv::KeyPoint> in_l_KPts,vector<cv::KeyPoint> in_r_KPts,
                                         vector<cv::DMatch> in_KPts_matchID,Mat &out_ObjectHMatrix);

        void get_object_boundry(Mat in_ObjectHMatrix,vector<cv::Point2i> &out_ObjectBoundry);
    };
}

#endif //ROADPLANE_ESTIMATION_H
