//
// Created by nvidia on 7/30/18.
//

#include "Stereo_LaneDetection.h"

cv::Mat joint_stereoImage(cv::Mat l_img,cv::Mat r_img)
{
    cv::Mat js(l_img.rows+r_img.rows,l_img.cols,l_img.type());
    l_img.convertTo(js.rowRange(0,l_img.rows),js.type());
    r_img.convertTo(js.rowRange(l_img.rows,js.rows),js.type());
    return js;
}

Stereo_LaneDetection::Stereo_LaneDetection() {}

Stereo_LaneDetection::~Stereo_LaneDetection() {

}

void Stereo_LaneDetection::init(LaneDetectPara ldp) {

    //m_para=...

    //initialize lane tracing parameter>>m_scanlines

    //calculate roadHMatrix based on initial parameter
    //m_roadHMatrix=...

}

void Stereo_LaneDetection::run_lane_detection(cv::Mat in_L_img,cv::Mat in_R_img,cv::Mat in_roadHMatrix,
                                              cv::Mat in_transMatrix,MapINFO in_mapInfo)
{

    //check consistency of left image and right image
    if(in_L_img.size()!=in_R_img.size()) return;

    if(in_L_img.type()!=CV_8UC1)in_L_img.convertTo(m_L_CurImgGrey,CV_8UC1);
    else m_L_CurImgGrey=in_L_img.clone();

    if(in_R_img.type()!=CV_8UC1)in_R_img.convertTo(m_R_CurImgGrey,CV_8UC1);
    else m_R_CurImgGrey=in_R_img.clone();

    if(in_roadHMatrix.size()!=cv::Size(3,3)) in_roadHMatrix.convertTo(m_roadHMatrix,CV_32FC1);
    else m_roadHMatrix=m_initalRoadHMatrix.clone();

    if(in_transMatrix.size()!=cv::Size(4,4)) in_transMatrix.convertTo(m_transMatrix,CV_32FC1);
    else m_transMatrix=cv::Mat::eye(4,4,CV_32FC1);

    if(in_mapInfo.angle >= -CV_PI/2 && in_mapInfo.angle <= CV_PI/2) m_mapInfo=in_mapInfo;
    else m_mapInfo.angle=-1000;

    //get candidate lines from stereo images >> candidate points on scan lines
    get_candidate_lines_from_stereo_images();

    //lane tracing:m_pre_lane>>m_cur_lane
    lanes_tracing();

    //lanes analysis and fitting

    //update tracing targets

    //save current information as previous information of next frame
    m_L_PreImgGrey=m_L_CurImgGrey.clone();
    m_R_PreImgGrey=m_R_CurImgGrey.clone();
    m_pre_lane.swap(m_cur_lane);
    //...
}

void Stereo_LaneDetection::get_candidate_lines_from_stereo_images() {
    //create canny edge map
    cv::Mat t_L_blur,t_R_blur;
    cv::Mat t_L_canny,t_R_canny;
    cv::GaussianBlur(m_L_CurImgGrey,t_L_blur,cv::Size(5,5),2.0,2.0);
    cv::GaussianBlur(m_R_CurImgGrey,t_R_blur,cv::Size(5,5),2.0,2.0);
    cv::Canny(t_L_blur,t_L_canny,30,80);
    cv::Canny(t_R_blur,t_R_canny,30,80);

    //get contours from canny map
    vector<vector<cv::Point>> t_L_contours,t_R_contours;
    cv::findContours(t_L_canny,t_L_contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
    cv::findContours(t_R_canny,t_R_contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

#ifdef DEBUG_SHOW_CONTOURS

    cv::RNG rng;
    for (int i = 0; i <t_L_contours.size(); ++i) {
        if(t_L_contours[i].size()<30)continue;
        cv::drawContours(m_L_CurImgGrey,t_L_contours,i,cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));
    }
    for (int i = 0; i <t_R_contours.size(); ++i) {
        if(t_R_contours[i].size()<30)continue;
        cv::drawContours(m_R_CurImgGrey,t_R_contours,i,cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)));
    }

    cv::imshow("CONTOURS",joint_stereoImage(m_L_CurImgGrey,m_R_CurImgGrey));
    cv::waitKey(1);

#endif

    //match contours using roadHMatrix

    //judge up or down peak in canny map
}

void Stereo_LaneDetection::lanes_tracing() {

}

void Stereo_LaneDetection::single_line_tracing(vector<ScanPoint> in_pre_lane,vector<ScanPoint> &out_cur_lane) {

}