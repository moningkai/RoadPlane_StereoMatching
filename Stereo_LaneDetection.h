//
// Created by nvidia on 7/30/18.
//

#ifndef ROADPLANE_STEREOMATCHING_STEREO_LANEDETECTION_H
#define ROADPLANE_STEREOMATCHING_STEREO_LANEDETECTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;

/********************************definition of coordination*********************************
 * road coordination:original point on camera optical point,forward(+Z),right(+X),down(+Y)
 * camera cooradination:original point on camera optical point,optical axis forward(+Z),right(+X),down(+Y)
 * */

struct LaneDetectPara{
    //...
};

struct MapINFO {
    float laneArea_width;//meter
    float laneNum;
    float angle;//[-PI/2,PI/2],included angle of road and camera optical axis,camera optical axis is the datum,anti-clockwise is positive
    //...
    MapINFO():laneArea_width(2.5),laneNum(2),angle(-1000){};
};

struct ScanPoint{
    cv::Point2f p;
    int flag;
    float confidence;
};

class Stereo_LaneDetection
{
public:
    virtual ~Stereo_LaneDetection();

public:
    Stereo_LaneDetection();

    //input inital lane detection parameter
    LaneDetectPara m_para;

    //current map information
    MapINFO m_mapInfo;

    void init(LaneDetectPara ldp);//>>m_para

    //current images and previous images
    cv::Mat m_L_CurImgGrey,m_R_CurImgGrey,m_L_PreImgGrey,m_R_PreImgGrey;

    //roadHMatrix(H):Homography Matrix of road plane(l_point=H*r_point)
    cv::Mat m_roadHMatrix,m_initalRoadHMatrix;

    //transMatrix(T):transform 3D points from previous frame to current current frame(cur_point3d=T*pre_point3d)
    cv::Mat m_transMatrix;

    //in_roadHMatrix(H):Homography Matrix of road plane(l_point=H*r_point)
    //in_transMatrix(T):transform 3D points from previous frame to current current frame(cur_point3d=T*pre_point3d)
    void run_lane_detection(cv::Mat in_L_img,cv::Mat in_R_img,cv::Mat in_roadHMatrix = cv::Mat(),
                            cv::Mat in_transMatrix = cv::Mat(),MapINFO in_mapInfo = MapINFO());

    void get_candidate_lines_from_stereo_images();

    vector<vector<ScanPoint>> m_pre_lane,m_cur_lane;

    void lanes_tracing();

    void single_line_tracing(vector<ScanPoint> in_lane,vector<ScanPoint> &out_cur_lane);
};


#endif //ROADPLANE_STEREOMATCHING_STEREO_LANEDETECTION_H
