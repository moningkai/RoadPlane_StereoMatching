
#ifndef STEREOMEASUREMENT_H
#define STEREOMEASUREMENT_H

#include <opencv2/opencv.hpp>
#include <vector>

/*The coordinate of this class is defined as what opencv have defined(X:right,Y:down,Z:forward)
 *
 * */

using namespace std;

enum Stereo_State
{
	STEREO_BAD,
	STEREO_INITIAL_OK,
	STEREO_RECTIFY_OK//图像已进行畸变校正
};

enum SPT_Type { SPT_SURF,SPT_SIFT,SPT_ORB };

struct Stereo_GPUbuffer
{
	cv::Mat dis_LImg,dis_RImg,
			undis_LImg,undis_RImg,
			undis_LImgGrey,
			undis_RImgGrey,
			l_mask,r_mask,
			rec_map[2][2],//remap()的映射矩阵.[0][0].[0][1]代表左相机的mapx,mapy;
			temp_map[2],
			temp_GreyImg;
};

class StereoMeasurement
{

public:
	StereoMeasurement();
	~StereoMeasurement();

    //------------------------------camera-parameter---------------------------//
    //主距
    double m_l_focalLength, m_r_focalLength;

    //distance between two cameras
    double m_Base_length;

    //左右相机内参矩阵
	cv::Mat m_l_K, m_r_K;

    //左右相机畸变参数
	cv::Mat m_l_distor, m_r_distor;

    //右相机到左相机的变换矩阵
	cv::Mat m_r2l_Tran;

    //最大视差，由相机基线和相机像素焦距决定，initial_stereo_system的结果
    //单位:像素.含义:D=X_left - X_right
	int m_max_disparity,m_min_disparity,m_max_yDisparityErr;

    //stereo images recitification parameter
    cv::Mat m_rec_Rl,//左相机到公共平面的旋转矩阵,大小:3x3
            m_rec_Rr,//右相机到公共平面的旋转矩阵,大小:3x3
            m_rec_Pl,//左相机投影矩阵--把三维空间的点投影到平面,大小:3x4
            m_rec_Pr,//右相机投影矩阵,大小:3x4
            m_rec_Q;//重投影矩阵--把图像上的点投影到三维空间,大小:视调用函数结果而定

    //stereo images recitification parameter
    cv::Mat m_rec_map[2][2];//remap()的映射矩阵.[0][0].[0][1]代表左相机的mapx,mapy;[1][0],[1][1]代表右相机的mapx,mapy

    //Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    Stereo_GPUbuffer m_gpu_buffer;

	Stereo_State m_state;

    //rectified image
    cv::Mat m_undisLeftImg,m_undisRightImg;

    //range of detection
    double m_min_range,m_max_range;

    //standard deviation of min_range and max_range
	//标准差.
    double m_min_sd,m_max_sd;

	//初始化,in_left_K:内参矩阵 in_left_distor：畸变系数 in_r2l_Tran：右影像到左影像的变换矩阵 min/max_range:min/max range of detection
	void initial_stereo_system(cv::Size image_size,cv::Mat in_left_K, cv::Mat in_right_K,
                               cv::Mat in_left_distor, cv::Mat in_right_distor, cv::Mat in_r2l_Tran, double min_range,double max_range);

	//影像畸变纠正
	void stereo_images_rectification(cv::Mat dis_leftImg, cv::Mat dis_rightImg, cv::Mat &undis_leftImg, cv::Mat &undis_rightImg);


    //------------------------------KeyPoints-descripter---------------------------//
    cv::Ptr<cv::ORB> m_gpu_orb;

    //r_Pts[l_m_dest[i].queryIdx] is the match point of l_Pts[l_m_dest[i].trainIdx]
    void run_ORB_KeyPoints_detection_and_matching(vector<cv::KeyPoint> &out_l_Pts,vector<cv::KeyPoint> &out_r_Pts,vector<cv::DMatch> &out_l_m_dest,
                                                  cv::Mat in_l_Mask,cv::Mat in_r_Mask);

	//计算图像输入点的描述子
	void get_points_descripter(int left_or_right, vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc,SPT_Type spt_type=SPT_ORB);

    void get_points_SUFTdescripter(int left_or_right, vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc);

    void get_points_ORBdescripter(int left_or_right, vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc);

    void get_points_SIFTdescripter(int left_or_right, vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc);

    //----------------------------------Matching-----------------------------------//
    cv::Ptr<cv::DescriptorMatcher> m_DMatcher_normL2,m_DMatcher_normHammin;

    //l_desc为referrenc,r_desc为target，匹配结果>>vector<int> t_m_dest对应右点集id;
	void match_descripter_of_Pts(cv::Mat t_l_desc, cv::Mat t_r_desc, vector<cv::DMatch> &t_l_m_dest,SPT_Type spt_type=SPT_ORB);

    void match_double_Points_implement(int left_or_right,vector<cv::KeyPoint> in_ref_Pts, vector<cv::KeyPoint> in_tar_Pts,cv::Mat reference_desc,cv::Mat target_desc,vector<cv::DMatch> &m_dest,
                                       SPT_Type spt_type=SPT_ORB);

	//进行左右一致性/双向验证检测>>vector<int> t_m_dest对应右点集id;
	void leftRightConsistency_and_FBCheck(vector<cv::DMatch> t_l_m_dest, vector<cv::DMatch> t_r_m_dest, vector<cv::DMatch> &t_l_consistency_dest);

	//在已有匹配结果上，获取亚像素精度的视差
	void get_subpixel_disparity(cv::Mat in_leftImg, cv::Mat in_rightImg, vector<cv::KeyPoint> in_l_Pts,
                                vector<int> t_l_consistency_dest, vector<double> &t_out_disp);

	//计算左点集在右图像上的候选区域
	void get_candidate_corresponding_area_of_leftPts(vector<cv::KeyPoint> in_l_Pts, vector< vector<cv::KeyPoint>> &t_cand_rightPts);

	//根据相机内参、视差，计算点集的空间坐标
	void cal_Points_3D_coordinate(cv::Mat in_K, vector<cv::KeyPoint> in_l_Pts, vector<double> t_out_disp, vector<cv::Point3d> &out_l_Pts);

	//对点集进行测量:输出左影像点集在左相机空间坐标系的坐标
	void run_points_measurement_process(vector<cv::KeyPoint> in_l_Pts, vector<cv::KeyPoint> in_r_Pts,
                                        vector<cv::Point3d> &out_l_Pts, SPT_Type spt_type = SPT_ORB);

	//对左点集进行图像匹配
	void matching_single_Points_process(vector<cv::KeyPoint> in_l_Pts,vector<cv::DMatch> &out_matchID,SPT_Type spt_type=SPT_ORB);

	//对左点-右点集进行图像匹配
	void matching_double_Points_process(vector<cv::KeyPoint> in_l_Pts, vector<cv::KeyPoint> in_r_Pts, vector<cv::DMatch> &out_matchID,SPT_Type spt_type=SPT_ORB);

};

#endif
