#include "StereoMeasurement.h"


StereoMeasurement::StereoMeasurement()
{
	m_state = STEREO_BAD;
}


StereoMeasurement::~StereoMeasurement()
{

}

void StereoMeasurement::initial_stereo_system(cv::Size image_size,cv::Mat in_left_K, cv::Mat in_right_K,
											  cv::Mat in_left_distor, cv::Mat in_right_distor, cv::Mat in_r2l_Tran,
                                              double min_range,double max_range)
{
	if (in_left_K.rows!=3 || in_left_K.cols != 3 || in_right_K.rows != 3 || in_right_K.cols != 3
		|| in_r2l_Tran.rows != 4 || in_r2l_Tran.cols != 4
		|| in_left_distor.cols!=5 || in_left_distor.rows!=1 || in_right_distor.cols!=5 || in_right_distor.rows!=1)
	{
		m_state = STEREO_BAD;
		cerr<<"paramerter data error!"<<endl;
		return;
	}

    m_l_K = in_left_K.clone(); m_r_K = in_right_K.clone();
	m_l_distor = in_left_distor.clone(); m_r_distor = in_right_distor.clone();
	m_r2l_Tran = in_r2l_Tran.clone();

    m_l_focalLength = (m_l_K.at<double>(0, 0) + m_l_K.at<double>(1, 1)) / 2;
	m_r_focalLength = (m_r_K.at<double>(0, 0) + m_r_K.at<double>(1, 1)) / 2;
    m_min_range=min_range>0.0f?min_range:0.0f;
    m_max_range=max_range>0.0f?max_range:0.0f;
    m_Base_length=sqrt(m_r2l_Tran.at<double>(0,3)*m_r2l_Tran.at<double>(0,3)+
                       m_r2l_Tran.at<double>(1,3)*m_r2l_Tran.at<double>(1,3)+
                       m_r2l_Tran.at<double>(2,3)*m_r2l_Tran.at<double>(2,3));

    //calculate standard deviation
    m_min_sd=(m_min_range*m_min_range)/(m_l_focalLength*m_Base_length);
    m_max_sd=(m_max_range*m_max_range)/(m_l_focalLength*m_Base_length);

    m_min_disparity=(m_l_focalLength*m_Base_length)/m_max_range;
    m_max_disparity=(m_l_focalLength*m_Base_length)/m_min_range;

    m_max_disparity=m_max_disparity>255?255:m_max_disparity;
    m_min_disparity=m_min_disparity<0?0:m_min_disparity;
    m_max_yDisparityErr=1;

	//rectify
	cv::Mat t_R=m_r2l_Tran.rowRange(0,3).colRange(0,3).inv();//left image to right image
	cv::Mat t_T=-t_R*m_r2l_Tran.rowRange(0,3).col(3);//left image to right image
	cv::stereoRectify(m_l_K,m_l_distor,m_r_K,m_r_distor,image_size
					  ,t_R,t_T,m_rec_Rl,m_rec_Rr,m_rec_Pl,m_rec_Pr,m_rec_Q,cv::CALIB_ZERO_DISPARITY,0);//设置CV_CALIB_ZERO_DISPARITY,函数的作用是使每个相机的主点在校正后的图像上有相同的像素坐标。

	//计算映射
    //计算图像的映射表 mapx,mapy
    //mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	cv::initUndistortRectifyMap(m_l_K,m_l_distor, m_rec_Rl, m_rec_Pl, image_size, CV_32FC1, m_rec_map[0][0], m_rec_map[0][1]);
	cv::initUndistortRectifyMap(m_r_K,m_r_distor, m_rec_Rr, m_rec_Pr, image_size, CV_32FC1, m_rec_map[1][0], m_rec_map[1][1]);

	m_gpu_buffer.rec_map[0][0]=m_rec_map[0][0].clone();
	m_gpu_buffer.rec_map[0][1]=m_rec_map[0][1].clone();
	m_gpu_buffer.rec_map[1][0]=m_rec_map[1][0].clone();
	m_gpu_buffer.rec_map[1][1]=m_rec_map[1][1].clone();
    m_gpu_buffer.temp_map[0].create(image_size,CV_32FC1);
    m_gpu_buffer.temp_map[1].create(image_size,CV_32FC1);

    //initialize gpu images
    m_gpu_buffer.dis_LImg.create(image_size,CV_8UC3);
    m_gpu_buffer.dis_RImg.create(image_size,CV_8UC3);
    m_gpu_buffer.undis_LImg.create(image_size,CV_8UC3);
    m_gpu_buffer.undis_RImg.create(image_size,CV_8UC3);
    m_gpu_buffer.undis_LImgGrey.create(image_size,CV_8UC1);
    m_gpu_buffer.undis_RImgGrey.create(image_size,CV_8UC1);
    m_gpu_buffer.l_mask.create(image_size,CV_8UC1);
    m_gpu_buffer.r_mask.create(image_size,CV_8UC1);
    m_gpu_buffer.temp_GreyImg.create(image_size,CV_8UC1);

    m_DMatcher_normHammin=cv::DescriptorMatcher::create(cv::NORM_HAMMING);
    m_DMatcher_normL2=cv::DescriptorMatcher::create(cv::NORM_L2);

    //initialize points detector
    //m_gpu_orb=cv::ORB::create(250,0.8f,4,50,0,2,cv::ORB::FAST_SCORE,7,10);
    m_gpu_orb=cv::ORB::create();
    m_state = STEREO_INITIAL_OK;

}

//影像畸变纠正
void StereoMeasurement::stereo_images_rectification(cv::Mat dis_leftImg, cv::Mat dis_rightImg, cv::Mat & undis_leftImg, cv::Mat & undis_rightImg)
{
	if (m_state == STEREO_BAD)
	{
        cerr<<"initial_error!"<<endl;
        return;
    }//if rectification isn't work

	m_gpu_buffer.dis_LImg=dis_leftImg.clone();
	m_gpu_buffer.dis_RImg=dis_rightImg.clone();

    //rectify image by remap
	cv::remap(m_gpu_buffer.dis_LImg, m_gpu_buffer.undis_LImg, m_gpu_buffer.rec_map[0][0], m_gpu_buffer.rec_map[0][1], cv::INTER_LINEAR);//左校正
	cv::remap(m_gpu_buffer.dis_RImg, m_gpu_buffer.undis_RImg, m_gpu_buffer.rec_map[1][0], m_gpu_buffer.rec_map[1][1], cv::INTER_LINEAR);//右校正


    undis_leftImg = m_gpu_buffer.undis_LImg.clone();
    undis_rightImg = m_gpu_buffer.undis_RImg.clone();

    m_undisLeftImg=undis_leftImg.clone();
    m_undisRightImg=undis_rightImg.clone();

    m_state = STEREO_RECTIFY_OK;

    //将校正后的图像显示出来,查看结果
#ifdef DEBUG_SHOW_RECTIFYIMAGE

	cv::Mat showImage(dis_leftImg.size().height,2*dis_leftImg.size().width,CV_8UC3);

    //将校正后的图像合并在一张图上
	cv::Rect rectLeft(0,0,dis_leftImg.size().width,dis_leftImg.size().height);
	cv::Rect rectRight(dis_leftImg.size().width,0,dis_leftImg.size().width,dis_leftImg.size().height);

	undis_leftImg.copyTo(showImage(rectLeft));
	undis_rightImg.copyTo(showImage(rectRight));

	//画上对应的线条
	for (int i = 0; i < 10; ++i)
	{
		cv::line(showImage,cv::Point(0,showImage.rows*i/10),cv::Point(showImage.cols,showImage.rows*i/10),cv::Scalar(0,255,0));
	}
    cv::resize(showImage, showImage, cv::Size(dis_leftImg.cols,dis_leftImg.rows/2), 0, 0, cv::INTER_LINEAR);//将图像大小调整为左相机图片一样大小,方便查看
	cv::imshow("image after remap()",showImage);
	cv::moveWindow("image after remap()",0,0);
	cv::waitKey(1);
#endif

}

void StereoMeasurement::get_points_descripter(int left_or_right, vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc,SPT_Type spt_type)
{
    switch(spt_type)
    {
        case SPT_SURF:
            get_points_SUFTdescripter(left_or_right,in_Pts,t_out_desc);
            break;
        case SPT_ORB:
            get_points_ORBdescripter(left_or_right,in_Pts,t_out_desc);
            break;
        case SPT_SIFT:
            get_points_SIFTdescripter(left_or_right,in_Pts,t_out_desc);
            break;
    }
}

void StereoMeasurement::get_points_SUFTdescripter(int left_or_right, std::vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc){
}

void StereoMeasurement::get_points_ORBdescripter(int left_or_right, std::vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc){

    cv::/*cuda::*/Mat t_desc;

    if(left_or_right==-1 && in_Pts.size()>0)m_gpu_orb->compute(m_gpu_buffer.undis_LImgGrey,in_Pts,t_desc);

    if(left_or_right== 1 && in_Pts.size()>0)m_gpu_orb->compute(m_gpu_buffer.undis_RImgGrey,in_Pts,t_desc);

    t_out_desc = t_desc.clone();
}

void StereoMeasurement::get_points_SIFTdescripter(int left_or_right, std::vector<cv::KeyPoint> in_Pts, cv::Mat &t_out_desc){

}

void StereoMeasurement::match_descripter_of_Pts(cv::Mat t_l_desc, cv::Mat t_r_desc, std::vector<cv::DMatch> &t_l_m_dest,SPT_Type spt_type)
{
    switch (spt_type)
    {
        case SPT_ORB:
            m_DMatcher_normHammin->match(t_r_desc,t_l_desc,t_l_m_dest);
            break;
        case SPT_SURF:
            break;
        case SPT_SIFT:
            break;
    }
}

void StereoMeasurement::leftRightConsistency_and_FBCheck(vector<cv::DMatch> t_l_m_dest, vector<cv::DMatch> t_r_m_dest, vector<cv::DMatch> &out_matchID)
{
    //---------------Identify left-right consistency---------------//

    //find max id of right key points
    int tr_maxSize=0;

    for (int i = 0; i < t_r_m_dest.size(); ++i) if(t_r_m_dest[i].trainIdx>tr_maxSize) tr_maxSize=t_r_m_dest[i].trainIdx;

    tr_maxSize++;

    //create a look up table of right points
    vector<cv::DMatch> r_cand_match;

    r_cand_match.resize(tr_maxSize);

    for (int i = 0; i < tr_maxSize; ++i) r_cand_match[i].distance=-10000;

    for (int i = 0; i < t_r_m_dest.size(); ++i) r_cand_match[t_r_m_dest[i].trainIdx]=t_r_m_dest[i];

    //find point which fits left-right consistency
    vector<cv::DMatch> ().swap(out_matchID);

    out_matchID.reserve(t_l_m_dest.size());

    for (int i = 0; i < t_l_m_dest.size(); ++i)
    {
        if(r_cand_match[t_l_m_dest[i].queryIdx].distance >= 0 && r_cand_match[t_l_m_dest[i].queryIdx].queryIdx==t_l_m_dest[i].trainIdx) out_matchID.push_back(t_l_m_dest[i]);
    }//for(i)

}

void StereoMeasurement::get_subpixel_disparity(cv::Mat in_leftImg, cv::Mat in_rightImg, vector<cv::KeyPoint> in_l_Pts,
                                               vector<int> t_l_consistency_dest, vector<double> &t_out_disp)
{
}

void StereoMeasurement::get_candidate_corresponding_area_of_leftPts(vector<cv::KeyPoint> in_l_Pts, vector<vector<cv::KeyPoint>>& t_cand_rightPts)
{
}

void StereoMeasurement::cal_Points_3D_coordinate(cv::Mat in_K, vector<cv::KeyPoint> in_l_Pts, vector<double> t_out_disp, vector<cv::Point3d>& out_l_Pts)
{
}

void StereoMeasurement::run_points_measurement_process(vector<cv::KeyPoint> in_l_Pts, vector<cv::KeyPoint> in_r_Pts,
                                                       vector<cv::Point3d> &out_l_Pts, SPT_Type spt_type)
{
	vector<double> t_out_disp(in_l_Pts.size(),0.0f);
    vector<cv::DMatch> t_out_matchID;

    //two matching mode
	if (in_r_Pts.size() == 0)
		matching_single_Points_process(in_l_Pts, t_out_matchID,spt_type);
	else
		matching_double_Points_process(in_l_Pts, in_r_Pts, t_out_matchID,spt_type);

    //cal disparity
    for (int i = 0; i < t_out_matchID.size(); ++i) {
        auto tm=t_out_matchID[i];
        t_out_disp[tm.trainIdx]=((double)(in_r_Pts[tm.queryIdx].pt.x-in_l_Pts[tm.trainIdx].pt.x));
    }//for(i)

	cal_Points_3D_coordinate(m_l_K, in_l_Pts, t_out_disp, out_l_Pts);
}

void StereoMeasurement::matching_single_Points_process(vector<cv::KeyPoint> in_l_Pts,
                                                       vector<cv::DMatch> &out_matchID,SPT_Type spt_type)
{
	vector<vector<cv::KeyPoint>> t_cand_rightPts;
	get_candidate_corresponding_area_of_leftPts(in_l_Pts, t_cand_rightPts);//>>vector<vector<Point2d>>cand_rightPts
	for (int i = 0; i < in_l_Pts.size(); i++) {
		vector<cv::KeyPoint> t_in_l_P(1);
		t_in_l_P[0] = in_l_Pts[i];
		matching_double_Points_process(t_in_l_P, t_cand_rightPts[i], out_matchID,spt_type);
        //.............
	}//for(i)
}

void StereoMeasurement::matching_double_Points_process(vector<cv::KeyPoint> in_l_Pts, vector<cv::KeyPoint> in_r_Pts,
                                                       vector<cv::DMatch> &out_matchID,SPT_Type spt_type)
{
	cv::Mat t_out_l_desc, t_out_r_desc;
	get_points_descripter(-1, in_l_Pts, t_out_l_desc,spt_type);//根据特征点计算描述子
	get_points_descripter( 1, in_r_Pts, t_out_r_desc,spt_type);

    std::vector<cv::DMatch> t_l_m_dest, t_r_m_dest;

    //left reference:find best-fit target from candidate points
    match_double_Points_implement(-1,in_l_Pts,in_r_Pts,t_out_l_desc,t_out_r_desc,t_l_m_dest,spt_type);

    //right reference:find best-fit target from candidate points
    match_double_Points_implement( 1,in_r_Pts,in_l_Pts,t_out_r_desc,t_out_l_desc,t_r_m_dest,spt_type);

	leftRightConsistency_and_FBCheck(t_l_m_dest, t_r_m_dest, out_matchID);//取交集

    //-----------------------debug----------------------//
#ifdef DEBUG_SHOW_KERPOINTMATCH
    cv::Mat show_matching_l;
    cv::drawMatches(m_undisRightImg,in_r_Pts,m_undisLeftImg,in_l_Pts,out_matchID,show_matching_l,cv::Scalar(0,255,0),cv::Scalar(0,0,255));
    cv::imshow("show_matching_l",show_matching_l);
    cv::waitKey(1);
#endif

}

//train query
void StereoMeasurement::match_double_Points_implement(int left_or_right, vector<cv::KeyPoint> in_ref_Pts, vector<cv::KeyPoint> in_tar_Pts,
                                                      cv::Mat ref_desc,cv::Mat tar_desc,vector<cv::DMatch> &m_dest,SPT_Type spt_type){
    for (int i = 0; i <in_ref_Pts.size(); i++)
    {

        std::vector<cv::DMatch> temp_m_dest;
        cv::Mat temp_l_desc;
        temp_l_desc=ref_desc.row(i).clone();

        std::vector<int> temp_r_id;
        temp_r_id.reserve(10);

        int trefNUM=in_ref_Pts.size();
        int ttarNUM=in_tar_Pts.size();
        if(trefNUM!=ref_desc.rows || ttarNUM!=tar_desc.rows)
        {
            return;
        }

        //pick candidate Pts
        for (int j = 0; j <in_tar_Pts.size(); j++)
        {

            if(left_or_right== 1)
            {
                if((in_tar_Pts[j].pt.x-in_ref_Pts[i].pt.x)<=m_max_disparity &&
                   (in_tar_Pts[j].pt.x-in_ref_Pts[i].pt.x)>=0 &&
                   abs(in_tar_Pts[j].pt.y-in_ref_Pts[i].pt.y)<=m_max_yDisparityErr)
                    temp_r_id.push_back(j);
            }
            else
            {
                if((in_ref_Pts[i].pt.x-in_tar_Pts[j].pt.x)<=m_max_disparity &&
                   (in_ref_Pts[i].pt.x-in_tar_Pts[j].pt.x)>=0 &&
                   abs(in_ref_Pts[i].pt.y-in_tar_Pts[j].pt.y)<=m_max_yDisparityErr)
                    temp_r_id.push_back(j);
            }

        }//for(j)

        if(temp_r_id.size()<1)
        {
            //no candidate points
            //m_dest.push_back(cv::DMatch(i,-1,10000.0f));
        }
        else
        {
            cv::Mat temp_r_desc(temp_r_id.size(),tar_desc.cols,tar_desc.type());//存储候选点描述子

            for (int j = 0; j <temp_r_id.size(); j++) temp_r_desc.row(j)=tar_desc.row(temp_r_id[j]).clone();

            //find best-fit target
            match_descripter_of_Pts(temp_l_desc, temp_r_desc, temp_m_dest,spt_type);

            if(temp_m_dest.size()>0){
                m_dest.push_back( cv::DMatch(temp_r_id[temp_m_dest[0].queryIdx],i,temp_m_dest[0].distance) );}
            //else{m_dest.push_back(cv::DMatch(i,-1,10000.0f));}

        }//if

    }//for(i)
}

void StereoMeasurement::run_ORB_KeyPoints_detection_and_matching(vector<cv::KeyPoint> &out_l_Pts,
                                                                 vector<cv::KeyPoint> &out_r_Pts,
                                                                 vector<cv::DMatch> &out_l_m_dest,
                                                                 cv::Mat in_l_Mask,cv::Mat in_r_Mask)
{
    if(m_state==STEREO_RECTIFY_OK)
    {
        //对BGR空间的图像直接进行计算很费时间，所以，需要转换为灰度图
        //convert image on gpu
        cv::cvtColor(m_gpu_buffer.undis_LImg,m_gpu_buffer.undis_LImgGrey,CV_BGR2GRAY);
        cv::cvtColor(m_gpu_buffer.undis_RImg,m_gpu_buffer.undis_RImgGrey,CV_BGR2GRAY);

        //upload mask to gpu
        if(in_l_Mask.size() == m_undisLeftImg.size() && in_r_Mask.size() == m_undisLeftImg.size())
        {
            m_gpu_buffer.l_mask=in_l_Mask.clone();
            m_gpu_buffer.r_mask=in_r_Mask.clone();
        }
        else
        {
            m_gpu_buffer.l_mask.setTo(cv::Scalar(255));
            m_gpu_buffer.r_mask.setTo(cv::Scalar(255));
        }

        //get ORB key points
        //首先对两幅图像进行特征点的检测(描述子的计算在后面进行计算)
        //Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
        //    matrix with non-zero values in the region of interest.
        //so 在路面进行特征点检测
        m_gpu_orb->detect(m_gpu_buffer.undis_LImgGrey,out_l_Pts,m_gpu_buffer.l_mask);
        m_gpu_orb->detect(m_gpu_buffer.undis_RImgGrey,out_r_Pts,m_gpu_buffer.r_mask);

//#ifdef DEBUG_SHOW
//        //可视化，显示关键点
//        cv::Mat ShowKeypoints;
//        drawKeypoints(m_gpu_buffer.undis_LImgGrey,out_l_Pts,ShowKeypoints,cv::Scalar(0,0,255)/*,cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);
//        imshow("leftImage关键点", ShowKeypoints);
//        cv::waitKey(0);
//#endif

        SPT_Type t_spt_type=SPT_ORB;
        matching_double_Points_process(out_l_Pts,out_r_Pts,out_l_m_dest,t_spt_type);
    }
    else
    {
        std::cout<<"Rectification of stereo images is not successful! stop points detection!" <<std::endl;
    }
}
