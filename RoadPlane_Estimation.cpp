//
// Created by nvidia on 7/17/18.
//

#include "RoadPlane_Estimation.h"

namespace rm
{
    int myget_time()
    {
        time_t t_sec0;
        struct timeb tb_start;
        ftime(&tb_start);
        time(&t_sec0);
        return (int)(t_sec0*1000+tb_start.millitm);
    }

    RoadPlane_Estimation::RoadPlane_Estimation() {}

    RoadPlane_Estimation::~RoadPlane_Estimation() {}

    void RoadPlane_Estimation::init(Stereo_CamPara& in_stereo_camPara)
    {

        //initialize stereo matcher
        auto ts=in_stereo_camPara;

        m_stereoMeasurer.initial_stereo_system(ts.image_size,ts.left_K,ts.right_K,ts.left_distor,ts.right_distor,ts.r2l_Tran,ts.min_range,ts.max_range);

        //initialize road model
        m_camHeight=ts.init_cam_height; m_camPitch=ts.init_cam_pitch;

        //initialize ENet
    }

    void RoadPlane_Estimation::run_road_plane_estimation(Mat in_leftImg, Mat in_rightImg)
    {
        //preprocess of stereo images
        int ts_pre=myget_time();
        //影像畸变纠正
        preprocess_stereo_images(in_leftImg,in_rightImg,m_undistorLimg,m_undistorRimg);
        cout<<"pre:"<<(myget_time()-ts_pre)<<"  ";
        //segment images => l/r_segImg in left or right image
        if(false)
        {
            segment_images(m_undistorLimg,m_l_segImg);
            segment_images(m_undistorRimg,m_r_segImg);
        }
        else
        {
            //just for test
            ts_pre=myget_time();
            segment_i_images(m_undistorLimg,-1,m_l_segImg);
            segment_i_images(m_undistorRimg, 1,m_r_segImg);
            m_cur_frameID++;
            cout<<"seg:"<<(myget_time()-ts_pre)<<"  ";
        }

        //get certain kind object ROI from segImg
        cv::Scalar road_color(128,64,128);

        ts_pre=myget_time();
        get_objectROI_from_segImg(m_l_segImg,road_color,m_l_roadROI);//将指定颜色的区域标白,其他区域颜色置黑
        get_objectROI_from_segImg(m_r_segImg,road_color,m_r_roadROI);
        cout<<"roi:"<<(myget_time()-ts_pre)<<"  ";

        //key points detection => l/r_ObjectKPts in left or right ObjectROI
        //find match points of l_ObjectKPts in r_ObjectKPts => ObjectKPts_matchID
        ts_pre=myget_time();
        m_stereoMeasurer.run_ORB_KeyPoints_detection_and_matching(m_l_roadKPts,m_r_roadKPts,m_roadKPts_matchID,m_l_roadROI,m_r_roadROI);
        cout<<"getP:"<<(myget_time()-ts_pre)<<"  ";

        //calculate homography matrix based on match points => Object_HMatrix: l_imgX = HMatrix*r_imgX
        ts_pre=myget_time();
        calculate_homography_matrix(m_l_roadKPts,m_r_roadKPts,m_roadKPts_matchID,m_road_HMatrix);//计算单应性矩阵
        cout<<"getH:"<<(myget_time()-ts_pre)<<"  ";

        //get object boundry based on left right image and homography matrix => Object_boundry
        ts_pre=myget_time();
        get_object_boundry(m_road_HMatrix,m_roadBoundry);//获取区域边界
        cout<<"getB:"<<(myget_time()-ts_pre)<<"  ";
    }

    void RoadPlane_Estimation::preprocess_stereo_images(Mat in_leftImg, Mat in_rightImg, Mat &out_leftImg,
                                                        Mat &out_rightImg)
    {
        //rectify distortion of image
        //影像畸变纠正
        m_stereoMeasurer.stereo_images_rectification(in_leftImg,in_rightImg,out_leftImg,out_rightImg);
    }

    void RoadPlane_Estimation::segment_images(Mat in_Img, Mat &out_segImg)
    {

    }

    //just for test , none real segment function
    void RoadPlane_Estimation::segment_i_images(Mat in_Img,int left_or_right, Mat &out_segImg)
    {
        char filename[256];

        if(left_or_right==-1) sprintf(filename,"/home/nvidia/Pictures/03/left_segments/%06d.png",m_cur_frameID);
        if(left_or_right== 1) sprintf(filename,"/home/nvidia/Pictures/03/right_segments/%06d.png",m_cur_frameID);

        Mat middel_segImg=cv::imread(filename);

        out_segImg=cv::Mat::zeros(in_Img.size(),CV_8UC3);

        middel_segImg.copyTo(out_segImg.colRange(out_segImg.cols/2 - middel_segImg.cols/2,out_segImg.cols/2 + middel_segImg.cols/2));

    }

    void RoadPlane_Estimation::get_objectROI_from_segImg(Mat in_segImg, cv::Scalar obj_classColor, Mat &out_objectROI)
    {
        //=============该方法耗时约 17/2=9 ms========================//

        // re-allocate binary map if necessary
        // same size as input image, but 1-channel
        out_objectROI.create(in_segImg.size(),CV_8UC1);


        // get the iterators
        cv::Mat_<cv::Vec3b>::const_iterator it= in_segImg.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::const_iterator itend= in_segImg.end<cv::Vec3b>();
        cv::Mat_<uchar>::iterator itout= out_objectROI.begin<uchar>();


        // for each pixel
        for ( ; it!= itend; ++it, ++itout)
        {
            // process each pixel ---------------------

            // compute distance from target color
            if ((*it)[0] == obj_classColor.val[0]
                && (*it)[1] == obj_classColor.val[1]
                && (*it)[2] == obj_classColor.val[2])
            {
                *itout= 255;
            }
            else
            {
                *itout= 0;
            }
            // end of pixel processing ----------------
        }
    }

    void RoadPlane_Estimation::calculate_homography_matrix(vector<cv::KeyPoint> in_l_KPts, vector<cv::KeyPoint> in_r_KPts,
                                                           vector<cv::DMatch> in_KPts_matchID, Mat &out_ObjectHMatrix)
    {
        vector<cv::Point2f> temp_l_pts,temp_r_pts;

        temp_l_pts.reserve(in_KPts_matchID.size());

        temp_r_pts.reserve(in_KPts_matchID.size());

        for (int i = 0; i < in_KPts_matchID.size(); ++i) {

            temp_l_pts.push_back(in_l_KPts[in_KPts_matchID[i].trainIdx].pt);

            temp_r_pts.push_back(in_r_KPts[in_KPts_matchID[i].queryIdx].pt);

        }//for(i)

        if(temp_r_pts.size()== temp_l_pts.size() && temp_r_pts.size() >= 4)
        {
            //l_p = H*r_p
            out_ObjectHMatrix = cv::findHomography(temp_r_pts,temp_l_pts,cv::RANSAC,5,cv::noArray(),2000,0.99);

            out_ObjectHMatrix.convertTo(out_ObjectHMatrix,CV_32FC1);

        }//if
        else out_ObjectHMatrix=cv::Mat();


#ifdef DEBUG_SHOW_HMATRIXRESULT
        cv::Mat r_ptsM(3,temp_r_pts.size(),CV_32FC1);
            for (int i = 0; i < temp_r_pts.size(); ++i) {
                r_ptsM.at<float>(0,i)=temp_r_pts[i].x;
                r_ptsM.at<float>(1,i)=temp_r_pts[i].y;
                r_ptsM.at<float>(2,i)=1.0f;
            }//for(i)

            cv::Mat l_ptsM=out_ObjectHMatrix*r_ptsM;

            vector<cv::Point2f> ().swap(temp_l_pts);

            temp_l_pts.reserve(temp_r_pts.size());

            for (int i = 0; i < temp_r_pts.size(); ++i) {
                temp_l_pts.push_back(cv::Point2f(l_ptsM.at<float>(0,i)/l_ptsM.at<float>(2,i), l_ptsM.at<float>(1,i)/l_ptsM.at<float>(2,i)));
            }//for(i)

            int t_iterNUM=temp_r_pts.size()/10;
            for (int i = 0; i < t_iterNUM; ++i) {
                vector<cv::KeyPoint> tt_lKPs,tt_rKPs;
                vector<cv::DMatch> tt_lMatch;
                for (int j = 0; j <10 ; ++j) {
                    tt_lKPs.push_back(cv::KeyPoint(temp_l_pts[i*t_iterNUM+j],2));
                    tt_rKPs.push_back(cv::KeyPoint(temp_r_pts[i*t_iterNUM+j],2));
                    tt_lMatch.push_back(cv::DMatch(j,j,1));
                }//for(j)
                cv::Mat show_matching_l;
                cv::drawMatches(m_stereoMeasurer.m_undisRightImg,tt_rKPs,m_stereoMeasurer.m_undisLeftImg,tt_lKPs,tt_lMatch,show_matching_l);
                cv::imshow("after H-trans",show_matching_l);
                cv::waitKey(1);
            }//for(i)
#endif
        
    }

    void RoadPlane_Estimation::get_object_boundry(Mat in_ObjectHMatrix, vector<cv::Point2i> &out_ObjectBoundry)
    {
        if(in_ObjectHMatrix.cols != 3 && in_ObjectHMatrix.rows != 3) return;
        if(in_ObjectHMatrix.type()!=CV_32FC1) in_ObjectHMatrix.convertTo(in_ObjectHMatrix,CV_32FC1);

        //calculate remaping map_x and map_y
        cv::Size img_size=m_stereoMeasurer.m_undisLeftImg.size();

        cv::Mat tmap_x(img_size,CV_32FC1);
        cv::Mat tmap_y(img_size,CV_32FC1);

        cv::Mat temp_3XN(3,img_size.width*img_size.height,CV_32FC1);

        float *fdata=(float*)temp_3XN.data;

        for (int i = 0; i <img_size.height; ++i) {
            for (int j = 0; j < img_size.width; ++j) {
                fdata[0*temp_3XN.cols+i*img_size.width+j]=(float)j;
                fdata[1*temp_3XN.cols+i*img_size.width+j]=(float)i;
                fdata[2*temp_3XN.cols+i*img_size.width+j]=1.0f;
            }//for(j)
        }//for(i)

        cv::Mat temp_result= in_ObjectHMatrix*temp_3XN;
        cv::divide(temp_result.row(0),temp_result.row(2),temp_result.row(0));
        cv::divide(temp_result.row(1),temp_result.row(2),temp_result.row(1));

        fdata=(float*)temp_result.data;

        float *fxp=(float*)tmap_x.data;
        float *fyp=(float*)tmap_y.data;

        for (int i = 0; i <img_size.height; ++i) {
            for (int j = 0; j < img_size.width; ++j) {
                fxp[i*img_size.width+j]= fdata[0*temp_result.cols+i*img_size.width+j];
                fyp[i*img_size.width+j]= fdata[1*temp_result.cols+i*img_size.width+j];
            }//for(j)
        }//for(i)

        m_stereoMeasurer.m_gpu_buffer.temp_map[0]= tmap_x.clone();
        m_stereoMeasurer.m_gpu_buffer.temp_map[1]= tmap_y.clone();

        cv::/*cuda::*/remap(m_stereoMeasurer.m_gpu_buffer.undis_LImgGrey,m_stereoMeasurer.m_gpu_buffer.temp_GreyImg,
        m_stereoMeasurer.m_gpu_buffer.temp_map[0],m_stereoMeasurer.m_gpu_buffer.temp_map[1],cv::INTER_LINEAR);

        cv::Mat temp_tranImg,ori_img;

        //cv::bilateralFilter(m_stereoMeasurer.m_gpu_buffer.temp_GreyImg,m_stereoMeasurer.m_gpu_buffer.temp_GreyImg,3,20,20);
        temp_tranImg =   m_stereoMeasurer.m_gpu_buffer.temp_GreyImg.clone();

        //cv::bilateralFilter(m_stereoMeasurer.m_gpu_buffer.undis_RImgGrey,m_stereoMeasurer.m_gpu_buffer.temp_GreyImg,3,20,20);
        ori_img = m_stereoMeasurer.m_gpu_buffer.undis_RImgGrey.clone();

        cv::GaussianBlur(ori_img,ori_img,cv::Size(5,5),2.0,2.0);
        cv::GaussianBlur(temp_tranImg,temp_tranImg,cv::Size(5,5),2.0,2.0);

        cv::absdiff(ori_img,temp_tranImg,temp_tranImg);

        temp_tranImg.convertTo(temp_tranImg,CV_8UC1);
    }

}
