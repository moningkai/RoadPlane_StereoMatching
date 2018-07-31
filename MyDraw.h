/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef MY_DRAW_CPP
#define MY_DRAW_CPP

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

#include <opencv2/opencv.hpp>

//#include <opencv2/features2d.hpp>
//#include "opencv2/imgproc.hpp"
//
//#include "opencv2/core/utility.hpp"
//#include "opencv2/core/private.hpp"
//#include "opencv2/core/ocl.hpp"
//#include "opencv2/core/hal/hal.hpp"

namespace rm
{
/*
 * Functions to draw keypoints and matches.
 */
    static inline void
    _drawKeypoint(cv::InputOutputArray img, const cv::KeyPoint &p, const cv::Scalar &color, int flags)
    {
        CV_Assert(!img.empty());
        cv::Point center(cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier));

        if (flags & cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS)
        {
            int radius = cvRound(p.size / 2 * draw_multiplier); // KeyPoint::size is a diameter

            // draw the circles around keypoints with the keypoints size
            circle(img, center, radius, color, 1, cv::LINE_AA, draw_shift_bits);

            // draw orientation of the keypoint, if it is applicable
            if (p.angle != -1)
            {
                float srcAngleRad = p.angle * (float) CV_PI / 180.f;
                cv::Point orient(cvRound(cos(srcAngleRad) * radius),
                                 cvRound(sin(srcAngleRad) * radius)
                );
                line(img, center, center + orient, color, 1, cv::LINE_AA, draw_shift_bits);
            }
#if 0
            else
        {
            // draw center with R=1
            int radius = 1 * draw_multiplier;
            circle( img, center, radius, color, 1, LINE_AA, draw_shift_bits );
        }
#endif
        }
        else
        {
            // draw center with R=3
            int radius = 3 * draw_multiplier;
            circle(img, center, radius, color, 1, cv::LINE_AA, draw_shift_bits);
        }
    }

//    void drawKeypoints( cv::InputArray image, const std::vector<cv::KeyPoint>& keypoints, cv::InputOutputArray outImage,
//                        const cv::Scalar& _color, int flags )
//    {
//        //cv::CV_INSTRUMENT_REGION()
//
//        if( !(flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG) )
//        {
//            if( image.type() == CV_8UC3 )
//            {
//                image.copyTo( outImage );
//            }
//            else if( image.type() == CV_8UC1 )
//            {
//                cvtColor( image, outImage, cv::COLOR_GRAY2BGR );
//            }
//            else
//            {
//                CV_Error( cv::Error::StsBadArg, "Incorrect type of input image.\n" );
//            }
//        }
//
//        cv::RNG& rng=cv::theRNG();
//        bool isRandColor = _color == cv::Scalar::all(-1);
//
//        CV_Assert( !outImage.empty() );
//        std::vector<cv::KeyPoint>::const_iterator it = keypoints.begin(),
//                end = keypoints.end();
//        for( ; it != end; ++it )
//        {
//            cv::Scalar color = isRandColor ? cv::Scalar(rng(256), rng(256), rng(256)) : _color;
//            _drawKeypoint( outImage, *it, color, flags );
//        }
//    }

    static void _prepareImgAndDrawKeypoints(cv::InputArray img1, const std::vector<cv::KeyPoint> &keypoints1,
                                            cv::InputArray img2, const std::vector<cv::KeyPoint> &keypoints2,
                                            cv::InputOutputArray _outImg, cv::Mat &outImg1, cv::Mat &outImg2,
                                            const cv::Scalar &singlePointColor, int flags)
    {
        cv::Mat outImg;
        cv::Size img1size = img1.size(), img2size = img2.size();
        cv::Size size(MAX(img1size.width, img2size.width), img1size.height + img2size.height);
        //cv::Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height) );
        if (flags & cv::DrawMatchesFlags::DRAW_OVER_OUTIMG)
        {
            outImg = _outImg.getMat();
            if (size.width > outImg.cols || size.height > outImg.rows)
                CV_Error(cv::Error::StsBadSize, "outImg has size less than need to draw img1 and img2 together");
            //outImg1 = outImg( cv::Rect(0, 0, img1size.width, img1size.height) );
            //outImg2 = outImg( cv::Rect(img1size.width, 0, img2size.width, img2size.height) );

            outImg1 = outImg(cv::Rect(0, 0, img1size.width, img1size.height));
            outImg2 = outImg(cv::Rect(0, img1size.height, img2size.width, img2size.height));
        }
        else
        {
            _outImg.create(size, CV_MAKETYPE(img1.depth(), 3));
            outImg = _outImg.getMat();
            outImg = cv::Scalar::all(0);
            //outImg1 = outImg( cv::Rect(0, 0, img1size.width, img1size.height) );
            //outImg2 = outImg( cv::Rect(img1size.width, 0, img2size.width, img2size.height) );

            outImg1 = outImg(cv::Rect(0, 0, img1size.width, img1size.height));
            outImg2 = outImg(cv::Rect(0, img1size.height, img2size.width, img2size.height));

            if (img1.type() == CV_8U)
                cvtColor(img1, outImg1, cv::COLOR_GRAY2BGR);
            else
                img1.copyTo(outImg1);

            if (img2.type() == CV_8U)
                cvtColor(img2, outImg2, cv::COLOR_GRAY2BGR);
            else
                img2.copyTo(outImg2);
        }

        // draw keypoints
        if (!(flags & cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS))
        {
            cv::Mat _outImg1 = outImg(cv::Rect(0, 0, img1size.width, img1size.height));
            drawKeypoints(_outImg1, keypoints1, _outImg1, singlePointColor,
                          flags | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

            //cv::Mat _outImg2 = outImg( cv::Rect(img1size.width, 0, img2size.width, img2size.height) );
            cv::Mat _outImg2 = outImg(cv::Rect(0, img1size.height, img2size.width, img2size.height));
            drawKeypoints(_outImg2, keypoints2, _outImg2, singlePointColor,
                          flags | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        }
    }

    static inline void
    _drawVerticalMatch(cv::InputOutputArray outImg, cv::InputOutputArray outImg1, cv::InputOutputArray outImg2,
               const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Scalar &matchColor, int flags)
    {
        cv::RNG &rng = cv::theRNG();
        bool isRandMatchColor = matchColor == cv::Scalar::all(-1);
        cv::Scalar color = isRandMatchColor ? cv::Scalar(rng(256), rng(256), rng(256)) : matchColor;

        _drawKeypoint(outImg1, kp1, color, flags);
        _drawKeypoint(outImg2, kp2, color, flags);

        cv::Point2f pt1 = kp1.pt,
                pt2 = kp2.pt,
                //dpt2 = cv::Point2f(std::min(pt2.x + outImg1.size().width, float(outImg.size().width - 1)), pt2.y),
                dpt2 = cv::Point2f(pt2.x, std::min(pt2.y + outImg1.size().height, float(outImg.size().height - 1)));


        line(outImg,
             cv::Point(cvRound(pt1.x * draw_multiplier), cvRound(pt1.y * draw_multiplier)),
             cv::Point(cvRound(dpt2.x * draw_multiplier), cvRound(dpt2.y * draw_multiplier)),
             color, 1, cv::LINE_AA, draw_shift_bits);
    }

    void drawVerticalMatches(cv::InputArray img1, const std::vector<cv::KeyPoint> &keypoints1,
                     cv::InputArray img2, const std::vector<cv::KeyPoint> &keypoints2,
                     const std::vector<cv::DMatch> &matches1to2, cv::InputOutputArray outImg,
                     const cv::Scalar& matchColor=cv::Scalar::all(-1), const cv::Scalar& singlePointColor=cv::Scalar::all(-1),
                     const std::vector<char>& matchesMask=std::vector<char>(), int flags=cv::DrawMatchesFlags::DEFAULT)
    {
        if (!matchesMask.empty() && matchesMask.size() != matches1to2.size())
            CV_Error(cv::Error::StsBadSize, "matchesMask must have the same size as matches1to2");

        cv::Mat outImg1, outImg2;
        _prepareImgAndDrawKeypoints(img1, keypoints1, img2, keypoints2,
                                    outImg, outImg1, outImg2, singlePointColor, flags);

        // draw matches
        for (size_t m = 0; m < matches1to2.size(); m++)
        {
            if (matchesMask.empty() || matchesMask[m])
            {
                int i1 = matches1to2[m].queryIdx;
                int i2 = matches1to2[m].trainIdx;
                CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
                CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));

                const cv::KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
                _drawVerticalMatch(outImg, outImg1, outImg2, kp1, kp2, matchColor, flags);
            }
        }
    }

    void drawVerticalMatches(cv::InputArray img1, const std::vector<cv::KeyPoint> &keypoints1,
                             cv::InputArray img2, const std::vector<cv::KeyPoint> &keypoints2,
                             const std::vector<std::vector<cv::DMatch> > &matches1to2, cv::InputOutputArray outImg,
                             const cv::Scalar& matchColor=cv::Scalar::all(-1), const cv::Scalar& singlePointColor=cv::Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=cv::DrawMatchesFlags::DEFAULT)
    {
        if (!matchesMask.empty() && matchesMask.size() != matches1to2.size())
            CV_Error(cv::Error::StsBadSize, "matchesMask must have the same size as matches1to2");

        cv::Mat outImg1, outImg2;
        _prepareImgAndDrawKeypoints(img1, keypoints1, img2, keypoints2,
                                    outImg, outImg1, outImg2, singlePointColor, flags);

        // draw matches
        for (size_t i = 0; i < matches1to2.size(); i++)
        {
            for (size_t j = 0; j < matches1to2[i].size(); j++)
            {
                int i1 = matches1to2[i][j].queryIdx;
                int i2 = matches1to2[i][j].trainIdx;
                if (matchesMask.empty() || matchesMask[i][j])
                {
                    const cv::KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
                    _drawVerticalMatch(outImg, outImg1, outImg2, kp1, kp2, matchColor, flags);
                }
            }
        }
    }
}

#endif