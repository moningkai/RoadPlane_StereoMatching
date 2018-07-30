cmake_minimum_required(VERSION 3.9)
project(RoadPlane_StereoMatching)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

aux_source_directory(./ SOURCE_FILE)

add_executable(RoadPlane_StereoMatching ${SOURCE_FILE})

target_link_libraries(RoadPlane_StereoMatching ${OpenCV_LIBS})

#定义DEBUG_SHOW宏,该宏用于开发调试,显示过程信息
#option(DEBUG_mode "ON for debug or OFF for release" ON)
#IF(DEBUG_mode)
#ADD_DEFINITIONS(-DDEBUG_SHOW_RECTIFYIMAGE)
ADD_DEFINITIONS(-DDEBUG_SHOW_KERPOINTMATCH)
#ADD_DEFINITIONS(-DDEBUG_SHOW_HMATRIXRESULT)
#ENDIF()