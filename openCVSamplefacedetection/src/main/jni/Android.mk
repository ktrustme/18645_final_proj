LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#opencv
#OPENCV_CAMERA_MODULES:=on
#OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
OPENCV_ROOT:=/Users/kuoxin/StudyStudy/CMUStudy/18645How_to_write_fast_code/FinalProj645/OpenCV-android-sdk
include ${OPENCV_ROOT}/sdk/native/jni/OpenCV.mk

 LOCAL_SRC_FILES := org_opencv_samples_facedetect_FastCodeTracker.cpp mycascadeclassifier.cpp datamatrix.cpp distancetransform.cpp featurepyramid.cpp fft.cpp haar.cpp hog.cpp latentsvm.cpp latentsvmdetector.cpp matching.cpp objdetect_init.cpp resizeimg.cpp routine.cpp
 LOCAL_C_INCLUDES += $(LOCAL_PATH)
 LOCAL_LDLIBS     += -llog -ldl
 LOCAL_CFLAGS += -fopenmp
 LOCAL_LDFLAGS += -fopenmp

 LOCAL_MODULE     := fast_code_tracker 

 include $(BUILD_SHARED_LIBRARY)
