#include <jni.h>

#include "org_opencv_samples_facedetect_FastCodeTracker.h"

#include "objdetect.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include <android/log.h>



#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace cv;
using namespace std;

#include "mycascadeclassifier.hpp"


inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

void myDetectAndDisplay( Mat* frame,Mat* mat_faces, int size );

MyCascadeClassifier * face_cascade_pointer;
int size;
string window_name = "Capture - Face detection";

/*
 * Class:     org_opencv_samples_facedetect_FastCodeTracker
 * Method:    nativeCreateObject
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_org_opencv_samples_facedetect_FastCodeTracker_nativeCreateObject
(JNIEnv * jenv, jclass, jstring jFileName, jint face_size){
    jlong result = 0;
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);
    String fileName= stdFileName;
    
    try
    {
        face_cascade_pointer = new MyCascadeClassifier();
        face_cascade_pointer->load(fileName);
    
        result = (jlong)face_cascade_pointer;
        size = face_size;
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }
    return result;
}


/*
 * Class:     org_opencv_samples_facedetect_FastCodeTracker
 * Method:    nativeDestroyObject
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_FastCodeTracker_nativeDestroyObject
(JNIEnv *, jclass, jlong thiz){
    if(thiz != 0)
    {
        delete (MyCascadeClassifier*)thiz;
    }
}

/*
 * Class:     org_opencv_samples_facedetect_FastCodeTracker
 * Method:    nativeStart
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_FastCodeTracker_nativeStart
(JNIEnv *, jclass, jlong){}

/*
 * Class:     org_opencv_samples_facedetect_FastCodeTracker
 * Method:    nativeStop
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_FastCodeTracker_nativeStop
(JNIEnv *, jclass, jlong){}




/*
 * Class:     org_opencv_samples_facedetect_FastCodeTracker
 * Method:    nativeDetectAndDisplay
 * Signature: (JJJI)V
 */
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_FastCodeTracker_nativeDetectAndDisplay
(JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong mat_faces, jint size){
    myDetectAndDisplay((Mat *)imageGray,(Mat*)mat_faces,size);
}



void myDetectAndDisplay( Mat * frame, Mat * mat_faces, int size )
{
    //int size = this.size;  // default 30 / the larger the better.
    //size=30;
    std::vector<Rect> faces;
    Mat frame_gray;
    frame_gray = *frame;
    int jj;
    

    //-- Detect faces
    
    if(face_cascade_pointer!=NULL){
        face_cascade_pointer->detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(size, size) );
    }

    vector_Rect_to_Mat(faces, *((Mat*)mat_faces));
}




