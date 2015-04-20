package org.opencv.samples.facedetect;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

public class FastCodeTracker
{
    private long mNativeObj = 0;
    private int size = 0;

    public FastCodeTracker(String cascadeName, int minFaceSize) {
        this.size = minFaceSize;
        mNativeObj = nativeCreateObject(cascadeName, minFaceSize);
    }

    public void start() {
        //nativeStart(mNativeObj);
    }

    public void stop() {
        //nativeStop(mNativeObj);
    }
    public void setMinFaceSize(int size) {
        this.size=size;
    }

    /*
    public void setMinFaceSize(int size) {
        nativeSetFaceSize(mNativeObj, size);
    }
    */

    public void detect(Mat imageGray, MatOfRect faces) {
        //That's it! The ultimate goal of our maximization!
        //nativeDetect(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr());
        nativeDetectAndDisplay(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr(), size);
    }

    public void release() {
        nativeDestroyObject(mNativeObj);
        mNativeObj = 0;
    }



    private static native long nativeCreateObject(String cascadeName, int minFaceSize);
    private static native void nativeDestroyObject(long thiz);
    private static native void nativeStart(long thiz);
    private static native void nativeStop(long thiz);
    private static native void nativeDetectAndDisplay(long thiz, long inputImage, long faces, int minFaceSize);
}
