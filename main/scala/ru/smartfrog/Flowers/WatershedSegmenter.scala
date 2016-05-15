package ru.smartfrog.Flowers


import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgproc._


/**
 * Helper class for section "Segmenting images using watersheds".
 * https://github.com/bytedeco/javacv-examples/blob/master/OpenCV2_Cookbook/src/main/scala/opencv2_cookbook/chapter05/WatershedSegmenter.scala
 */
class WatershedSegmenter {

  private var _markers: Mat = _


  def setMarkers(markerImage: Mat) {
    _markers = new Mat()
    markerImage.convertTo(_markers, CV_32SC1)
  }


  def process(image: Mat): Mat = {
    watershed(image, _markers)
    _markers
  }


  def segmentation: Mat = {
    // all segment with label higher than 255
    // will be assigned value 255
    val result = new Mat()
    _markers.convertTo(result, CV_8U, 1 /* scale */ , 0 /* shift */)
    result
  }


  def watersheds: Mat = {
    val result = new Mat()
    _markers.convertTo(result, CV_8U, 255 /* scale */ , 255 /* shift */)
    result
  }
}