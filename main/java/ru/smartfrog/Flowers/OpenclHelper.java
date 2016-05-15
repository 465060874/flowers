package ru.smartfrog.Flowers;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javafx.scene.image.Image;

public class OpenclHelper {

	OpenclBOWCluster cluster;
	public OpenclHelper(int vocabularySize)
	{
		cluster = new OpenclBOWCluster(vocabularySize);
	}

	private Mat loadAndNormalize(String file)
	{
		Mat rawImage = Imgcodecs.imread(file);
		/*Mat image = new Mat();
		int height = rawImage.height();
		int width = rawImage.width();
		int min = height<width?height:height;
		//opencv_imgproc.resize(rawImage, image, new Size((width*100)/min,(height*100)/min));*/
		return rawImage;
	}

	/**
	 * Perform the operations needed for removing a uniform background
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image with only foreground objects
	 */
	public Mat doBackgroundRemoval(Mat frame)
	{
		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();
		
		int thresh_type = Imgproc.THRESH_BINARY_INV;
		
		// threshold the image with the average hue value
		hsvImg.create(frame.size(), CvType.CV_8U);
		Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
		Core.split(hsvImg, hsvPlanes);
		
		// get the average hue value of the image
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));
		
		Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 255.0, thresh_type);
			
		Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));
		
		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);
		
		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);
		
		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		frame.copyTo(foreground, thresholdImg);
		
		return foreground;
	}
	/**
	 * Get the average hue value of the image starting from its Hue channel
	 * histogram
	 * 
	 * @param hsvImg
	 *            the current frame in HSV
	 * @param hueValues
	 *            the Hue component of the current frame
	 * @return the average Hue value
	 */
	private double getHistAverage(Mat hsvImg, Mat hueValues)
	{
		// init
		double average = 0.0;
		Mat hist_hue = new Mat();
		// 0-180: range of Hue values
		MatOfInt histSize = new MatOfInt(180);
		List<Mat> hue = new ArrayList<>();
		hue.add(hueValues);
		
		// compute the histogram
		Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));
		
		// get the average Hue value of the image
		// (sum(bin(h)*h))/(image-height*image-width)
		// -----------------
		// equivalent to get the hue of each pixel in the image, add them, and
		// divide for the image size (height and width)
		for (int h = 0; h < 180; h++)
		{
			// for each bin, get its value and multiply it for the corresponding
			// hue
			average += (hist_hue.get(h, 0)[0] * h);
		}
		
		// return the average hue of the image
		return average = average / hsvImg.size().height / hsvImg.size().width;
	}
	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	private Image mat2Image(Mat frame)
	{
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}
	
	void cropImage(String file)
	{
		Mat rawImage = Imgcodecs.imread(file);
		Mat img = doBackgroundRemoval(rawImage);
		Imgcodecs.imwrite("C:/WORK/study/images_workspace/Flowers/resources/tmp.jpg", img);
	}
	
	public static void main(String[] args) throws Exception {
		 System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		OpenclHelper builder = new OpenclHelper(500);
		ArrayList<String> patches = new ArrayList<String>();
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/very_small/c1/");
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/very_small/c2/");
		builder.cropImage("C:/WORK/study/images_workspace/Flowers/resources/jpg/vetrenica/image_05992.jpg");
		System.out.println("Complete!");

	}
}
