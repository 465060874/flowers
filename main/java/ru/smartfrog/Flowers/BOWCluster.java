package ru.smartfrog.Flowers;

import static org.bytedeco.javacpp.opencv_core.*;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_features2d.*;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_objdetect.HOGDescriptor;
import org.bytedeco.javacpp.opencv_xfeatures2d.SIFT;
import org.bytedeco.javacpp.opencv_xfeatures2d.SURF;

/**
 * Bag of Word cluster module.
 * https://github.com/Sapphirine/hvision/blob/master/core/src/main/java/com/emadbarsoum/lib/BOWCluster.java
 */
public class BOWCluster
{
    private TermCriteria termCriteria;
    private BOWImgDescriptorExtractor bowDescriptorExtractor;
    private BOWKMeansTrainer bowTrainer;
    private SIFT featureDetector;
    private SIFT descriptorExtractor;
    private BFMatcher matcher;
    private Mat vocabulary;
    private int clusterCount;
    private Mat bowDescriptor;
    private int descriptorSize = 25;
    private int descriptorSpacing = 20;
    
    public int getClusterCount()
    {
    	return clusterCount;
    }

    public Mat getVocabulary()
    {
        return this.vocabulary;
    }

    public void setVocabulary(Mat vocabulary)
    {
        this.vocabulary = vocabulary;
    }

    public BOWCluster(int clusterCount, int descriptorSize)
    {
    	this.clusterCount = clusterCount;
    	this.descriptorSize = descriptorSize;
        this.termCriteria = new TermCriteria(CV_TERMCRIT_ITER, 100, 0.001);
		int nFeatures = 0;
		int nOctaveLayers = 6;
		double contrastThreshold = 0.03;
		int edgeThreshold = 10;
		double sigma = 1.6;
       //this.featureDetector = SURF.create(500, 4, 2, true, false);//SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
       this.descriptorExtractor = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);//SURF.create(500, 4, 2, true, false);//SURF.create();
       //HOG = 
        //this.descriptorExtractor = DescriptorExtractor.create("SURF");
        this.matcher = new BFMatcher();

        this.bowTrainer = new BOWKMeansTrainer(clusterCount, this.termCriteria, 1, 2);
        this.bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor,matcher);
    }
    public Mat compute(Mat image)
    {
        this.bowDescriptor = new Mat();
        KeyPointVector keypoints = calcKeypoints(image);
        Mat descriptor = new Mat();
        this.descriptorExtractor.compute(image, keypoints, descriptor); 
        this.bowDescriptorExtractor.compute(image, keypoints, this.bowDescriptor);
        return bowDescriptor;
    }
 
    private KeyPointVector calcKeypoints(Mat image)
    {
    	int height = image.size().height();
        int width = image.size().width();
        int size = (height - descriptorSize)*(width - descriptorSize)/(descriptorSpacing*descriptorSpacing);
        KeyPointVector keypoints = new KeyPointVector(size);
        //System.out.println(keypoints.size());
        int npoint = 0;

    	for (int y=descriptorSize; y+descriptorSize<height; y+=descriptorSpacing)
    		for (int x=descriptorSize; x+descriptorSize<width; x+=descriptorSpacing)
    		{
    			KeyPoint k = new KeyPoint(x+descriptorSize/2, y+descriptorSize/2, descriptorSize);
    			
    			keypoints.put(npoint,k);
    			npoint++;
    		}
        return keypoints;    	
    }
    public Mat computeSimpleDescriptors(Mat image)
    {
        Mat descriptor = new Mat();
        KeyPointVector keypoints = calcKeypoints(image);
        //System.out.println(keypoints.size());
        this.descriptorExtractor.compute(image, keypoints, descriptor); 
        //System.out.println(descriptor.size().height()+" "+descriptor.size().width());
        return descriptor;
        // Mat featureImage = new Mat();
		// opencv_features2d.drawKeypoints(image, keypoints, featureImage, new Scalar(255, 255, 255, 0), DrawMatchesFlags.DRAW_RICH_KEYPOINTS);
		// OpenCVUtils.show(featureImage, "SIFT Features");
    }    

    public void cluster()
    {
        this.vocabulary = this.bowTrainer.cluster();
        this.bowDescriptorExtractor.setVocabulary(this.vocabulary);
    }

    public void add(Mat image)
    {
        Mat descriptor = computeSimpleDescriptors(image);
         //System.out.println(bowDescriptor.size().height()+" "+bowDescriptor.size().width());
        this.bowTrainer.add(descriptor);
    }

    public void clear()
    {
        this.bowTrainer.clear();
    }

    public void save(String path) throws Exception
    {
        if (this.vocabulary == null)
        {
            throw new IllegalStateException("You need to call cluster() before saving the result.");
        }

        ImageHelper.serializeMat("BOWCluster", this.vocabulary, path);
    }

    public void load(String path) throws Exception
    {
        this.vocabulary = ImageHelper.deserializeMat("BOWCluster", path,opencv_core.CV_32F);
        this.bowDescriptorExtractor.setVocabulary(this.vocabulary);
    }
}