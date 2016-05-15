package ru.smartfrog.Flowers;

import static org.bytedeco.javacpp.opencv_core.CV_TERMCRIT_ITER;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;

public class OpenclBOWCluster {
	FeatureDetector detector;
	DescriptorExtractor extractor;
    private Mat vocabulary;
    private int clusterCount;
    Mat trainData;
    Mat centers;
    
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

    public OpenclBOWCluster(int clusterCount)
    {
    	this.clusterCount = clusterCount;
		//detector = FeatureDetector.create(FeatureDetector.HARRIS);
		//extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
		//trainData = new Mat();
    }

    public void compute(Mat image)
    {
      /*  this.bowDescriptor = new Mat();
        KeyPointVector keypoints = new KeyPointVector();

        this.featureDetector.detect(image, keypoints);
        this.featureDetector.compute(image, keypoints, this.bowDescriptor);
        //System.out.println(this.bowDescriptor.type());
        //System.out.println(this.bowDescriptor.size().width()+" "+this.bowDescriptor.size().height());
        this.bowDescriptorExtractor.compute(image, keypoints, this.bowDescriptor);*/
    }
    

    public void cluster()
    {
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
		centers = new Mat();
		Mat labels = new Mat();
		Core.kmeans(trainData, clusterCount, labels, criteria, 5, Core.KMEANS_PP_CENTERS, centers);	
		System.out.println(centers.height()+" "+centers.width());
        //this.vocabulary = this.bowTrainer.cluster();
        //this.bowDescriptorExtractor.setVocabulary(this.vocabulary);
    }

    public void add(org.opencv.core.Mat image)
    {
		MatOfKeyPoint keypoints = new MatOfKeyPoint();
		Mat descriptors = new Mat();
		detector.detect(image, keypoints);
		extractor.compute(image, keypoints, descriptors);
        //ImageHelper.serializeMat("BOWCluster", descriptor, "C:/WORK/study/images_workspace/Flowers/resources/test/");
		trainData.push_back(descriptors);
    }

    public void save(String path) throws Exception
    {
        /*if (this.vocabulary == null)
        {
            throw new IllegalStateException("You need to call cluster() before saving the result.");
        }

        ImageHelper.serializeMat("BOWCluster", this.vocabulary, path);*/
    }

    public void load(String path) throws Exception
    {
    	
        //this.vocabulary = ImageHelper.deserializeMat("BOWCluster", path,5);
       // this.bowDescriptorExtractor.setVocabulary(this.vocabulary);
    }
}
