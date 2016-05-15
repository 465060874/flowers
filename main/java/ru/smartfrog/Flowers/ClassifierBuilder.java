package ru.smartfrog.Flowers;

import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import static org.bytedeco.javacpp.opencv_imgproc.threshold;

import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.util.ArrayList;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.KeyPointVector;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_features2d;
import org.bytedeco.javacpp.opencv_features2d.DrawMatchesFlags;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ml;
import org.bytedeco.javacpp.opencv_ml.KNearest;
import org.bytedeco.javacpp.opencv_ml.SVM;
import org.bytedeco.javacpp.opencv_ml.TrainData;
import org.bytedeco.javacpp.opencv_xfeatures2d.SIFT;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;



import ru.smartfrog.Flowers.OpenCVUtils;
public class ClassifierBuilder {

	BOWCluster cluster;
	KNearest classificator;
	ArrayList<String> pathes;
	String configPath;
	
	public ClassifierBuilder(int vocabularySize, String configPath, ArrayList<String> pathes)
	{
		this.configPath = configPath;
		this.pathes = pathes;
		cluster = new BOWCluster(vocabularySize,25);
	}
	
	private Mat loadAndNormalize(String file)
	{
		Mat rawImage = imread(file);
		return normalize(rawImage);
	}
	private Mat normalize(Mat rawImage)
	{
		Mat image = new Mat();
		int height = rawImage.size().height();
		int width = rawImage.size().width();
		int min = height<width?height:width;
		opencv_imgproc.resize(rawImage, image, new Size((width*100)/min,(height*100)/min));
		image = OpenCVUtils.watershedSegmentation(image);
		return image;
	}	
	private ArrayList<String> getFilesFromDir(String dataPath)
	{
		File f = new File(dataPath);
        FilenameFilter filter = new FilenameFilter(){
            public boolean accept(File dir, String name){
                if(name.indexOf(".jpg") > 0) return true;   
                else return false;
            }
        };
        File[] files = f.listFiles(filter);   
        ArrayList<String> names =new ArrayList<String>(); 
        for(int i=0; i<files.length; i++){
            names.add(files[i].getName());
        }
        return names;
	}
	
	public void genVocabulary() throws Exception
	{
		for (String dataPath:pathes)
		{
			ArrayList<String> fileNames = getFilesFromDir(dataPath);
			for (String file:fileNames)
			{
				Mat image = loadAndNormalize(dataPath+file);
			    if (image.empty()) {
			    	continue;
			    }
				cluster.add(image);      
			} 
		}
		cluster.cluster();
		cluster.save(configPath+"vocabulary.yml");
		
	}
	
	public void reverseMap() throws Exception
	{
		ArrayList<ArrayList<String>> patches = new ArrayList<ArrayList<String>>();
		int size = 0;
		for (String dataPath:this.pathes)
		{
			ArrayList<String> fileNames = getFilesFromDir(dataPath);
			patches.add(fileNames);
			size += fileNames.size();
		}
		Mat trainData = new Mat();
		Mat labels = new Mat(size,1,opencv_core.CV_32S);
		IntIndexer labelsIndexer = (IntIndexer)labels.createIndexer();
		int classIndex=0;
		int sampleIndex=0;
		for (ArrayList<String> fileNames:patches)
		{
	 		for (String file:fileNames)
			{
				Mat image = loadAndNormalize(this.pathes.get(classIndex)+file);
			    if (image.empty()) {
			    	continue;
			    }
			    Mat bowDescriptor = cluster.compute(image);  
			    if (bowDescriptor == null)
			    	continue;
				trainData.push_back(bowDescriptor);
				labelsIndexer.put(sampleIndex, 0, classIndex);
				sampleIndex++;
			} 
	 		classIndex++;
		}
		if (labels.size().height() > sampleIndex)
			labels.resize(sampleIndex);
		labelsIndexer.release();
 		//ImageHelper.serializeMat("reverse", trainData, configPath+"reverse.yml");
		//ImageHelper.serializeMat("labels", labels, configPath+"labels.yml");
		//Mat intLabels = new Mat();
		//labels.convertTo(intLabels, opencv_core.CV_32S);
		classificator = KNearest.create();
		classificator.setDefaultK(1);
		classificator.setIsClassifier(true);
		classificator.train(trainData, opencv_ml.ROW_SAMPLE, labels);
		labels.release();
		trainData.release();
		
	}
		
	public double recognize(Mat image) throws Exception
	{
		Mat clearImage = normalize(image);
		Mat bowDescriptor = cluster.compute(clearImage);  
		Mat results = new Mat();
		classificator.predict(bowDescriptor,results,0);
	    //OpenCVUtils.show(clearImage,"");
		FloatIndexer resultsIndexer = (FloatIndexer)results.createIndexer();
		//for (int i = 0;i<results.)
		return resultsIndexer.get(0);
	}
	
	public void load() throws Exception
	{
		cluster.load(configPath+"vocabulary.yml");
		Mat trainData = new Mat();
		Mat labels = new Mat();
		trainData = ImageHelper.deserializeMat("reverse", configPath+"reverse.yml",opencv_core.CV_32F);
		labels = ImageHelper.deserializeMat("labels",configPath+"labels.yml",opencv_core.CV_32S);
		Mat intLabels = new Mat();
		labels.convertTo(intLabels, opencv_core.CV_32F);
		classificator = KNearest.create();
		classificator.setDefaultK(1);
		classificator.setIsClassifier(true);
		classificator.train(trainData, opencv_ml.ROW_SAMPLE, intLabels);
		labels.release();
		trainData.release();
		
	}
	
	public double test(ArrayList<String> dataPathes) throws Exception
	{
		
		ArrayList<ArrayList<String>> patches = new ArrayList<ArrayList<String>>();
		int size = 0;
		for (String dataPath:dataPathes)
		{
			ArrayList<String> fileNames = getFilesFromDir(dataPath);
			patches.add(fileNames);
			size += fileNames.size();
		}
		Mat trainData = new Mat();
		Mat labels = new Mat(size,1,opencv_core.CV_32S);
		IntIndexer labelsIndexer = (IntIndexer)labels.createIndexer();
		int classIndex=0;
		int sampleIndex=0;

		for (ArrayList<String> fileNames:patches)
		{
	 		for (String file:fileNames)
			{
				Mat image = loadAndNormalize(dataPathes.get(classIndex)+file);
			    if (image.empty()) {
			    	continue;
			    }
			    OpenCVUtils.show(image, this.pathes.get(classIndex)+file);

			    Mat bowDescriptor = cluster.compute(image);  
				trainData.push_back(bowDescriptor);
				labelsIndexer.put(sampleIndex, 0, classIndex);
				sampleIndex++;
				//Mat results = new Mat();
			} 
	 		classIndex++;
		}
		TrainData data = TrainData.create(trainData, opencv_ml.ROW_SAMPLE, labels);
	    return classificator.calcError(data,true,labels);
	}

	public void crop(ArrayList<String> dataPathes) throws Exception
	{
		
		ArrayList<ArrayList<String>> patches = new ArrayList<ArrayList<String>>();
		int size = 0;
		for (String dataPath:dataPathes)
		{
			ArrayList<String> fileNames = getFilesFromDir(dataPath);
			patches.add(fileNames);
			size += fileNames.size();
		}
		Mat trainData = new Mat();
		int classIndex=0;

		for (ArrayList<String> fileNames:patches)
		{
	 		for (String file:fileNames)
			{
				normalizeSimple(dataPathes.get(classIndex)+file,file);
			  
			} 
	 		classIndex++;
		}
	}
	private Mat normalizeSimple(String file, String filename)
	{
		Mat rawImage = imread(file);
		Mat image = new Mat();
		int height = rawImage.size().height();
		int width = rawImage.size().width();
		int min = height<width?height:width;
		opencv_imgproc.resize(rawImage, image, new Size((width*100)/min,(height*100)/min));
		OpenCVUtils.save(new File("C:/WORK/study/images_workspace/Flowers/resources/config/images/"+filename),image);
		image = OpenCVUtils.watershedSegmentation(image);
		OpenCVUtils.save(new File("C:/WORK/study/images_workspace/Flowers/resources/config/crops/"+filename),image);
		return image;
	}			
	public static void main(String[] args) throws Exception {

		ArrayList<String> patches = new ArrayList<String>();
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/glazki/");
		/*patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/very_small/c2/"); // глазки 0
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/very_small/c4/"); // ромашки 1
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/very_small/c1/"); // нарциссы 2*/
		ClassifierBuilder builder = new ClassifierBuilder(100,"C:/WORK/study/images_workspace/Flowers/resources/config/",patches);
		builder.crop(patches);
		//System.out.println("Calc vocabulary");
		/*builder.genVocabulary();
		//System.out.println("Reverse vocabulary");
		builder.reverseMap();
		//builder.load();
		System.out.println("Test vocabulary");
		System.out.println(builder.test (patches));
		Mat image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/glazki/image_04235.jpg"); //0
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/glazki/image_04190.jpg"); // 0
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/glazki/image_04227.jpg"); // 0
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/glazki/image_04231.jpg"); // 0
		System.out.println(builder.recognize(image));
		//image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05711.jpg");
		//System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/camomile/image_06228.jpg"); //1
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/camomile/image_06212.jpg"); //1
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/camomile/image_06223.jpg"); //1
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05743.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05737.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05734.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05726.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05724.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05719.jpg"); //2
		System.out.println(builder.recognize(image));
		image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/narciss/image_05709.jpg"); //2
		System.out.println(builder.recognize(image));
		//image = imread("C:/WORK/study/images_workspace/Flowers/resources/jpg/kupalnica/image_04625.jpg");
		//System.out.println(builder.recognize(image));
		/*patches.add("C:/WORK/study/images_workspace/Flowers/resources/test/viola/");
		//patches.add("C:/WORK/study/images_workspace/Flowers/resources/test/iris/");
		/*patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/train/gibiskus/");
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/train/akvilegia/");
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/train/ciklamen/"); 
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/oduvan/");
		patches.add("C:/WORK/study/images_workspace/Flowers/resources/jpg/strange/");
		/*builder.genTraining(patches);
		builder.reverseMap(patches);/*
		builder.load();
		Mat image = builder.loadAndNormalize("C:/WORK/study/images_workspace/Flowers/resources/jpg/oduvan/image_06570.jpg");
		builder.recognize(image);
		image = builder.loadAndNormalize("C:/WORK/study/images_workspace/Flowers/resources/jpg/gerbera/image_02315.jpg");
		builder.recognize(image);
		builder.test (patches);*/
		/*Mat train = new Mat(3,2,opencv_core.CV_32F);
		FloatIndexer trainIndexer = (FloatIndexer)train.createIndexer();
		trainIndexer.put(0, 0, 1);
		trainIndexer.put(0, 1, 1);
		trainIndexer.put(1, 0, 3);
		trainIndexer.put(1, 1, 12);
		trainIndexer.put(2, 0, 4);
		trainIndexer.put(2, 1, 12);
		
		Mat test = new Mat(1,2,opencv_core.CV_32F);
		FloatIndexer testIndexer = (FloatIndexer)train.createIndexer();
		trainIndexer.put(0, 0, 2);
		trainIndexer.put(1, 0, 13);
		
		Mat labels = new Mat(3,1,opencv_core.CV_32S);
		IntIndexer labelsIndexer = (IntIndexer)labels.createIndexer();		
		labelsIndexer.put(0, 0, 1);
		labelsIndexer.put(1, 0, 12);
		labelsIndexer.put(2, 0, 12);
		KNearest classificator = KNearest.create();
		classificator.train(train,opencv_ml.ROW_SAMPLE, labels);
		System.out.println(classificator.predict(test));*/
		System.out.println("Complete!");

	}
	
}
