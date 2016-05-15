package ru.smartfrog.Flowers;

import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * A helper class that simplify dealing with JavaCV images.
 * https://github.com/Sapphirine/hvision/blob/master/core/src/main/java/com/emadbarsoum/common/ImageHelper.java
 */
public class ImageHelper
{
    // Creating IplImage from a raw uncompressed image data.
    public static IplImage createIplImageFromRawBytes(byte[] imageData, int length, MetadataParser metadata)
    {
        int width = metadata.getAsInt("width");
        int height = metadata.getAsInt("height");
        int channelCount = metadata.getAsInt("channel_count");
        int depth = metadata.getAsInt("depth");

        return createIplImageFromRawBytes(imageData, length, width, height, channelCount, depth);
    }

    // Creating IplImage from a raw uncompressed image data.
    public static IplImage createIplImageFromRawBytes(byte[] imageData, int length, int width, int height, int channelCount, int depth)
    {
        IplImage image = IplImage.create(width, height, depth, channelCount);

        ByteBuffer buffer = image.getByteBuffer();
        byte[] rawBuffer = Arrays.copyOf(imageData, length);
        buffer.put(rawBuffer);

        return image;
    }

    public static void serializeMat(String name, Mat mat, String path) throws Exception
    {
    	//System.out.println(mat.size().width()+" "+mat.size().height());
        FileStorage storage = new FileStorage();
        storage.open(path, FileStorage.WRITE);
        CvMat mat1 = new CvMat(mat);
        storage.writeObj(name, mat1);
        storage.close();
        //storage.release();
    }

    public static Mat deserializeMat(String name, String path,int type) throws Exception
    {
        FileStorage storage = new FileStorage(path, FileStorage.READ);
        CvMat mat1 = new CvMat(storage.get(name).readObj());
        //System.out.println(mat1.height()+" "+mat1.width());
        
        Mat mat = new Mat(mat1.height(),mat1.width(),type);
        mat.data().put(mat1);
        
//        System.out.println(mat.size().width()+" "+mat.size().height());
        storage.close();
        return mat;
    }

   /*public static MatData matToBytes(Mat mat)
    {
        return MatData.create(mat);
    }

    public static Mat bytesToMat(MatData matData)
    {
        return matData.toMat();
    }*/
}
