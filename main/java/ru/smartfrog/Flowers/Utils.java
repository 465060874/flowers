package ru.smartfrog.Flowers;

import java.io.ByteArrayInputStream;

import javafx.scene.image.Image;

public class Utils {
	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	/*public static final Image mat2Image(Mat frame)
	{
		// create a temporary buffer
	/*	MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}*/
}
