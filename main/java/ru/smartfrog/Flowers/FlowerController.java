package ru.smartfrog.Flowers;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.opencv_core.IplImage;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.json.*;

public class FlowerController {
	// images to show in the view
		@FXML
		private ImageView originalImage;
		@FXML
		private ImageView transformedImage;
		@FXML
		private ImageView antitransformedImage;
		// a FXML button for performing the transformation
		@FXML
		private Button analizeButton;
		// a FXML button for performing the antitransformation
		@FXML
		private Button genVocabularyButton;
		@FXML
		private Button loadVocabularyButton;
		
		@FXML
		private Label resultLabel;
		// the main stage
		private Stage stage;
		// the JavaFX file chooser
		private FileChooser fileChooser;
		// support variables
		private Mat image;
		private List<Mat> planes;
		
		private boolean vocabularyReady = false;
		private boolean imageReady = false;
		
		ImageConverter converter = new ImageConverter();
		private ArrayList<String> pathes;
		private ArrayList<String> categories;
		private String filePath;
		private String configPath;
		private int vocabularySize;
		
		
		ClassifierBuilder cbuilder;
		
		/**
		 * Init the needed variables
		 * @throws JSONException 
		 * @throws FileNotFoundException 
		 */
		protected void init() throws JSONException, FileNotFoundException
		{
			this.fileChooser = new FileChooser();
			this.image = new Mat();
			this.planes = new ArrayList<>();
			loadConfig();
			this.cbuilder = new ClassifierBuilder(vocabularySize,configPath, pathes);
		}
		
		protected void loadConfig() throws JSONException, FileNotFoundException
		{
			categories = new ArrayList<>();
			pathes = new ArrayList<>();
			JSONTokener tokenizer = new JSONTokener(new FileReader(new File("C:/WORK/study/images_workspace/Flowers/resources/config.json")));
			JSONObject obj = (JSONObject) tokenizer.nextValue();
			filePath = obj.getString("file_path");
			configPath = obj.getString("config_path");
			vocabularySize = obj.getInt("vocabulary_size");
			JSONArray arr = obj.getJSONArray("categories");
			for (int i = 0; i < arr.length(); i++)
			{
			    String name = arr.getJSONObject(i).getString("name");
			    categories.add(name);
			    String path = arr.getJSONObject(i).getString("path");
			    pathes.add(path);
			}
		}
		
		/**
		 * Load an image from disk
		 */
		@FXML
		protected void loadImage()
		{
			File file = new File(filePath);
			this.fileChooser.setInitialDirectory(file);
			// show the open dialog window
			file = this.fileChooser.showOpenDialog(this.stage);
			if (file != null)
			{
				// read the image in gray scale
				this.image = imread(file.getAbsolutePath());
				// show the image
				this.originalImage.setImage(converter.convert(this.image));
				// set a fixed width
				this.originalImage.setFitWidth(250);
				// preserve image ratio
				this.originalImage.setPreserveRatio(true);
				// update the UI
				imageReady = true;
				if (vocabularyReady)
					this.analizeButton.setDisable(false);
				
				resultLabel.setText("Картинка готова к анализу");				
			}
		}

		@FXML
		protected void analizeImage() throws Exception
		{
			int cat = (int) cbuilder.recognize(this.image);
			System.out.println(cat);
			resultLabel.setText(categories.get(cat));
			//Mat padded = doBackgroundRemoval(this.image);
			//this.transformedImage.setImage(converter.convert(padded));
		}

		@FXML
		protected void loadVocabulary() throws Exception
		{
			cbuilder.load();
			vocabularyReady = true;
			genVocabularyButton.setDisable(true);
			loadVocabularyButton.setDisable(true);
			if (imageReady)
				analizeButton.setDisable(false);
		}
		
		/**
		 * The action triggered by pushing the button for apply the inverse dft to
		 * the loaded image
		 * @throws Exception 
		 */
		@FXML
		protected void genVocabulary() throws Exception
		{
			cbuilder.genVocabulary();
			cbuilder.reverseMap();
			vocabularyReady = true;
			genVocabularyButton.setDisable(true);
			loadVocabularyButton.setDisable(true);
			if (imageReady)
				analizeButton.setDisable(false);
		}
		
		/**
		 * Set the current stage (needed for the FileChooser modal window)
		 * 
		 * @param stage
		 *            the stage
		 */
		public void setStage(Stage stage)
		{
			this.stage = stage;
		}
		

}
