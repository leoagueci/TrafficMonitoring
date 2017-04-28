#include "../include/TrafficMonitoring.hpp"

/* Check if the rectangle around the object touches the border of the frame */
bool notBorderObject(Rect rec){

	Point topLeft = rec.tl();
	Point bottomRight = rec.br();

	if(topLeft.x == 1 || topLeft.y == 1 || bottomRight.x == frameWidth - 1 || bottomRight.y == frameHeight - 1)
		return false;

	return true;
}

/* Avoid really small objects founded. This objects are difficult to label for the ground truth*/
bool checkDimension(Rect rec){
	//Size as to be greater than or equal to 40x15 or 15x40
	if((rec.width < 15 || rec.height < 15) || (rec.width < 40 && rec.height < 40))
		return false;

	return true;
}

/* Given several vector of points, found the relative objects and return the number of objects founded */
int findObjects(vector<vector<Point> > contours){
	int objects = 0;

	//Clean structures
	recs.clear();
	boundingBoxes.clear();
	predictions.clear();
	massCenters.clear();

	//Scan each region found
	for (unsigned int i = 0; i < contours.size(); i++) {
		//Create the rectangle around the object
		Rect aux = boundingRect(contours[i]);
		/* Consider only rectangles with area greater than a given threshold and that are not border objects
		 * This allows to avoid classifying very small objects and partial objects
		 */
		if (checkDimension(aux) && notBorderObject(aux)){
			recs.push_back(aux);
			boundingBoxes.push_back(Mat(frame, recs.back()));
			//Compute the center of mass of the object
			massCenters.push_back(Track::computeMassCenter(contours[i]));
			objects++;
		}
	}
	return objects;
}

/* Classify objects when the tracking mode is off */
void classifyObjects(Classifier classifier, float probTH){
	//If there is at least one founded object
	if(boundingBoxes.size() > 0){

		//set batch size on-fly and classify
		classifier.setBatchSize(boundingBoxes.size());
		predictions = classifier.ClassifyBatch(boundingBoxes, NUM_CLASSES, 1);
	}
	
	String guess;
	float prob;
	int baseline;
	for(unsigned int i = 0; i < recs.size(); i++){
		guess = predictions.at(i).at(0).first;
		prob  = predictions.at(i).at(0).second;
		if(prob >= probTH && strToEnum(guess) != background){
			Size textSize = getTextSize(guess, FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
			rectangle(frame, recs.at(i).tl() - Point(1, 1), recs.at(i).tl() + Point(textSize.width, -(textSize.height + 6)), recColors[strToEnum(guess)], CV_FILLED);
			putText(frame, guess, Point(recs.at(i).x, recs.at(i).y - 4), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,0,0), 1);
			rectangle(frame, recs.at(i).br(), recs.at(i).tl(), recColors[strToEnum(guess)], 2);
		}
	}
}

/* Classify objects when tracking mode is on */
void classifyObjectsWithTracking(Classifier classifier, float probTH, float distanceTH, float avgColorTH, int noUpdateTH, int lifetimeTH){
	//Updates tracks with possible matches and removes the object associated to the tracks
	Track::updateTracks(frameDiagonal, distanceTH, avgColorTH);

	//Create new tracks with the unassigned objects
	Track::createNewTracks();

	//Classify tracks not yet classified
	Track::classifyTracks(classifier, NUM_CLASSES);

	//Remove useless tracks
	Track::deleteUselessTracks(noUpdateTH, lifetimeTH);

	//Draw all the assigned tracks
	Track::drawTracks(frame, probTH);

}

void analyzeVideoStream(string netPath, string videoPath, bool classification, bool tracking, int maxObjs, float probTH, float distanceTH, float avgColorTH, int noUpdateTH, int lifetimeTH, int sf){
	Ptr<BackgroundSubtractorMOG2> mog2;	//MOG2 Background Subtraction method
	Mat mask;  							//Foreground mask
	Mat blur;							//Frame with some noise removed
	VideoCapture input;					//Input stream
	int keyboard; 						//Input from keyboard

	/* Load Caffe net, mean image and labels */
	Classifier classifier(netPath + "/deploy.prototxt", netPath + "/deploy.caffemodel", netPath + "/mean.binaryproto", netPath + "/labels.txt", false, 1);

	//Open the video stream
	if(videoPath.compare("") == 0)
		input.open(CV_CAP_ANY); //Acquire from the default camera
	else
		input.open(videoPath); //Acquire from a video
	if (!input.isOpened()) {
		cerr << "ERROR! Unable to open video stream\n";
		exit(EXIT_FAILURE);
	}

	//Create the background subtractor
	mog2 = createBackgroundSubtractorMOG2();
	//No shadow detection
	mog2->setDetectShadows(false);

	//Read until ESC, q is pressed
	while(((char) keyboard != 'q' && (char) keyboard != 27)){
		//If stream acquired from a video, read until the video end
		if(videoPath.compare("") != 0)
			if(input.get(CV_CAP_PROP_POS_FRAMES)  >= input.get(CV_CAP_PROP_FRAME_COUNT))
				break;

		//Read the current frame (BGR color-space)
		if (!input.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}

		//Resize the frame (Maintain 16:9 aspect ratio)
		frameWidth = 16 * sf;
		frameHeight = 9 * sf;
		resize(frame, frame, Size(frameWidth, frameHeight), 0, 0, INTER_LINEAR);
		//Set the diagonal of the frame
		frameDiagonal = sqrt(pow(frameWidth, 2) + pow(frameHeight, 2));

		//Blur applied to eliminate some noise
		GaussianBlur(frame, blur, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0);

		//Mixture of Gaussian subtractor applied to the current frame
		mog2->apply(blur, mask);

		//Apply some transformations to the foreground mask
		dilate(mask, mask,
				getStructuringElement(MORPH_DILATE,
						Size(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)));
		erode(mask, mask,
				getStructuringElement(MORPH_ERODE,
						Size(ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE)));

		//Find the contours of the moving object detected
		Mat hierarchy;
		vector<vector<Point> > contours;
		findContours(mask, contours, hierarchy, RETR_EXTERNAL,
					CHAIN_APPROX_SIMPLE);

		//Find the rectangle around the object
		int objects = 0;
		objects = findObjects(contours);

		//Classify and draw only if the number of objects found is less than a given threshold
		//Avoid to perform operations when, because of background changes, the subtractor finds a lot of moving objects
		if(objects <= maxObjs){
			//Classification mode on
			if(classification){
				//Tracking mode on
				if(tracking){
					classifyObjectsWithTracking(classifier, probTH, distanceTH, avgColorTH, noUpdateTH, lifetimeTH);
				}
				else{//Tracking mode off
					classifyObjects(classifier, probTH);
				}
			}
			else{//Classification mode off
				//Draw only the rectangles without classification
				for(unsigned int i = 0; i < recs.size(); i++){
					rectangle(frame, recs.at(i).br(), recs.at(i).tl(), recColors[0], 2);
				}
			}
		}

		//Create a window to show the video
		namedWindow("Real time classification", WINDOW_AUTOSIZE);
		moveWindow("Real time classification", 100, 50);
		imshow("Real time classification", frame);

		//Acquire input from the keyboard
		keyboard = waitKey(1);
	}

	//Release the input stream
	input.release();
	//Destroy the video window
	destroyAllWindows();
	//Release the background subtractor
	mog2.release();
}

int main(int argc, char **argv){

	//Parameters
	string	net_path,
			video_path;
	int		sf,
			maxObjs,
			noUpdateTH,
			lifetimeTH;
	float 	probTH,
		  	distanceTH,
			avgColorTH;
	bool 	classification,
		 	tracking;

	// Declare a group of options that will be
	// allowed only on command line
	namespace po = boost::program_options;
	po::options_description cmdline_options("Generic options");
	cmdline_options.add_options()
	("help,h", "Print help message")
	("classification,c", "Enable classification mode")
	("tracking,t", "Enable tracking mode")
	("video,v", po::value<string>(&video_path)->default_value(""), "Video path, if not specified the video is acquired from the device camera");

	// Declare a group of options that will be
	// allowed only in config file
	po::options_description config_file_options("Configuration parameters");
	config_file_options.add_options()
	("net_path", po::value<string>(&net_path)->required(), "Specify the path of the CNN")
	("scaling_factor", po::value<int>(&sf)->required(), "Set the scaling factor of the frame, aspect ratio 16:9")
	("maxObjs", po::value<int>(&maxObjs)->required(), "Set maximum number of objects per frame")
	("probTH", po::value<float>(&probTH)->required(), "Set probability threshold")
	("distanceTH", po::value<float>(&distanceTH)->required(), "Set distance threshold")
	("avgColorTH", po::value<float>(&avgColorTH)->required(), "Set average color threshold")
	("noUpdateTH", po::value<int>(&noUpdateTH)->required(), "Set no update threshold")
	("lifetimeTH", po::value<int>(&lifetimeTH)->required(), "Set lifetime threshold");

	po::variables_map vm;
	try{
		po::store(po::parse_command_line(argc, argv, cmdline_options),vm);
		po::store(po::parse_config_file<char>("Config.txt", config_file_options),vm);
		po::notify(vm);

		// --help option
		if (vm.count("help")){
			cout<< cmdline_options << endl << config_file_options << endl;
			return EXIT_SUCCESS;
		}

		// --classification option
		if (vm.count("classification"))
			classification = true;
		else
			classification = false;

		// --tracking option
		if (vm.count("tracking"))
			tracking = true;
		else
			tracking = false;
	}
	catch(po::error& e){
		cerr<< "ERROR: "<< e.what()<< endl;
		cerr<< cmdline_options << endl << config_file_options << endl;
		return EXIT_FAILURE;
	}

	//Analyze the video stream with the specified parameters
	analyzeVideoStream(net_path,
						video_path,
							classification,
									tracking,
										maxObjs,
											probTH,
												distanceTH,
													avgColorTH,
														noUpdateTH,
															lifetimeTH,
																	sf);
}
