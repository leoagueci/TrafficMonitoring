#include "../include/Tracking.hpp"

vector<Track> Track::tracks;	//vector containing all the tracks

/* Class constructor */
Track::Track(float x, float y, Rect rec, Mat bndBox){
	//variable initialization
	this->x = x;
	this->y = y;
	this->avgColor = mean(bndBox);
	this->rec = rec;
	this->bndBox = bndBox;
	this->label = "";
	this->prob = 0;
	this->assigned = true;
	this->framesWithoutUpdate = 0;
	this->lifeTime = 1;
}

/* Update a track */
void Track::updateTrack(float x, float y, Rect rec, Mat bndBox){
	this->x = x;
	this->y = y;
	this->rec = rec;
	this->bndBox = bndBox;
	this->avgColor = mean(bndBox);
}

/* State that the track was not updated in this frame */
void Track::noUpdateThisFrame(){
	this->assigned = false;
	this->framesWithoutUpdate++;
	this->lifeTime++;
}

/* State that the track was updated in this frame */
void Track::gotUpdate(){
	this->assigned = true;
	this->framesWithoutUpdate = 0;
	this->lifeTime++;
}

/* Compare the track object with another object based on the average color */
bool Track::checkMeanColor(Mat bndBox, float avgColorTH){
	Scalar meanColor = mean(bndBox);
	float distance;

	//Compute euclidean distance between the two mean color
	distance = norm(this->avgColor, meanColor);

	//Bound distance between 0 and 1 (441.673 is the max distance in case of BGR color-space)
	distance = distance / 441.673;

	if(distance <= avgColorTH){
		return true;
	}

	return false;
}

/* Find the index of the minimum element in a vector */
int Track::findIndexMinElement(double v [], int size){
	int minIndex = size - 1;
	double min = 1;
	for(int i = 0; i < size; i++){
		if (v[i] < min){
			min = v[i];
			minIndex = i;
		}
	}
	return minIndex;
}

/* Compute the center of mass of a vector of points */
Point2f Track::computeMassCenter(vector<Point> contours){
	Moments mu;
	Point2f mc;

	//Get the moment
	mu = moments( contours, false );

	//Get mass center
	mc = Point2f(mu.m10/mu.m00 , mu.m01/mu.m00);

	return mc;//mc.x, mc.y
}

/* Compute the euclidean distance between the track object and another object
 * Note that distances are independent from the frame size as long as the aspect ratio is 16:9 */
float Track::computeDistanceBetweenObjects(Point2f point, float frameDiagonal){
	float distance = sqrt( pow(this->x - point.x, 2) + pow(this->y - point.y, 2));

	//Bound the distance between 0 and 1
	return distance/frameDiagonal;
}

/* Try to assign an object to the track based on the centers of mass distance */
bool Track::massCenterAssignment(float frameDiagonal, float distanceTH, float avgColorTH){
	int bndBoxesSize = boundingBoxes.size();
	double euclideanDistance[bndBoxesSize]; //Contains the computed distances

	//Compute all the distances between the track and all others objects
	for(int i = 0; i < bndBoxesSize; i++){
		euclideanDistance[i] = computeDistanceBetweenObjects(massCenters.at(i), frameDiagonal);
	}

	//Find the index of the minimum distance in the vector
	int indexMassCenterMin = findIndexMinElement(euclideanDistance, bndBoxesSize);

	//Check if the distance is under the specified threshold and do the same for the mean color
	if(indexMassCenterMin >= 0 && euclideanDistance[indexMassCenterMin] <= distanceTH && checkMeanColor(boundingBoxes.at(indexMassCenterMin), avgColorTH)){
		//Object assigned to the track
		updateTrack(massCenters.at(indexMassCenterMin).x, massCenters.at(indexMassCenterMin).y , recs.at(indexMassCenterMin), boundingBoxes.at(indexMassCenterMin));
		gotUpdate();
		boundingBoxes.erase(boundingBoxes.begin() + indexMassCenterMin);
		recs.erase(recs.begin() + indexMassCenterMin);
		massCenters.erase(massCenters.begin() + indexMassCenterMin);
		return true;
	}else{
		//Object not assigned to the track
		noUpdateThisFrame();
	}
	return false;
}

/* Update all the tracks trying to assign each of the to an object */
void Track::updateTracks(float frameDiagonal, float distanceTH, float avgColorTH){
	int tracksSize = tracks.size();
	for(int i = 0; i < tracksSize; i++){
		tracks.at(i).massCenterAssignment(frameDiagonal, distanceTH, avgColorTH);
	}
}

/* Create new tracks from the not assigned objects */
void Track::createNewTracks(){
	int bndBoxSize = boundingBoxes.size(); //massCenters, recs and boundingBoxes have the same size and they have the informations about an object in the same position
	for(int i = 0; i < bndBoxSize; i++){
		Track newTrack = Track(massCenters.at(i).x,massCenters.at(i).y, recs.at(i), boundingBoxes.at(i));
		tracks.push_back(newTrack);
	}
}

/* Classify the tracks not yet classified */
void Track::classifyTracks(Classifier classifier, int classes){
	vector<Track *> toClassify; //Vector of pointers to track objects to classify
	vector<Mat> 	batch; 		//Batch of objects to classify

	//Search for track to classify
	for(int i = 0; i < (int)tracks.size(); i++){
		if(tracks.at(i).label.compare("") == 0){
			toClassify.push_back(&tracks.at(i));
			batch.push_back(tracks.at(i).bndBox);
		}
	}

	//If there are objects to classify
	if(batch.size() > 0){
		//set batch size on-fly and classify
		classifier.setBatchSize(batch.size());
		predictions = classifier.ClassifyBatch(batch, classes, 1);
	}

	//Update the relative tracks
	for(int i = 0; i < (int)predictions.size(); i++){
		toClassify.at(i)->label = predictions.at(i).at(0).first;
		toClassify.at(i)->prob  = predictions.at(i).at(0).second;
	}
}

/* Delete either the tracks not updated for a certain period or too old */
void Track::deleteUselessTracks(int noUpdateTH, int lifetimeTH){
	for(unsigned int i = 0; i < tracks.size(); i++){
		if(tracks.at(i).framesWithoutUpdate > noUpdateTH || tracks.at(i).lifeTime > lifetimeTH){
			tracks.erase(tracks.begin() + i);
			i--;
		}
	}
}

/* Draw on the frame all the tracks assigned to an object*/
void Track::drawTracks(Mat frame, float probTH){
	int baseline;

	for(unsigned int i = 0; i < tracks.size(); i++){
		Track * auxTrack = &tracks.at(i);
		//Avoid to print either unassigned tracks or tracks classified as "other" or not yet classified or classified with a too low probability
		if(auxTrack->assigned && strToEnum(auxTrack->label) != background && auxTrack->prob >= probTH){
			Size textSize = getTextSize(auxTrack->label, FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
			//Draw the filled rectangle for the text
			rectangle(frame, auxTrack->rec.tl() - Point(1, 1), auxTrack->rec.tl() + Point(textSize.width, -(textSize.height + 6)), recColors[strToEnum(auxTrack->label)], CV_FILLED);
			//Draw the classification and the relative probability
			putText(frame, auxTrack->label, Point(auxTrack->rec.x, auxTrack->rec.y - 4), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,0,0), 1);
			//Draw the rectangle
			rectangle(frame, auxTrack->rec.br(), auxTrack->rec.tl(), recColors[strToEnum(auxTrack->label)], 2);
		}
	}
}

