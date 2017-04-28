#ifndef SRC_TRACKING_HPP_
#define SRC_TRACKING_HPP_

#include "../include/Classifier.hpp"

extern vector<Mat> 					boundingBoxes;		//Objects found
extern vector<Rect> 				recs;				//Rectangles of the objects
extern vector<Point2f> 				massCenters;    	//Centers of mass of the objects
extern vector< vector<Prediction> > predictions;		//Predictions assigned to the objects
extern Scalar recColors[8];

class Track{

	private:

		float 	x, y;					// Position of the centroid
		Scalar	avgColor;				// Mean color of the image
		String 	label;					// Label assigned through classification
		float   prob;					// Classification probability
		Rect	rec;					// Contains the rect of the image
		Mat 	bndBox;					// Contains the image
		bool 	assigned;				// Assigned in the current frame
		int 	framesWithoutUpdate;	// Keeps count of the number of consecutive frames, where it is remained unassigned.
		 	 	 	 	 	 	 		// If the count exceeds a specified threshold, I assume that the object left the field of view and the track will be removed.
		int 	lifeTime;				// Keeps count of the number of frames since it was created
										// If the count exceeds a specified threshold, the track becomes old and it will be removed (avoid wrong classifications to last too much)
	public:
		static vector<Track> tracks;	// vector containing all the tracks

		Track(float x, float y, Rect rec, Mat bndBox);

		static Point2f computeMassCenter(vector<Point> contours);

		static void updateTracks(float frameDiagonal, float distanceTH, float avgColorTH);

		static void createNewTracks();

		static void classifyTracks(Classifier classifier, int classes);

		static void deleteUselessTracks(int noUpdateTH, int lifetimeTH);

		static void drawTracks(Mat frame, float probTH);

	private:
		void noUpdateThisFrame();

		void gotUpdate();

		void updateTrack(float x, float y, Rect rec, Mat bndBox);

		bool checkMeanColor(Mat bndBox, float avgColorTH);

		bool massCenterAssignment(float frameDiagonal, float distanceTH, float avgColorTH);

		int findIndexMinElement(double v [], int size);

		float computeDistanceBetweenObjects(Point2f point, float frameDiagonal);
};

#endif /* SRC_TRACKING_HPP_ */
