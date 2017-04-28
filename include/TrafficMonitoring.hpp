#ifndef SRC_VEHICLECLASSIFICATION_HPP_
#define SRC_VEHICLECLASSIFICATION_HPP_

#include "../include/Tracking.hpp"
#include <boost/program_options.hpp>

#define NUM_CLASSES 			9				//Number of possible objects classes
#define BLUR_KERNEL_SIZE 		11				//Dimension of the blur kernel
#define ERODE_KERNEL_SIZE 		11				//Dimension of the erode kernel
#define DILATE_KERNEL_SIZE 		11				//Dimension of the dilate kernel

Mat 							frame; 			//current frame
int 							frameWidth;		//Frame width
int 							frameHeight;	//Frame height
float 							frameDiagonal;	//Diagonal of the frame
vector<Mat> 					boundingBoxes;	//Objects founded
vector<Rect> 					recs;			//Rectangles of the objects
vector<Point2f> 				massCenters;	//Centers of mass of the objects
vector< vector<Prediction> > 	predictions;	//Predictions assigned to the objects

Scalar recColors [8] = { Scalar(0, 255, 0), 	//Green
						 Scalar(203, 192, 255),	//Pink
						 Scalar(0, 0, 255),		//Red
						 Scalar(0, 153, 255),	//Orange
						 Scalar(0, 216, 255),	//Yellow
						 Scalar(255, 127, 0),	//Azure
						 Scalar(255, 0, 0),		//Blue
						 Scalar(92, 11, 227) };	//Raspberry

bool  notBorderObject(Rect rec);
bool  checkDimension(Rect rec);
int   findObjects(vector<vector<Point> > contours);
void  classifyObjects(Classifier classifier, float probTH);
void  classifyObjectsWithTracking(Classifier classifier, float probTH, float distanceTH, float avgColorTH, int noUpdateTH, int lifetimeTH);
void  analyzeVideoStream(string netPath, string videoPath, bool classification, bool tracking, int maxObjs, float probTH, float distanceTH, float avgColorTH, int noUpdateTH, int lifetimeTH, int sf);

#endif /* SRC_VEHICLECLASSIFICATION_HPP_ */
