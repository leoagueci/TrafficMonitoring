#ifndef SRC_CLASSIFIER_HPP_
#define SRC_CLASSIFIER_HPP_

#define CPU_ONLY

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

using namespace std;
using namespace cv;
using namespace caffe;

enum Classes
	{car, person, bus, truck, van, motorbike, bicycle, tram, background, other};

Classes strToEnum(string s);

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {

	private:
		shared_ptr<caffe::Net<float> > 	net_;				//The imported net
		cv::Size 						input_geometry_;	//Input layer width and height
		int 							num_channels_;		//Input layer channels
		cv::Mat 						mean_;				//Mean image
		std::vector<string> 			labels_;			//Class labels
		int 							batch_size_;		//Batch size

	public:
		Classifier(const string& model_file,
					const string& trained_file,
					const string& mean_file,
					const string& label_file,
					const bool use_GPU,
					const int batch_size);

		std::vector< vector<Prediction> > ClassifyBatch(const vector< cv::Mat > imgs, int num_classes, int N);

		void setBatchSize (int batch_size);

		static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);

		static std::vector<int> Argmax(const std::vector<float>& v, int N);

	private:
		void SetMean(const string& mean_file);

		std::vector< float > PredictBatch(const vector< cv::Mat > imgs) ;

		void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);

		void PreprocessBatch(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
};

#endif /* SRC_CLASSIFIER_HPP_ */
