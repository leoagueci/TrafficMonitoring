/*
 * Thanks to erogol!
 * Taken from https://gist.github.com/erogol/67e02e87f94ce9dc0c63
 */

#include "../include/Classifier.hpp"

Classes strToEnum(string s){

	if(s.compare("car") == 0)
		return car;
	if(s.compare("person") == 0)
		return person;
	if(s.compare("bus") == 0)
		return bus;
	if(s.compare("truck") == 0)
		return truck;
	if(s.compare("van") == 0)
		return van;
	if(s.compare("motorbike") == 0)
		return motorbike;
	if(s.compare("bicycle") == 0)
		return bicycle;
	if(s.compare("tram") == 0)
		return tram;
	if(s.compare("background") == 0)
		return background;

	return other;
}

/* Class constructor */
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file,
                       const bool use_GPU,
					   const int batch_size) {

	if (use_GPU)
		Caffe::set_mode(Caffe::GPU);
	else
		Caffe::set_mode(Caffe::CPU);

	/* Set batchsize */
	batch_size_ = batch_size;

	/* Load the network. */
	net_.reset(new caffe::Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

/* Change batch size on-fly */
void Classifier::setBatchSize(int batch_size){
	batch_size_ = batch_size;
}

/* Compare classification based on prob */
bool Classifier::PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
std::vector<int> Classifier::Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), Classifier::PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector< vector<Prediction> > Classifier::ClassifyBatch(const vector< cv::Mat > imgs, int num_classes, int N){
    std::vector<float> output_batch = PredictBatch(imgs);
    std::vector< std::vector<Prediction> > predictions;
    N = min<int>(num_classes, N);
    for(unsigned int j = 0; j < imgs.size(); j++){
        std::vector<float> output(output_batch.begin() + j*num_classes, output_batch.begin() + (j+1)*num_classes);
        std::vector<int> maxN = Classifier::Argmax(output, N);
        std::vector<Prediction> prediction_single;
        for (int i = 0; i < N; ++i) {
          int idx = maxN[i];
          prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
        }
        predictions.push_back(std::vector<Prediction>(prediction_single));
    }
    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

/* Forward a batch of images through the net*/
std::vector< float > Classifier::PredictBatch(const vector< cv::Mat > imgs) {
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];

	input_layer->Reshape(batch_size_, num_channels_,
						input_geometry_.height,
						input_geometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector< std::vector<cv::Mat> > input_batch;
	WrapBatchInputLayer(&input_batch);

	PreprocessBatch(imgs, &input_batch);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels()*imgs.size();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels.push_back(channel);
          input_data += width * height;
        }
		input_batch -> push_back(vector<cv::Mat>(input_channels));
	}
}

/* Apply some transformation to the batch of images */
void Classifier::PreprocessBatch(const vector<cv::Mat> imgs,
                                      std::vector< std::vector<cv::Mat> >* input_batch){
    for (unsigned int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
          cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
          cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
          sample = img;

        cv::Mat sample_resized;
        if(sample.size() != input_geometry_)
          cv::resize(sample, sample_resized, input_geometry_);
        else
          sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_ == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);
    }
}


