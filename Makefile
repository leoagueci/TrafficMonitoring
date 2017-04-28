SRC_DIR = ./src/
INCLUDE_DIR = ./include/
CAFFE_INCLUDE = -I/home/parallels/Desktop/caffe/include #change to the correct path
CAFFE_LIB = -L/home/parallels/Desktop/caffe/build/lib -lcaffe #change to the correct path
OPENCV_LIB = `pkg-config --cflags --libs --static opencv`
LIBS = -lprotobuf -lglog -lboost_system -lz -lboost_program_options
CC = g++
CFLAGS = -g
WFLAGS = 
	
alliwanttodo: TrafficMonitoring
Classifier.o: $(SRC_DIR)Classifier.cpp $(INCLUDE_DIR)Classifier.hpp
	$(CC) -c $(CAFFE_INCLUDE) $(CFLAGS) $(WFLAGS) $<
Tracking.o: $(SRC_DIR)Tracking.cpp $(INCLUDE_DIR)Tracking.hpp $(INCLUDE_DIR)Classifier.hpp
	$(CC) -c $(CAFFE_INCLUDE) $(CFLAGS) $(WFLAGS) $<
TrafficMonitoring.o: $(SRC_DIR)TrafficMonitoring.cpp $(INCLUDE_DIR)TrafficMonitoring.hpp $(INCLUDE_DIR)Tracking.hpp
	$(CC) -c $(CAFFE_INCLUDE) $(CFLAGS) $(WFLAGS) $<
TrafficMonitoring: Classifier.o Tracking.o TrafficMonitoring.o
	$(CC) -o $@ $^ $(OPENCV_LIB) $(CAFFE_LIB) $(LIBS)
	
clean:
	rm -f TrafficMonitoring
	rm -f *.o
