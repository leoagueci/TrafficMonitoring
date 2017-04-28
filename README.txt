########################################
#         TRAFFIC MONITORING	         #
########################################

This file contains briefly documentation on the implemented Traffic Monitoring system.

AUTHOR

Leonardo Agueci

DESCRIPTION

This is a Moving Object Detection and Classification System: provided with a video stream from a static camera, the system is able to detect moving objects and classify them into one of the possible categories. Moreover, the system is able to track objects over time, namely it can identify whether an object was already present in a previous frame or not, in order to prevent the reclassification. Several tunable parameters are made available to the user, so that he/she can adapt the system depending on the demands of performance and the context in which it is inserted.

DEPENDENCIES

Caffe
OpenCV

BUILD INSTRUCTION
	
  make clean    delete .o and executable files 
  make 		      build TrafficMonitoring executable 

USAGE

  ./TrafficMonitoring [ -h ] [ -c ] [ -t ] [ -v <input> ]

OPTIONS

Generic options (Cdm line):
  -h [ --help ]            	Print help message
  -c [ --classification ]  	Enable classification mode
  -t [ --tracking ]        	Enable tracking mode
  -v [ --video ] arg       	Video path, if not specified the video is acquired from the device camera

Configuration parameters (Config.txt):
  --net_path arg        	  Specify the path of the CNN
  --scaling_factor arg  	  Set the scaling factor of the frame, aspect ratio 16:9
  --maxObjs arg         	  Set maximum number of objects per frame
  --probTH arg          	  Set probability threshold
  --distanceTH arg      	  Set distance threshold
  --avgColorTH arg      	  Set average color threshold
  --noUpdateTH arg      	  Set no update threshold
  --lifetimeTH arg      	  Set lifetime threshold

EXAMPLES

./TrafficMonitoring -ct

Classify moving objects, with the tracking mechanism enabled, in the video stream acquired from the default camera.

./TrafficMonitoring -c -v video.avi

Classify moving objects, with the tracking mechanism disabled, in the video stream given as input. 

########################################
#              END README              #
########################################
