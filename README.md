# Daisy's Jetson Code
This repository contains the primary set of code necessary for Daisy to identify individuals, issue movement commands, receive Alexa prompts, and serve a web-based Dashboard that provides visualization of data that daisy collects.

## Requirements
System Requirements:
* Python3 (Scripts are all targeted for Python3)
* OpenCV 3.4.1 for Jetson - See [buildOpenCVTX2](https://github.com/jetsonhacks/buildOpenCVTX2). You will want to change the version of OpenCV found in the build script to version 3.4.1 (or later, this code base was executed using OpenCV 3.4.1). Check the [OpenCV Repository](https://github.com/opencv/opencv) to see what the latest version of OpenCV is. IMPORTANT: Make sure to also modify the build script to include the OPENCV_EXTRA_MODULES_PATH variable. See the next bulletpoint to see how to include into the build.
* OpenCV-contrib 3.4.1 - See [OpenCV Extra Modules](https://github.com/opencv/opencv_contrib). These modules contain the trackers that are used for track individuals after facial recognition fails.
* libfreenect2 - See [installLibfreenect2](https://github.com/jetsonhacks/installLibfreenect2). This consists of the libraries and drivers necessary to use the kinect2 with python and linux.

Python Libraries (Use pip to install the following):
* face_recognition - Python library for facial recognition. See: [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition). See daisy_eye.py.
* pySerial - Used for serial communication with arduino. See daisy_spine.py.
* pylibfreenect2 - See (pylibfreenect2)[https://github.com/r9y9/pylibfreenect2]. These are the python bindings used to interface with the drivers of libfreenect2. Very important!
* numpy - Used for faster matrix operations
* flask - Used to host the Dashboard for Daisy
* flask_httpauth - Used for basic authentication for the Dashboard page of Daisy
* matplotlib - Used to generate visualizations for Dashboard page of Daisy
* pymongo - Used to interface with mongodb service to grab collected data/information.

Hardware Requirements:
* Arduino
* Microsoft Kinect2
* Jetson TX2

## Usage
1) First install all the dependencies listed above.
2) Connect an Arduino and Microsoft Kinect2 to the Jetson TX2
3) Make sure the port name for the Arduino is properly setup and the Arduino is loaded with the code found in the following repository: [DaisyEmbedded](https://github.com/J-Pai/DaisyEmbedded/blob/master/DaisyEmbedded.ino).
4) Create a faces directory inside of the repository and add images of faces that you want the code to identify to in the folder. Make sure to also update the names and file locations in the python script daisy_brain.py. After that, open up 5 terminals. The following commands can by ran either using ./ or python3.
5) In terminal 1 run ngrok-arm http 8080. This is to open up a link/connection for daisy_server.py which is a flask server hosting a dashboard for the robot.
6) In terminal 2 run daisy_neuron.py. This setups a object manager for interprocess communication(python concept see [Multiprocessing Managers](https://docs.python.org/2/library/multiprocessing.html#managers)).
7) In terminal 3 run execute_brain.sh. This script first sets the DISPLAY environment variable to the primary display output for the jetson and then executes daisy_brain.py. The reason for this is because if you are running daisy_brain.py over SSH, the application will attempt to use the GPU/Rendering engine for the monitor of the host machine (the current machine). Since the Kinec2 processing requires OpenGL, this will result in an error since SSH cannot really handle graphics based instructions. Setting the DISPLAY environment variable forces the application to use the Jetson's GPU.
8) In terminal 4 run daisy_server.py. By default the server will be hosted at localhost:8080. With ngrok-arm running, you should be able to access the webpage at whatever the ngrok url is.
9) If you have access to an Alexa interface, check out the the Alexa repository to see the Alexa implementation. [Alexa Repo](https://github.com/tewodros88/DaisyAlexa).

## Configuration
For our project, the data is stored and pulled from a service called mLab. See [mLab](https://mlab.com/). It allows us to obtain a MongoDB instance in the cloud. For our purposes the free tier is more than adaquet for our usage.

## References/Links
[ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)

[Object Tracking using OpenCV](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/)

[GOTURN Documentation](https://github.com/davheld/GOTURN)

[GOTURN Pretrained Model](https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking)

[Trackers in OpenCV-contrib 3.4.1](https://docs.opencv.org/trunk/d0/d0a/classcv_1_1Tracker.html)

Directions: Download all files. Combine zip parts into 1 zip file. Unzip model file. Place files in the directory that contains holds this repository.
