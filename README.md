# Satark: A Disaster Detection & Mitigation System

## Overview

Satark is a Deep Learning disaster detection system designed to identify natural disasters from images and provide rapid alerts. The system leverages Deep Learning and Computer Vision techniques to classify disaster events such as floods, fires, earthquakes, landslides, and cyclones, enabling faster response and improved situational awareness.

## Features

* Multi-class disaster classification
* Real-time image-based disaster detection
*  alert generation
* Interactive web interface
* High-accuracy deep learning model
* Disaster response support system

## Tech Stack

* Python
* TensorFlow / Keras
* Vision Transformer (ViT)
* ResNet50
* OpenCV
* NumPy
* Pandas
* Flask

## Model Architecture

The project uses a hybrid deep learning approach combining:

* Vision Transformer (ViT) for capturing global image features
* ResNet50 for extracting local spatial features
* MLP Head for final disaster classification

## Dataset

The model was trained on a curated disaster image dataset containing images from multiple disaster categories:

* Flood
* Fire
* Landslide
* Earthquake
* Cyclone

## Results

* Achieved high classification accuracy across multiple disaster categories.
* Successfully identified disaster scenes from unseen test images.
* Demonstrated robust performance in real-world disaster scenarios.

## Project Workflow

1. Image Upload
2. Image Preprocessing
3. Feature Extraction using ViT and ResNet50
4. Disaster Classification
5. Alert Generation
6. Result Visualization

## Screenshots

### Home Page

![Home Page](homepage1 (1).png)

### Disaster Detection Result

![Result](images/result.png)

### Alert Generation

![Alert](images/alert.png)

### Dashboard

![Dashboard](images/dashboard.png)

## Future Improvements

* Real-time CCTV integration
* Drone-based disaster monitoring
* GIS and map integration
* SMS and Email emergency alerts
* Mobile application support

## Authors

Shubh Sharma
Kartikey Negi & 
Aayush Gupta 

## License

This project is developed for educational and research purposes.
