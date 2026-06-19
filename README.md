# Real-Time Human Detection and Counting System

This repository contains a real-time computer vision and deep learning application designed to detect and count people from live camera streams. The project leverages the power of the YOLO (You Only Look Once) architecture integrated with OpenCV for seamless frame processing.

## 🚀 Key Features
* **Real-Time Processing:** High-speed frame handling with low latency for instant feedback.
* **Accurate Detection:** Leverages pre-trained convolutional neural networks to identify human targets under varying conditions.
* **Dynamic Counter:** Tracks and displays the total number of people currently visible in the camera frame.
* **Visual Bounding Boxes:** Draws precise bounding boxes and class labels around detected targets dynamically.

## 🛠️ Project Structure
* **`human_counter.py`:** The main executable script that handles the video capture loop, model inference, bounding box rendering, and UI overlays.
* **`yolo26n.pt`:** The trained YOLO network weights optimized for efficient object detection performance.

## 📊 System Overview
The model processes every input frame through a single pass of the neural network, predicting bounding box coordinates and class probabilities simultaneously. When class ID `0` (Person) is detected, the counter increments, and a visual overlay is generated on the output stream.

### Advantages
* High frame rate (FPS) performance with minimal inference delay.
* Easy deployment utilizing the Ultralytics API ecosystem.
* GPU acceleration compatibility for high-performance computing.

### Limitations
* Potential occlusion errors in highly crowded environments where people overlap.
* Reduced detection confidence in extremely low-light scenarios.

## 💻 Tech Stack
* Python 3
* Ultralytics YOLO API
* OpenCV (Computer Vision Framework)
