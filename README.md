# Objection-detection
## 🚗 Car Plate Detection using YOLOv8 & PaddleOCR
This project focuses on detecting and recognizing vehicle license plates from images and videos using a powerful combination of YOLOv8 (for object detection) and PaddleOCR (for optical character recognition). It accurately identifies license plates and extracts readable text, making it ideal for applications like surveillance, smart parking, and automated vehicle monitoring.

## 🔍 Features
YOLOv8-based License Plate Detection
Detects the precise location of license plates in both images and videos with high accuracy.

# PaddleOCR-based Text Recognition
Extracts and reads text from detected license plates using PaddleOCR's multilingual and robust OCR engine.

# Image and Video Support
Processes single images or full-length videos frame by frame, with real-time visualization.

# Duplicate Filtering
Eliminates repeated or slightly different OCR results to ensure clean, consistent output.

# Streamlit Web App
A user-friendly interface for uploading and analyzing media files, displaying results and OCR tables instantly.

## 🛠 Tech Stack
YOLOv8 – for fast and accurate license plate detection

PaddleOCR – for extracting readable text from plates

OpenCV – for image/video processing

Streamlit – for building the interactive web interface

NumPy / pandas – for handling data and organizing OCR results

## 🚀 Use Cases
Smart traffic systems

Toll booth automation

Parking management

Vehicle tracking for law enforcement

Entry/exit monitoring in gated communities

## 🖼 Example Output
Input: Image or video containing vehicles

Output:

Detected license plate regions

Recognized text (displayed in table format)

Annotated image/video preview



