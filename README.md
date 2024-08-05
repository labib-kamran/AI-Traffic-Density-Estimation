# Vehicle Detection and Traffic Analysis using YOLOv8

This repository contains code for vehicle detection and traffic density analysis using the YOLOv8 object detection model. The project involves fine-tuning a pre-trained YOLOv8 model on a custom vehicle detection dataset, performing object detection, and analyzing traffic density from video footage.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the environment and install the necessary dependencies.

1. **Clone the repository**

   ```sh
   git clone https://github.com/labib-kamran/AI-Traffic-Density-Estimation.git
   cd vehicle-detection-yolov8
Create a virtual environment

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required libraries

sh
Copy code
pip install -r requirements.txt
Dataset
Download the dataset from the provided link and place it in the ./Vehicle_Detection_Image_Dataset directory.

https://drive.google.com/file/d/1FkfG8fntezgUaSphCVbE1Keh60ycre41/view?pli=1

Ensure your dataset directory structure looks like this:

kotlin
Copy code
./Vehicle_Detection_Image_Dataset
├── train
│   ├── images
│   └── labels
├── valid
│   ├── images
│   └── labels
└── data.yaml
Project Structure
kotlin
Copy code
.
├── Vehicle_Detection_Image_Dataset
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── valid
│   │   ├── images
│   │   └── labels
│   └── data.yaml
├── traffic_density_analysis.mp4
├── processed_sample_video.mp4
├── requirements.txt
├── README.md
└── vehicle_detection_notebook.ipynb
Usage
Running the Jupyter Notebook
Start Jupyter Notebook

sh
Copy code
jupyter notebook
Open the vehicle_detection_notebook.ipynb file and run the cells step by step.

Fine-Tuning YOLOv8
Load Pre-trained Model

Load a pre-trained YOLOv8 model from Ultralytics.

python
Copy code
model = YOLO('yolov8n.pt')
Data Exploration

Load and explore the dataset.

Training the Model

Fine-tune the YOLOv8 model on the custom dataset.

python
Copy code
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer='auto',
    lr0=0.0001,
    lrf=0.1,
    dropout=0.1,
    patience=50,
    seed=0
)
Saving the Model

Save the trained model.

python
Copy code
model.save("trained_model_yolov8n.pt")
Traffic Density Analysis
Perform Inference on Videos

Perform inference on a sample video and save the output.

python
Copy code
best_model.predict(source='sample_video.mp4', save=True)
Analyze Traffic Density

Analyze traffic density by counting vehicles in defined regions of interest within the video frames.

Results
Sample Image Detection




Learning Curves




Confusion Matrix



Traffic Density Analysis



Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

License
This project is licensed under the MIT License.

