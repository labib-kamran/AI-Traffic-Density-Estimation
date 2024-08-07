
# Vehicle Detection and Traffic Analysis using YOLOv8

This repository contains code for vehicle detection and traffic density analysis using the YOLOv8 object detection model. The project involves fine-tuning a pre-trained YOLOv8 model on a custom vehicle detection dataset, performing object detection, and analyzing traffic density from video footage.



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
    ```

2. **Create a virtual environment**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries**

    ```sh
    pip install -r requirements.txt
    ```

## Dataset

1. **Link to dataset**

    [Download Dataset](https://drive.google.com/file/d/1RbTX5Jmd-WdmEbPdoUjEtpvv4ljsHfC1/view?usp=sharing)

2. **Download the dataset from the provided link and place it in the `./Vehicle_Detection_Image_Dataset` directory.**

3. **Ensure your dataset directory structure looks like this:**

    ```kotlin
    ./Vehicle_Detection_Image_Dataset
    ├── train
    │   ├── images
    │   └── labels
    ├── valid
    │   ├── images
    │   └── labels
    └── data.yaml
    ```

## Project Structure

    ```kotlin
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
    ```

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**

    ```sh
    jupyter notebook
    ```

2. **Open the `vehicle_detection_notebook.ipynb` file and run the cells step by step.**

### Fine-Tuning YOLOv8

1. **Load Pre-trained Model**

    ```python
    model = YOLO('yolov8n.pt')
    ```

2. **Data Exploration**

    Load and explore the dataset.

3. **Training the Model**

    Fine-tune the YOLOv8 model on the custom dataset.

    ```python
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
    ```

4. **Saving the Model**

    Save the trained model.

    ```python
    model.save("trained_model_yolov8n.pt")
    ```

### Traffic Density Analysis

1. **Perform Inference on Videos**

    Perform inference on a sample video and save the output.

    ```python
    best_model.predict(source='sample_video.mp4', save=True)
    ```

2. **Analyze Traffic Density**

    Analyze traffic density by counting vehicles in defined regions of interest within the video frames.

## Results

### Sample Image Detection

![Sample Image Detection 1](https://drive.google.com/uc?export=view&id=1J00wkqRBX7CsOqvMmHVCV3qU268tYmFr)
![Sample Image Detection 2](https://drive.google.com/uc?export=view&id=13r3cdcXpsR1zAOZhmTlq-_AKWg3RieyR)


### Learning Curves

![Learning Curves 1](https://drive.google.com/uc?export=view&id=1PF9OuE6Uhk4ukLLMOqRqZxPaw4LdcWF-)
![Learning Curves 2](https://drive.google.com/uc?export=view&id=1ayQMoxul3wU46S0NWJRLdXyyCK3iyt6M)
![Learning Curves 3](https://drive.google.com/uc?export=view&id=1rE5bC82NFI6ISQ9WmEMmjDNSRMS6LZtR)

### Confusion Matrix

![Confusion Matrix](https://drive.google.com/uc?export=view&id=1UHNCRJ6MCBnCUqxDbNxKJ_Rzklu6NdQj)

### Traffic Density Analysis

![Traffic Density Analysis](https://drive.google.com/uc?export=download&id=1LFsVmTqou7GVN1SlJb50qMmFwWegwWnU)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.
