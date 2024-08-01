Description
This is an AI-based traffic-density estimation project. The model analyzes traffic in a defined portion of a video and estimates whether the traffic is smooth, heavy, or low. The YOLOv8 model has been used for object detection and fine-tuned for vehicle detection. All the steps are described in the Jupyter notebook.

Link to the Dataset:
https://drive.google.com/file/d/1FkfG8fntezgUaSphCVbE1Keh60ycre41/view?usp=sharing

Instructions
Extract the dataset in the same folder where you have placed the Jupyter notebook. If you have placed the dataset in another directory, adjust the paths in the code accordingly.

Provide the correct paths:

Ensure the correct paths are provided, especially for training the model. The correct path to data.yaml is necessary.
Visualization:

The visualizations done after fine-tuning the model depend on files in the runs folder. If this folder doesn't display when you train the model locally, try using online platforms like Colab or Kaggle.
Density Estimation:

The density estimation code in this notebook is designed for the sample_video.mp4 provided in the dataset. If you want to estimate density in your video, adjust the parameters of the rectangle used for density estimation.
"# AI-Traffic-Density-Estimation" 
