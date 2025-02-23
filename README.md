This projects is developed for the KrackHack Hackathon it enables real-time traffic object detection from any live-streamed video, including YouTube live feeds. Using YOLOv8 for object detection, a Flask backend, and OpenCV for frame processing, it efficiently detects vehicles and pedestrians. The frontend provides a live dashboard to visualize detections in real time. Optimized for performance without FFmpeg, it achieves ~82% mAP and ~30 FPS, making it a scalable solution for traffic monitoring and smart city applications.
It is trained over 50k images using BDD100K dataset as of now.

YT video link (Live Traffic Detection) - https://youtu.be/j1pwkwVErj8
YT video link (pre-recorded video) - https://youtu.be/Mzxkp8zATjY

The dataset mainly consisted of cars so it gives a greater accuracy for that.

Structure:

1)"backend" -> "best.pt" which is the model path. Just download that and it should work on an traffic data for object detection.

2)"backend" -> "app.py" which is the flask backend.

3)"backend" -> "templates" -> "index.html" which contains the html interface code on which the flask backend runs.

4)"1.ipynb" contains the model training code with the accuracies.

5)"test.ipynb" contains the code that uses the best.pt for testing pre-downloaded videos on the model.

6)"data.yaml" contains all the important paths for the images and labels. Also it has the number of classes the model is trained on(in this model mainly for vehicles.)

7)"yolov8n.pt" and "yolov11n" are the nano model paths for YoloV8. I have used YoloV8n for this project.

Dataset:

Dataset initially had several files. On manually processing them, finally made 3 folders - 

1) Images - containing train(70k), test(20k), val(10k)

2) Labels - containing the json file for the train and val images.

3) Yolo Labels - YoloV8 model works on .txt labels, therefore, converted the json labels to .txt files with code and then matched these yolo_labels to the corresponding images.

eg: "0000f77c-62c2a288.txt" is matched to "0000f77c-62c2a288.jpg" and so on for all the 100k images.

But used only 50k 384x640 resolution images for training due to resource and time constraints over 2 epochs.

Watch the YT videos using the link for demonstration.




