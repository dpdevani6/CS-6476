---
title: Home
layout: home
---

# CS 6476 Group 60 Project Update

## Introduction
We are investigating the problem of how of accurately ball actions from a video feed of a soccer game using state-of-the-art computer vision methods. Examples of ball actions include passes, drives, crosses and many others.

Being able to accurately identify such ball actions can be extremely useful. For example, if one does not have access to a video stream of the game due to slow internet access, they can still follow an automated live commentary feed to follow the important plays in a game. In addition, such information can augment a live broadcast of a soccer game by automating the process of keeping a camera's focus on not only where the ball is, but the entire region of interest based on the ball action currently being performed.

Such a method can also be extrapolated beyond the arena of sports to efficiently find instances of any action in video.

## Related Works

### Human Action Recognition (Action Classifier)

Human Action Recognition (HAR) involves the identification and classification of human actions in videos. Although originally focused on everyday activities, it has been increasingly applied to sports.

Of the many models and methods used for HAR, newer models have focused on deep learning models such as Convolutional Neural Networks (CNN). Specifically, a variation of CNN called 3-dimensional CNN has shown promise in adding spatio-temporal information of video action behaviors by preserving temporal information in all different video frames and generating a single output frame. It does so using 3D pooling which produces one output pixel for the same color performing along adjacent video frames [2].

Another method used alongside deep learning in HAR is Support Vector Machine (SVM) which is extremely fast and performant. In [3], SVM was used in addition to a method called Modified Motion History Images to keep historical information from the video and produce vectors from that information. It was found that this method improved recognition performance with similar computational resources to other leading methods. Despite these promising results, the application of historical data may not be applicable to soccer action detection due to soccer motion not occurring in a stationary location within a frame.

### Transformer Network:

The Action Transformer Network is a novel approach to recognizing and localizing human actions in videos. Through attention mechanisms, it is able to focus on specific body regions such as hands and faces which enhances its ability to detect actions on the human body. The transformers in this network use encoder-decoder structures that allow input information to simultaneously be processed with a self-attention mechanism that allows for long-term dependencies within the input data to be captured [4]. This method may be worth exploring for our use case as transformers have shown promise compared to other methods, but they have not yet been fully explored in sports video applications. 

### Action Detection (Temporal Location)

The aforementioned HAR can only occur once the action in a video is first detected. In [5], action spotting is a problem that is discussed which entails localizing action instances  in a video. To address it, a Recurrent Neural Network was used to spot where an action occurred temporally. This resolves the issue of temporal localization of action instances, but, in the application of soccer, multiple actions may be occurring at once which presents the need for a ball spotter to be utilized in order to make sure only actions near the ball are being recorded.

Expanding on temporal action spotting, [6] found that a model called ActionSpotter was able to detect actions temporally with 65% accuracy and skipped 23% of video frames. The model utilized reinforcement learning with a neural network that had underlying programs for frame-to-frame movement and spot action frame probability predictions to extract one frame for each action.


### Sources
[1] K. Host and M. Ivašić-Kos, “An overview of human action recognition in sports based on Computer Vision,” Heliyon, vol. 8, no. 6, Jun. 2022. doi:10.1016/j.heliyon.2022.e09633 
[2] M. Al-Faris, J. Chiverton, D. Ndzi, and A. I. Ahmed, “A review on computer vision-based methods for human action recognition,” Journal of Imaging, vol. 6, no. 6, p. 46, Jun. 2020. doi:10.3390/jimaging6060046
[3] H. Meng, N. Pears, and C. Bailey, “A human action recognition system for embedded computer vision application,” 2007 IEEE Conference on Computer Vision and Pattern Recognition, Jun. 2007. doi:10.1109/cvpr.2007.383420
[4] R. Girdhar, J. Joao Carreira, C. Doersch, and A. Zisserman, “Video action transformer network,” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2019. doi:10.1109/cvpr.2019.00033 
[5] H. Alwassel, F. Caba Heilbron, and B. Ghanem, “Action search: Spotting actions in videos and its application to temporal action localization,” Computer Vision – ECCV 2018, pp. 253–269, 2018. doi:10.1007/978-3-030-01240-3_16
[6] G. Vaudaux-Ruth, A. Chan-Hon-Tong, and C. Achard, “ActionSpotter: Deep Reinforcement Learning Framework for temporal action spotting in videos,” 2020 25th International Conference on Pattern Recognition (ICPR), Jan. 2021. doi:10.1109/icpr48806.2021.9413153 

## Methods/Approach

### Event CNN Model

The method we attempted is using a CNN on the camera live feed to classify each frame and whether a ball action has occurred or not. Although this experiment is very simple, I believe we can obtain decent results based on very distinct classes such as free kicks, goals, and outs. We hope to be able to run our model on individual frames and select a certain threshold as our positive class, indicating whether an event occurred at all, and the max being the specific event that has occurred.

We began preprocessing our data by creating a separate dataframe with path to images and the corresponding soccer event action for that frame in a separate folder. We then converted all the event names like “PASS”, “GOAL”, … to integer categoricals and then into one hot encodings. Next, we shuffled our dataset and then created an image data loader which can lazily load the image data as it is needed in the training loop. Before passing the images into the CNN, we pass the images into the YOLO model to output a localized frame. We then ran repeated epochs to train the data splitting 80% of the data for training and 20% for validation for each epoch.

### YOLO Model
One issue we ran into with the live camera feed is that we’re currently using the full broadcast image in the CNN model. This can be confusing for the CNN model, meaning we need a more localized image of the frame (without the crowd). An approach we considered was custom training a YOLOv5 (You Only Look Once) model, a well-known object detection model, on annotated game film to identify the ball in many frames. After we trained this model, we fed it video data and received an annotated video. The model is trained to find “sports-ball”. Our goal was to increase this bounding box to give our model a more localized frame to train on. This trained model works on both image and video data.


## Experiemnts/Results

### Event CNN Model

Above shows the final training loss, training accuracy, validation loss and validation accuracy of our event CNN classification model.

Although our accuracy is low above, we have struggled to be able to properly train our model at the speed we expect. Although our dataset is not too significantly large, each epoch required approximately 1 hour and 30 minutes to complete. Due to the size of our images, we estimate that holding all of the event images in memory would be ~2.43 GB. This usually would not be a problem in our local machines but we ran into issues holding the images in memory on the PACE cluster we were using. In addition, we believe that Jupyter running on the machine was causing threading problems due to the lack of memory available and crashing the kernel.

Our solution to this problem was to make use of image batching to reduce the memory overhead of the images. We selected a batch size of 32 which although allows for quicker I/O travel times, definitely reduced the speed of each epoch due to the number of visits to I/O we completed. In addition, we are hoping to make use of a larger PACE instance so we have more memory to store our images. Our last idea is to take our code to a python script rather than in a Jupyter notebook to eliminate the threading problem we ran into. Thanks to these speedups, we hope to train our model more and obtain a better accuracy value.

One problem with this model's current approach is that it does not have a label for no event occurring. This means we cannot feed a stream of images and expect to get accurate results. Although we pointed out the idea of using a threshold to judge whether an event is occurring or not, we plan to take random frames from the video feed where no events are occurring adding an additional label called “no event” so our model can tell us if there is something occurring or not occurring.

### YOLO Model

This model was trained with only around 300 images and did a pretty good job holding up. When the ball was against the grass, the model was very well not only detecting the ball but did so with high confidence. However, when the ball was in the air (3rd photo), it was much harder for the ball to even detect the ball. A further complication was ball-like objects getting detected. In the fourth image, you can see the channel’s logo being detected as sports-balls. However, this is an easy fix since we can filter to only look at frames where at most 2 balls were detected. Further, this model was extremely quick to train (~5 minutes) on Google Colab on all 300 images with 10 epochs and a batch size of 32. All in all, we were able to train a successful model based on how confident the sports-ball detections were and how few images we used. We can further improve this model with a larger training dataset that captures “tougher” angles (ie ball in the air).

## What's Next?

In addition to the aforementioned speedups and “no event” labels in our basic CNN model, we plan to begin to incorporate ball labels as part of the input to our CNN model. Because the majority of the events have to do with the soccer ball’s position, we believe that it would be valuable data to feed to our CNN for it to better understand the context of the players.

We also hope to expand the YOLO offering to the players on the field and labeling opposing players to see what is occurring. This data is helpful for our model to figure out whether a ball movement is a shot at a goalie or pass to a teammate. 

We hope to include temporal information by making use of transformers. We have identified examples of video action transformers, where the model is able to consider the sequence of frames as part of its ability to output what action is occurring in the frame. This should be helpful because understanding the trajectory of the ball is able to provide important information on whether someone has taken a shot or pass, for instance. 

We also plan to finish our implementation of infrastructure for plugging in a video file and returning model outputs for high probability timestamps. Currently, we have infrastructure for inserting a model and data file and returning a textual output for ball actions recognized by the model above a certain input threshold probability. The integration between the CNN model and this infrastructure alongside video frame outputs for detections will be worked on next.


## Contributions

Add No Event Label and Training Data to Event CNN Data - 2/2/2024 - Rohan
Speedups for Event CNN Model - 2/4/2024 - Rohan
YOLO Tracking for Players - 2/2/2024 - Sanjay
Feature Engineering with Ball Position and Event CNN Model - 2/5/2024 - Dhruv
Vision Action Transformer Model working on Test Data - 2/2/2024 - Rishi
Evaluate Event CNN Model with Player Positions and Ball Position - 2/9/2024 - Rohan & Dhruv
Baseline Test for Vision Action Transformer on Soccer Data - 2/14/2024 - Rishi & Rohan
Develop and Benchmark “Event Occuring Prefilter on Video” Model - 2/14/2024 - Dhruv & Sanjay
Prepare Visualization and Diagrams for Final Report - 2/19/2024 - Dhruv & Rohan & Rishi & Sanjay
