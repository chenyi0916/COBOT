The project aims to create an advanced collaborative assembly system to improve low-volume production lines. The main goals were to make processes flexible to adapt to customer demands quickly, ensure safe collaboration between humans and robots, and boost overall performance. This project involved various tasks, such as developing bin picking and assembling tools using standard equipment, designing workspaces where humans and robots could work together efficiently, and creating intelligent schedules for multiple robots in low-volume production. I collaborated on the Picking and Assembling part of the project. My assigned task involves detecting specific tagboard parts, including pose estimation and orientation detection, and integrating a predictor into the ROS pipeline for picking and assembling.

I used BlenderPoc2 to generate synthetic datasets which are devided into training set and validation set. Here are some examples of generated results:
1）various background：
![image](https://github.com/chenyi0916/COBOT/blob/main/cobot_2.png)
2）workstation：
![iamge](https://github.com/chenyi0916/COBOT/blob/main/cobot_3.png)

Afterwards, I trained datasets using the Mask R-CNN algorithm based on Detectron2. The following are the predicted results.
1）real image1：
![image](https://github.com/chenyi0916/COBOT/blob/main/cobot_4.png)
2）real image2：
![iamge](https://github.com/chenyi0916/COBOT/blob/main/cobot_5.png)
3）real image3：
![image](https://github.com/chenyi0916/COBOT/blob/main/cobot_6.png)

The training loss is shown here：
![image](https://github.com/chenyi0916/COBOT/blob/main/cobot_8.png)

It can be seen that the overall accuracy of the picked trained model is not bad. However, surrounding interfering objects, such as bolts, are also mistakenly detected as target objects. We believe this is because the distribution of feature data in the training set cannot be sufficiently similar to the ground truth.

