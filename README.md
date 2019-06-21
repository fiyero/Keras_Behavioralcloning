# Self driving car behavioral cloning simulation
## https://medium.com/@patrickhk/self-driving-car-behavioral-cloning-stimulation-320bfd642aa3
I have enrolled the Udacity Self Driving Car Engineer Nanodegree program and we have a chance to build a behavioral cloning simulation project. A self driving car should be able to steer the wheel and turn direction along the road. This behavioral cloning simulation project let us build a neural network and use images captured by the self driving car as input to predict the steering angle as output.<br/>

https://youtu.be/Yf4_CIUN6eU

### Capture data by using udacity car simulator
![p1](https://cdn-images-1.medium.com/max/800/1*ylXEc8iiewbHKx-uv7GUUw.png)<br/>
We need to manually control the direction and speed of the car to complete the lap under the training mode. The simulator will record several information such as speed, steering angle, brake and image from the point of view of the three cameras on the car at every instance. All these information are exported as the driving log in csv file.

Our driving style directly affect the data collected and this will affect our model performance. Garbage in garbage out, if we drive like crazy in the simulator the self driving car will drive like drunk as well.<br/>
### Extraction of data from driving log to create dataset
The driving log contains path of the images and other related data. Our goal is to use image to train the model to predict steering angle therefore the self driving car know how and when to turn. Therefore we need to build x_dataset with image tensor and y_dataset as ground truth.

Pandas and np.concatenate, sklearn , generator..etc can handle this. There are 71241 imgs for the training set and 17811 imgs for the validation set. I didn’t prepare test set because the BEST way to evaluate the performance is to inference the result in the simulator, to see how the self driving car drive.<br/>
![p2](https://cdn-images-1.medium.com/max/800/1*aF3GePEYX5vNwcHwjaaI1g.png)<br/>
Like all other neural network with images we always need to do pre-process. For example, cropping, augmentation and normalization are done in my project.<br/>
### Augmentation
With normal image classification task we can apply augmentation easily within Keras because the label won’t change. However this behavioral cloning stimulation is not a classification task and if we apply augmentation to our input data, the cor-respective output will change as well. Therefore I find it easiest to use np.fliplr to flip the image and assign steering angle*-1 to get augment_x and augment_y<br/>
![p3](https://cdn-images-1.medium.com/max/800/1*1avEctDVer914YE4TD_Meg.png)<br/>

### Cropping
Not the whole image is useful for our self driving car. Usually the top half of the image are sky, tree..etc and won’t provide information helpful to our driving. Therefore they can be crop out. Since the inference images returned by the simulator are fixed at dimension 160*320*3, which means we cannot just pre-process and modify the input shape of the model otherwise will have shape incompatible error. We have to apply cropping by cropping layer INSIDE the model. Use keras.layers.Cropping2D<br/>
![p4](https://cdn-images-1.medium.com/max/800/1*VdMmeyxDgAjO2ZJUkf916w.png)<br/>
### Normalizing
Same situation to cropping. We apply normalizing through the lambda layer in keras. Use keras.layers.lambda to /255. and -0.5<br/>
### Build the model
Not necessary to build one from scratch. I use Keras Resnet50 as the feature extractor. Then add several fully connected layer with dropout layer.<br/>

![p5](https://cdn-images-1.medium.com/max/800/1*Kg8pZqKyj-dtsdLr0nrJnA.png)<br/>
### Hyperparameters and training
As usual I use Adam, learning rate ranging from 0.001 to 0.0005, batch_size=64. I lose count on the total epochs I have trained because I keep adding new training data and train the model on previous saved weight. Fine tuning the model is just like a trial and error process. You have to get your hands dirty, try playing around model architecture, hyperparameters and dataset.<br/>
### Result
I have trained my model with 5 different versions(model1 to 5). Each is trained on top of previous version with larger dataset / slightly smaller learning. I uploaded the result on youtube.<br/>

Best version: model4 (not the longest training one)<br/>
https://youtu.be/Yf4_CIUN6eU <br/>

Human (left) Versus Model 4(right): <br/>
https://youtu.be/pR1eIWUMNr8<br/>


Failed attempt of model 1 in 30mph<br/>
https://youtu.be/izdO60lLueM<br/>

I am totally satisfied with the result of the self driving car in model 4. You can see the comparison video that some turns are not easy even I cannot control and steer well. I will say the self driving car have better subtle control of steering than me because computer can be very qualitative.

I am so excited about self driving car!<br/>

### Remark and update:
There is always a trade off between accuracy and resources. A deeper network can better map the relationship between the input to output but at the same time cost much more training time and GPU resoruces.<br/>

Actually it is not necessary to apply ResNet be the convolution base to extract features. Since this task is not for image identification and the requirement for feature extraction is not very demanding. We don’t require the model to recognize very complex object therefore can apply a few(such as 4–5 2D convolution layer) then the fully connective layers. This can greatly save a lot of training time. In real life the self driving car also LIDAR, RADA..etc for perception.<br/>

I discover I made a variable typo in and it leads to 1/6 of the dataset are noise in version 1. After correcting the data, my version 2 (very simple network instead of ResNet) can provide better performance but only 1/5 of the training time.<br/>

Model 6 (version2) in 30 mph:<br/>
https://youtu.be/KD2AZnKRZUg<br/>
-------------------------------------------------------------------------------------------------------------------------------------
### More about me
[[:pencil:My Medium]](https://medium.com/@patrickhk)<br/>
[[:house_with_garden:My Website]](https://www.fiyeroleung.com/)<br/>
[[:space_invader:	My Github]](https://github.com/fiyero)<br/>
