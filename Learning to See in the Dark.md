# Learning to See in the Dark
## Authors & Contributions
Floris van Leeuwen (4676092) - Writing, Coding, Training and Results, Collection of New Data (F.N.vanleeuwen@student.tudelft.nl)

Laurens Espada (5641012) - Writing, Coding, Training and Results (L.G.PieperEspada@student.tudelft.nl)

Hovsep Touloujian (5703301) - Writing, Coding, Training and Results, Visuals (h.touloujian@student.tudelft.nl)

## Introduction
In this blog, an application of deep learning techniques for the purposes of denoising for image processing from the work of Chen, Chen, et al [1] is studied, discussed, and reproduced with various modifications. The goal of this reproduction effort is to explore the challenges that deep learning methods are capable of overcoming, in addition to the effect of varying such methods under different evaluation metrics, neural network structures, a more limited dataset, and new data previously unseen by a developed model.

This reproduction report thus consists of a concise summary of the reference work, as well as an analysis of the work's performance under several variations, which will be documented and explained throughout the report.

## Brief Paper Breakdown
### Challenges of Image Brightening & Denoising 
The work in [1] stems from an endemic challenge in image processing in the form of the denoising of short-exposure raw images to enhance their appearance with respect to a long-exposure counterpart. In other words, the pixels of images taken in the dark can be amplified to obtain a brighter version of the image. However, due to the short exposure time of the image, i.e., the lack of information captured by the camera due to the short time that it had to generate the image, the resulting amplified image, already consisting of an elevated Signal-to-Noise Ratio (SNR) pre-amplification, would produce an even higher SNR, resulting in a rather noisy image.

Prior to the work developed in [1], several denoising techniques had been considered to effectively reduce the SNR of the amplified image in order to produce an enhanced and brightened image. However, the work in question is one of the first of its kind to develop a neural network training pipeline for the denoising problem. As shown in the below figure, the pipeline manages to produced a denoised version of the amplified camera output (ISO 409,600) by inputting the camera output (ISO 8,000) into a trained neural network.

![](https://i.imgur.com/DEJNFtE.png)

### Image Processing Pipeline
The raw images in this reproduction report are considered to be Bayer arrays, which are the array types produced by Sony cameras. The arrays are processed by packing them into four channels and reducing the spatial resolution by a factor of two, followed by the subtraction of the black level, and finally multiplication by some amplification ratio hyperparameter.

![](https://i.imgur.com/SJNeSQB.png)

This amplified input is fed into the ConvNet block, which consists of a series of several convolutional layers and Leaky ReLU activation functions, as shown in the below figure. As can be observed, the input is expanded into several channels, progressively multiplied by a factor of 2 until 512 channels are reached. This is followed by a series of upsample and concatenation blocks, which resize the larger channels into half of their size, and concatenate the resized terms with the terms of equal channel size. This is done progressively until the channel size is reduced to 32, thus followed by a final convolution layer to reduce the channel size to 12. This then allows conversion to an RGB image, which can be perceived and evaluated for denoising peformance.

![](https://i.imgur.com/tgXy4QX.png)


### Training Information
Each short-exposure image that is fed into the neural network corresponds to a ground truth long-exposure image. The two are compared during the training process, which aims to minimize the L1-loss function of the error between the output of the neural network and the ground truth image. This training is effected for 4000 epochs using the Adam optimizer, the first 2000 of which correspond to a learning rate of $10^{-4}$, after which the learning rate is changed to $10^{-5}$. In this reproduction work, the same training process is implemented besides the stated modifications. 


## Main Project Goals
Our reproduction of this paper consists of three objectives: (1) a retraining of the full network with the L1-loss function replaced with an L2-loss function, (2) an ablation study in which a selection layers are removed from the middle of the network, and (3) testing the trained model on an entirely new dataset consisting of images taken by one of our co-authors with a camera sensor outputting raw images with a compatible (but different) Sony sensor.

In objectives (1) and (2), the network is trained with 10% of the original Sony dataset, primarily due to computational resource limitations, however this serves the additional purpose of validating the model's performance with less data.

## Test 0 - Training on 10% of the Dataset
Before implementing direct changes to the training method, an analysis of training the original model on 10% of the original dataset is presented. It is observed in the below plot that the L1-loss function still manages to reduce to a small value compared to the initial training error. Furthermore, it is noticed that the training algorithm converges relatively early compared to the final epoch of 4000. 
![](https://i.imgur.com/rPP0wPe.png)
An example of training resuls is shown below, where the left image represents the ground truth and the right image represents the reconstruction of the ground truth from the short-exposure image. In this case, the ground truth image exhibits 250x more exposure than the input image. Visually, it is clear that the model is capable of rendering the image visible and the text readable. However, the quality of the image does not match that of the ground truth, as the image remains blurry around text and finer details, such as the logos on the jar and the can.

![](https://i.imgur.com/Zlv4pri.jpg)

The below image shows that the network still manages to reproduce coarse features and finer ones in some cases, such as the streaks of reflection along the edges of the chair. However, in similar fashion, the quality of the image is not equal to that of its ground truth.

![](https://i.imgur.com/FR2aMZO.jpg)

The model is also tested on data that has not been seen in training, i.e., from a testing dataset. One exemplary result is shown below, where the top image represents the ground truth, the middle image represents the scaled input that suffers from a high SNR, and the bottom image represents the network output, i.e., the denoised image. It is clear that the original model manages to estimate the broad image adequately, especially overcoming the noise resulting from scaling the image. However, some reduction in quality is experienced due to the appearance of a slight tint in the output image compared to the ground truth. This may possibly a bias in the model that the network was unable to estimate.

![](https://i.imgur.com/8ccDe8Z.jpg)


The reductions in quality and appearance of tint can be attributed to the obvious handicap of a reduced dataset. However, they can also be fundamental limitations of the learning pipeline. To further understand this, the next two tests are attempted so as to understand the behavior of the model under different circumstances.


## Test 1 - Replacement of Loss Function
The L1-loss between the ground truth image and the network output can be understood as the sum of each individual pixel-to-pixel distance between the two images. Thus, the training process aims to reduce the total error of each individual pixel. On the other hand, using the L2-loss is equivalent to mapping the pixels of the ground truth and the output image into a vector space of dimension equal to the number of pixels in each image, and measuring the distance between the two vectors. Therefore, employing the L2-loss in training would be less focused on individual pixels, but rather the general differences between the output and the ground truth.

The network is thus trained on 10% of the original dataset with the L2-loss using the Adam optimizer for 4000 epochs, resulting in the below progression of the loss function. It is clear that the L2-Loss exhibits much more fluctuation than the L1-loss, and that its value is much larger, although this could be attributed to not having taken the average loss during training. This means that the effective reproduction error may be comparably small. 

![](https://i.imgur.com/y9GayBd.png)

An example of the testing dataset results is shown below, with the ground truth image on top, the output for the original model trained on 10% of the training dataset using L1-loss, and the same model trained using the L2-loss. An unexpected result from the denoising of the input is that in addition to reducing the SNR, the image is also deblurred, as is clear with the text heading on top of the board. Between the L1-loss and L2-loss results, no substantial changes are noticed, and it is deduced that both models perform fairly equally. However, this may not be the case in applications different from denoising.

![](https://i.imgur.com/5G9KXf8.jpg)



## Test 2 - Ablation Study
The network described by the original model exhibits a typical structure, with several repetitions of the same sequence of layers: two convolutions, a Max-Pooling layer, and an upsampling and concatenation layer followed by two more convolutions. In the ablation study performed in this work, we removed one of these repeated sequences (shown in the gray box in the figure below), re-routing the data from the last Max-Pooling layer to the next upscale and concatenation layer. 
![](https://i.imgur.com/nap2My4.png)

Given the depth of the ablated layers, it is expected that the effected change will be the network's inability to deal with finer features and details in the reproduction of the ground truth image.

Training is done for 4000 epochs using the L1-loss and the Adam Optimizer, resulting in the below loss progression. Compared to the original training in Test 0, it is observed that the loss function exhibits some initial fluctuations during training, and converges to a slightly larger loss, which is an expected result due to the removal of parameters.

![](https://i.imgur.com/LPORw6s.png)

The below image illustrates the inability of the ablated model to reproduce the fine features of the ground truth image, as is comparable to the results of Test 0. However, it is noticeable that the coarser features of the images, such as the shapes and colors of the sign and bushes can be reproduced to a satisfactory level.

![](https://i.imgur.com/TD5lfEC.jpg)

To further illustrate the extent of failure in reproducing finer features, the below image can be compared to the similar reproduction of the image of a chair from Test 0. The light reflection streaks along the edges of the chairs are reproduced successfully, albeit with some reduction in quality and definition at some points. 


![](https://i.imgur.com/poxwpA0.jpg)

A note on the ablation study is that the removal of a few layers from the training pipeline has compromised the training error slightly, and has also slightly affected the performance of the trained model. However, based on the application of the denoising problem, the ablated model can be viewed as a regularization method aimed at reducing the number of parameters in the trained model. For example, if precision is not a requirement of the application, i.e., it is aimed to only observe the coarser features of the image rather than the finer ones, it is possible to train the ablated network to save computational time and complexity.

## Test 3 - New Data
The aim of the study was to determine whether the pre-trained model could generalize well to images captured with a different camera sensor, namely a Nikon d800-serie camera and to identify any issues that may arise due to differences in the technical specifications of the two cameras.

![](https://i.imgur.com/f1yQ2Xl.jpg)

To achieve this, we captured a series of raw images in NEF format with the Nikon camera under varying lighting conditions, ranging from heavily under exposed to properly exposed images, with constant ISO and aperture settings. We then tested the pre-trained model that had originally been trained on images captured with the Sony camera/sensor on the captured Nikon images. To adapt the original testing code to the new data format, we had to convert the data from ARW to NEF format and adjust for the different black level in comparison to the old sensor. The Nikon camera has the same color levels and bit depth (14 bit) as the original Sony camera. 

![](https://i.imgur.com/1HfBVX2.jpg)


Our results showed that the pre-trained model performed remarkably well on images captured with the Nikon camera. Despite the differences in camera sensors, the pre-trained model was able to generalize well and achieve high accuracy in image recovery. However, we did observe a greenish and purplish color shift in some of the images, which could potentially be attributed to a bad reading of the white balance. Further investigation is needed to understand the root cause of this issue and to identify potential solutions. An another point to investigate is whether the results are still as good with a camera brand like Canon, as Nikon has a long sensor production history with Sony. The sensor in the Nikon camera is different from the one used in the Sony camera, but might have more similarities than for example a Canon sensor might have. 

![](https://i.imgur.com/w8Dh2yH.jpg)


The results suggest that with appropriate adaptations to the testing code and careful consideration of technical specifications, pre-trained models can be effectively applied to a range of different camera models and may help photographers all over the world to denoise and recover their late night pictures. 




## Conclusion
In this blog, we have discussed a reproduction report of a deep learning-based denoising technique for image processing proposed by Chen, Chen, et al. The study reproduced the work with alterations to explore the challenges deep learning methods can overcome and their performance under different evaluation metrics, neural network structures, a more concise dataset, and new data previously unseen by the model. The reproduced work used the same model structure as the original paper. The network was trained on short-exposure images and their corresponding long-exposure ground truth images using the L1-loss function for 4000 epochs. The reproduced study aimed to retrain the network with an L2-loss function, perform an ablation study by reducing the number of layers, and test the trained model on an entirely new dataset. In objective (1) and (2), the network was trained on only 10% of the original Sony dataset to challenge the network and reduce training time.

The end goal of a study like this is to gain a better understanding of the model and to analyze the effects of certain components on this system. The results showed that the reproduced model was capable of rendering the image visible and readable, although the quality of the image did not always match the ground truth. The reduction in training data, the change of loss function and the removal of a layer in the network had a smaller loss of quality than we initially expected. The model still performed exceptionally well despite some drastic changes. Based on these results, we have reached the conclusion that the model may be unnecessarily complex. The setup of this study was not big enough to fully support this conclusion but might be a challenge in the future to fully discover the true source of the empirical gains. 

Overall, this study highlights the potential of deep learning-based denoising techniques and their ability to enhance the appearance of images without inducing high noise levels. Where common tools like Photoshop and Lightroom would be insufficient in recovering underexposed pictures in a quality conserving way, this model gives us a tool to help us see in the dark. 

## Shared Online Workspaces 

GitHub: https://github.com/epsilam/Learning-to-See-in-the-Dark.git

Nikon dataset: https://drive.google.com/file/d/1XguTOJ7onU2Ttc5Q0MyJuGGUDlL4kcrD/view?usp=sharing 

All images presented herein, Nikon dataset, are the exclusive property of Floris van Leeuwen. The images may only be used for non-commercial and educational purposes. 
## References
[1] Chen, Chen, et al. "Learning to see in the dark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.