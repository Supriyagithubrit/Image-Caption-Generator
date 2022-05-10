***Image Caption Generator***
-----------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/105269665/167598065-3035908c-32b1-409f-abac-b17b41e60bbc.png)

**Dataset:**
------------------------------------------------------------------------------------------------------
FLICKR_8K dataset: This dataset includes around 8095 images along with 5 different captions written by different people for each image. The images are all contained together while caption text file has captions along with the image number appended to it. The zip file is approximately over 1 GB in size.

![image](https://user-images.githubusercontent.com/105269665/167600950-6f4ae28b-d20d-40c7-a424-96c3e53b4d2a.png)

**Flow of the project**

a. Cleaning the caption data
b. Extracting features from images using VGG-16
c. Merging the captions and images
d. Building LSTM model for training
e. Predicting on test data
f. Evaluating the captions using BLEU scores as the metric

**Steps to follow:**
----------------------------------------------------------------------------------------------
**1. Cleaning the captions**
This is the first step of data pre-processing. The captions contain regular expressions, numbers and other stop words which need to be cleaned before they are fed to the model for further training. The cleaning part involves removing punctuations, single character and numerical values. After cleaning we try to figure out the top 50 and least 50 words in our dataset.

![image](https://user-images.githubusercontent.com/105269665/167602162-ecb1346d-23b3-42b4-a470-469ca4406477.png)

![image](https://user-images.githubusercontent.com/105269665/167602739-edf3e0aa-7346-4076-aad6-6d9f5055dc6a.png)

**2. Adding start and end sequence to the captions**
Start and end sequence need to be added to the captions because the captions vary in length for each image and the model has to understand the start and the end.

**3. Extracting features from images**

**.** After dealing with the captions we then go ahead with processing the images. For this we make use of the pre-trained InceptionV3 weights.
**.** Instead of using this pre-trained model for image classification as it was intended to be used. We just use it for extracting the features from the images. In order to do that we need to get rid of the last output layer from the model. The model then generates 2048 features from taking images of size (229,229,3).
![image](https://user-images.githubusercontent.com/105269665/167604546-293e9174-1b75-4deb-8c00-5e027e717553.png)

**4. Splitting the data for training and testing**
The tokenized captions along with the image data are split into training, test and validation sets as required and are then pre-processed as required for the input for the model.

![image](https://user-images.githubusercontent.com/105269665/167605641-21498a42-5fd4-4453-8af7-c9391ec1c84e.png)

**5. Building the CNN-LSTM model**

LSTM model is been used beacuse it takes into consideration the state of the previous cell's output and the present cell's input for the current output. This is useful while generating the captions for the images.
The step involves building the LSTM model with two or three input layers and one output layer where the captions are generated. The model can be trained with various number of nodes and layers. We start with 256 and try out with 512 and 1024. Various hyperparameters are used to tune the model to generate acceptable captions

![image](https://user-images.githubusercontent.com/105269665/167607121-e2896f78-b111-4ea5-8035-700f8c513b11.png)

**6. Predicting on the test dataset**

After the model is trained, it is tested on test dataset to see how it performs on caption generation for just 5 images. If the captions are acceptable then captions are generated for the whole test data.

![image](https://user-images.githubusercontent.com/105269665/167610212-74056995-8d0c-423b-8386-e0b3df88f141.png)

**Conclusion**
-----------------------------------------------------------------------------------------------------------------
Implementing the model is a time consuming task as it involved lot of testing with different hyperparameters to generate better captions. The model generates good captions for the provided image but it can always be improved.






