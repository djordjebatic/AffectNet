# AffectNet

This repo contains code for paper [TODO](TODO).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.
See deployment for notes on how to deploy the project on a NVIDIA Jetson Nano device.
Models were trained using a Tesla P100 GPU on the Google Cloud Platform.

### Performance

Our models reach near SOTA (state of the art) performance while at the same time minimizing space consumption. 
Average categorical accuracy is __60%__ (compared to baseline of __58%__ reported in [AffectNet paper](https://arxiv.org/abs/1708.03985)).
Per emotion categorical accuracy is the following:

    Neutral:         0.56
    Happy:           0.74 
    Sad:             0.63     
    Surprised:       0.59    
    Afraid:          0.62    
    Disgusted:       0.54    
    Angry:           0.59     
    Contemptuous:    0.53     

Average RMSE (root mean squared error) is __0.39__, same as baseline.

            VALENCE         AROUSAL
    RMSE    0.39            0.38

Both networks occupy total of just __28MB__ of space and achieve average inference speed of __**ms__ and __**ms__ respectively on NVIDIA Jetson Nano device.

Additional improvements in form of *** are planned.
For a detailed look at the performance metrics read our paper.

## Training

### Dependencies

Before training the network, you will need to install required dependencies in your virtual environment using pip.
If you are using conda you can create a new environment like this:

Step 1: Create a new environment
```
conda create -n affectnet-env python=3.6
```
Step 2: Activate the environment (don't continue to the following step without doing this!)
```
conda activate affectnet-env
```
Step 3: Install dependencies
```
pip install -r requirements.txt
```

### Training

To train a categorical model invoke the train function in __train.py__.
Following snippets are just examples, feel free to train with different parameters.


If you would like to train a categorical model:

```python
if __name__ == '__main__':
    model = mobilenet_v2_model(CLASSIFY)
    train(CLASSIFY, model, <OUTPUT_PATH>, 10, 64)
```
This will create an instance of the categorical model and then train it over 10 epochs with batch size of 64. 

##
If you would like to train a dimensional model:

```python
if __name__ == '__main__':
    model = mobilenet_v2_model(REGRESS)
    train(REGRESS, model, <OUTPUT_PATH>, 15, 16)
```

This will create an instance of the regression model and then train it over 15 epochs with batch size of 16. 

##
Lastly, if you would like to train a dimensional model by using weights from pretrained classification model:

```python
if __name__ == '__main__':
    model = mobilenet_v2_model(REGRESS)
    model.load_weights(<WEIGHTS_PATH>)
    for layer in model.layers:
        if type(layer) is Dropout:
            model.layers.remove(layer)
    regression_model = regressor_from_classifier(model, dropout=True)
    
    train(REGRESS, regression_model, <OUTPUT_PATH>, 10, 16)
```

This will create an instance of the categorical model and then load the weights of the pretrained one. 
First thing we need to do is to remove it's dropout layer. Second, by replacing the categorical model's output layer
with a 2 neuron linear output we create a regression model. It will train over 10 epochs with batch size of 16. 

##

To start training run the following command.

```bash    
$ python train.py
```
For additional information and detailed explanation regarding the training process read our paper.

## Inference

You can test model performance on you web camera or on video by placing an mp4 file with a name 'test.mp4' in deployment directory.

To run a test on web camera:
```bash
$ python inference.py -c
```
To run a test on video file:
```bash
$ python inference.py -v
```
You can trace both predictions and inference speed for each part of the process.
Running video test creates test_out.avi file. 

## Deployment

Models were deployed on NVIDIA Jetson Nano device running on the Ubuntu 18.04 OS with Keras version 2.2.4 and Tensorflow version 1.15.2 installed.
In order to use them on your own Jetson Nano device, clone this repository and run the script by typing the following command.
For additional information see [this link](https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification)
```bash
```

## Dataset

[AffectNet](http://mohammadmahoor.com/affectnet/) is one of the largest databases of facial expressions, valence, and arousal in the wild enabling research in automated facial expression recognition in two different emotion models.
It contains more than 400K facial images that were manually annotated for the presence of seven discrete facial expressions (categorial model) and the intensity of valence and arousal (dimensional model).
Both models were trained using only the manually annotated images.

## Authors


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
