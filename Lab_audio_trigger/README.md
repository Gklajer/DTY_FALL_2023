## Lab 5: Speech Recognition

This lab is based on a [Kaggle competition](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) and the [DL course](https://www.deeplearning.ai/).

In this Lab, we will construct a speech dataset and implement an algorithm for trigger word detection. Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri to wake up upon hearing a certain word.

Given that audio clips are of temporal nature, we will use Recurrent Neural Networks, where the new state at time `t` can be dependent on all the previous steps `[0, t-1]` (given enough computationnal resources), for a brief introduction to these types of models, please refer to [A short intro to RNNs](RNN_intro.pdf).

In this lab, our trigger word will be "Activate", so every time we detect the corresponding sound to the word "activate," we need to detect it, and we will output make a "bell" sound.

This lab will be devided into three sections to: 
- Synthesizing and processing the raw audio recordings to create train/validation datasets.
- Defining our model based on the provided architecture.
- Training and testing our model.

## Packages

After the first few labs, all the packages we might need are already installed, the only package missing is `pydub`, that we will use to manipulate our audio recordings and create a train and validation sets, please install it:

`pip install pydub`
 
Now you can start the lab by heading to the notebook [speech_lab.ipynb](speech_lab.ipynb)

If you finished and you have some time to kill, consider using the model to create a REST API: [Bonus: Keras & REST](bonus_keras_REST.md).

