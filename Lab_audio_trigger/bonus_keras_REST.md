# [BONUS] Keras & a REST API

In you finished lab 5 and want to go further, you can create a REST API to use our Keras model for trigger word detection across the web (for example, imagine you have one server for trigger word detection, and you want it to use across different rooms).

We'll create REST API, but this time, where we'll send an audio recoding and get a response, and it will be the model's prediction (is there a trigger in the audio clip, if yes, when).

### Building the Keras REST API

Our Keras REST API will be self-contained in a single file named `run_keras_server.py`. We'll keep the installation in a single file as a manner of simplicity,the implementation can be easily modularized as well.

Inside `run_keras_server.py` we'll define three functions, namely:

- `load_model`: Used to load our trained Keras model from the last section and prepare it for inference.
-  `prepare_audio`: This function preprocesses an given audio recording prior to passing it through our network for prediction.
- `predict`: The actual endpoint of our API that will classify the incoming data from the request and return the results to the client.

First install `flask` if you don't have it, and then let's start by importing the packages we'll use, 

```python
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import flask
import io
```

Let's initialize our Flask app:

```python
app = flask.Flask(__name__)
```

### Load the model

Now, implement the `load_model` to re-use our trigger word detection that we've trained:

```python
def loading_the_model(model_path):
    # load the pre-trained trigger word detection Keras model
    global model
    model = ....
```

### Preprocessing the data

Before we can perform prediction on any data coming from our client. We first need to prepare it and pre-process it. In our case, we'll get a raw audio file (the output of the function `scipy.io.wavfile`), we'll need to apply FFT and produce the spectrogram and swap the axes. Implement these steps in the following function (simply re-use some parts of the two functions `graph_spectrogram` and `detect_triggerword` from the last tutorial).

```python
def prepare_audio(audio):
    # Get spectrogram
    # swat axes and expand dim
    # return the processed audio
    return audio
```

### Post-processing the predictions

We are now ready to define the `predict` function, this function will process any requests to the `/predict` endpoint of our REST API. From the user we'll get a stream of bits corresponding to a given audio recording. It will be the inputs to our model. We read it as a `wav` file, pre-process it, pass it through the model and then process the predictions to find if a trigger (or more) was (were) detected.

Now it is your turn to implement the post processing function `process_predictions`, this function will return two variables:
- `detected`: Boolean variable, True if one or more occurrences of the trigger word were detected in the audio clip.
- `timestamps`: A list, empty if no trigger word was detected, the time-stamps of when we detected the trigger word in the audio clip.

You can re-use the code in the function `ring_on_activate` with the corresponding modifications, notice that we pass the duration of the clip in seconds, it'll be used to calculate the time-stamps just like in `ring_on_activate`:

```python
def process_predictions(predictions, audio_duration_sec, threshold=0.5):
    detected, timestamps = False, []
    # Loop over the predictions
    # Change `detected` if a trigger word was detected
    # Add its time-stamp (after calculating it) to `timestamps`
    # return the variables

    return detected, timestamps
```


### REST API

In the main route with the method POST, you'll need to call the functions we've created to load the audio as `wav`, pre-process it, use the model to make prediction and post-process them.

```python
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an input was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("audio"):
            # receive the audio from the request call
            audio = flask.request.files["audio"].read()
            rate, audio = wavfile.read(io.BytesIO(audio))

            # Add your code here 
            # calculate the amount of seconds in the audio
            # (using `rate` and audio.shape[0])
            # pre-process the audio and prepare it for classification
            # Pass the audio through our model
            # Pass the prediction though a post processing function

            # Did we find a trigger word, and when
            data["trigget_detected"] = detected
            data["timestamps"] = timestamps

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
```


### Running the web service

All that's left to do now is launch our service:

```python
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    model_path = # PATH TO YOUR TRAINED MODEL HERE
    loading_the_model(model_path)
    app.run()
```

First we call `load_model` which loads our Keras model from `model_path`.

The call to `load_model` is a blocking operation and prevents the web service from starting until the model is fully loaded. Had we not ensured the model is fully loaded into memory and ready for inference prior to starting the web service we could run into a situation where:

1. A request is `POST`'ed to the server.
1. The server accepts the request, pre-processes the data, and then attempts to pass it into the model.
3. ...but since the model isn't fully loaded yet, our script will error out.

**Note**: We are using the default Flask server that is single threaded, so it is critical to only load our model one time, at the beginning before the app was started. Say we re-loaded the `model` each and every time a new request comes in. This is incredibly inefficient and can even cause system to run out of memory, the API will be considerably slower. But in some situation with different models, and different applications. We might be forced to load different version on the fly, well in this case we need bigger guns, i.e. a dedicated server such as Apache or nginx, but that's for another time.

### Starting the Keras Rest API

Let's run our script to start the API: `python run_keras_server.py`. You'll see that the model will be loaded first, and then the app will start.

In this case, and unlike the first tutorial where we did implement an `index.html` page. we don't have any home pages were set in the Flask URLs routes, so we can't now access the server via `http://127.0.0.1:5000`.

All you'll see ais "Method Not Allowed" error. This error is due to the fact that the browser is performing a `GET `request, but `/predict` only accepts a `POST`, which we'll call, just like we did with the todo list REST API, using `cURL`.

### Using cURL to test the Keras REST API

Let's using some of the audio examples in the last tutorial, and send it to our server usign a `cURL` command.

```shell
$ curl -X POST -F audio=@trigger_word_detection/raw_data/test/1.wav "http://localhost:5000/predict"
```

The `-X` flag and POST value indicates we're performing a POST request.

We supply `-F audio=@trigger_word_detection/raw_data/test/1.wav` to indicate we're submitting form encoded data. The audio key is then set to the contents of the `1.wav` file. Supplying the `@` prior to `1.wav` implies we would like `cURL` to load the contents (audio in our case) and pass the data to the request. Finally, we have our endpoint: `http://localhost:5000/predict`

Make sure you get the correct response. You'll need to receive a success response with one time-stamp at around `2560 ms`. Now test it the second example `2.wav`. This time you need to receive two time-stamps at around `2370 ms` and `4930 ms`.

### Consuming the Keras REST API programmatically

Instead of using `cURL`, IRL, we'll use our API programmatically, this is a straightforward process using the `requests` Python package, in a new python file `simple_request.py`, add the following code:

```python
import requests

# initializing the Keras REST API endpoint URL along with the input
# audio path
KERAS_REST_API_URL = "http://localhost:5000/predict"
AUDIO_PATH = "raw_data/test/1.wav"

# loading the audio and constructing the payload for the request
audio = open(AUDIO_PATH, "rb").read()
payload = {"audio": audio}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensuring the request was successful and print the results
if r["success"]:
    print(r)

# otherwise, the request failed
else:
    print("Request failed")
```

The `KERAS_REST_API_URL` specifies our endpoint while the `AUDIO_PATH` is the path to our input audio residing on disk. Using the `AUDIO_PATH` we load the audio and then construct the payload to the request.

Given the payload we can `POST` the data to our endpoint using a call to requests.post. Appending .json() to the end of the call instructs requests that:

1. The response from the server should be in JSON
1. We would like the JSON object automatically parsed and deserialized for us

Once we have the output of the request, `r`, we can check if the prediction was a success (or not) and then print the predictions.

To test the script, run it `python simple_request.py`, but first ensure that `run_keras_server.py` (i.e., the Flask web server) is currently running. If everything in correct, you'll get the correct and same response as earlier.

### Testing

Now, you can test the API with some of your own recordings, just make sure the audio format is `.was`, if it is an `.mp3`, you can easily convert it [here](https://convertio.co/fr/mp3-wav/).

### Optional: Putting it together

If you are up to it, you can create a web-page for `Word detection`, where we can upload the audio-file (see [Uploading Files](http://flask.palletsprojects.com/en/1.0.x/patterns/fileuploads/)), send the audio-file to the Keras REST API, get the predictions and display them in the website.
