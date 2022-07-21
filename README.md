# Hand Gesture Detection
## A program for detecting hand gestures based on ASL.

Use `data_collection.py` for training. All you have to do is save the images of your hand in a certain position. 
Then go to the [Teachable Machine](https://teachablemachine.withgoogle.com) website and train the model, using photography training.
Export the `TenserFlow` model using the `Keras` model transformation type.
Then upload it to the `model` folder and then run `app.py`. 

Check `misc.py` for customization.
