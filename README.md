# Moodify App
This is an app which is a music recommender that takes in your picture, predicts your mood and accordingly suggests a music playlist made from your history
on spotify. This is a merger of a facial mood recognition system with a music recommendation algorithm.

The app has been made in streamlit, the ML models have been made using tensorflow and in order to integrate it with spotify, spotify api along with spotipy has been used.  In order to use the app you will have to provide an access token to your spotify account which will give it access to your history on Spotify and also allow it to add the generated playlist to your spotify account. 

## Dependencies
In order for the app to run you need to have the folllowing libraries:
- streamlit 0.84.0
- pandas 1.2.3
- numpy 1.19.5
- opencv-python-headless
- imageio
- tensorflow-cpu 2.5.0
- scikit-learn 0.23.2
- spotipy 2.18.0
- pickle-mixin 1.0.2

You can insall these using the requirements.txt file. You can run `pip install -r requirements.txt` file. 

## Running the app
Once you have got the repo cloned and the dependencies installed you can simply run the app using `streamlit run moodify_app.py`. After running you will be able to see the app:

![app interface](https://github.com/Karrthik-Arya/mooodify_app/blob/master/app-interface.png)

You can then use the app, first you will have to input the access token, then using the sidebar you can either got to the demo image(where will be able to use some of the demo images we have given) and see the results or you can go to the upload image option , upload whatever image you want, the model will predict the mood and suggest the songs. If the *Add playlist to my spotify account* option is selected the songs suggested would also be added as a playlist to your spotify account. 


The *moodify_app.py* is the main file which contains the code for the app and in which all the other functions are integrated. The *emotion_classify.py* and *test.py* contain the code for the image classifcation model. They detect the face in an image and then classify the and then classify them into the mood using the CNN model made in tensorflow. 

The *music_test.py* contains the code to recommend songs of a particular moods from a dataset of songs. It takes the dataset of songs and classifies the songs into moods and returns songs of the desired mood. It is using a feed forward neural net to classify the songs. 
The *auth.py* contains the code for integrating with spotify. It uses the spotify api and and spotipy to acces the songs in a user's history and also to make new playlists for them.
