import streamlit as st
import pandas as pd 
import numpy as np 
import cv2
import os
import imghdr
import tempfile
from PIL import Image, ImageOps
import emotion_classify as EC
from classify import recommend
import fetching_playlists as fp
import auth
#[theme]
#base="light"
#primaryColor="#751a55"
#secondaryBackgroundColor="#fbcfd5"

IMAGE_DISPLAY_SIZE = (330, 330)
IMAGE_DIR = 'demo_photos'
TEAM_DIR = 'moodify_team'

st.image(os.path.join(TEAM_DIR,'Logo.png'), use_column_width = True)
st.title('Welcome to Moodify!')
st.write(" ------ ")

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_MEET_TEAM]
#needs one arg as mood ,history should return csv format of history of songs including their stats ((csv format))
def load_model(mood, token):
    df  = auth.make_df(token)
    final_play_lst=recommend(mood,df)
    ids = list(final_play_lst["ID"])
    final_play_lst = final_play_lst[["Name", "Artist", "Album"]]
    #final_play_lst = final_play_lst.style.set_properties(**{'text-align': 'left'})
    st.write(final_play_lst)
    return ids
       

def load_and_preprocess_img(img_path, num_hg_blocks, bbox=None):
    img = Image.open(img_path).convert('RGB')
        
    # Required because PIL will read EXIF tags about rotation by default. We want to
    # preserve the input image rotation so we manually apply the rotation if required.
    # See https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image/
    # and the answer I used: https://stackoverflow.com/a/63798032
    try:
        img = ImageOps.exif_transpose(img)
    except: 
        pass

    new_img = cv2.resize(np.array(img), IMAGE_DISPLAY_SIZE,
                        interpolation=cv2.INTER_LINEAR)

    return new_img

def run_app(img, token):
    left_column, right_column = st.beta_columns(2)
    xb = load_and_preprocess_img(img, num_hg_blocks=1)
    display_image = cv2.resize(xb, IMAGE_DISPLAY_SIZE,
                        interpolation=cv2.INTER_LINEAR)
    mood_img=EC.emotion_detect(display_image)
    left_column.image(display_image, caption = "Selected Input")
    right_column.image(display_image, caption = "Predicted mood:" + str(mood_img) )
    ids = load_model(mood_img, token)
    return ids
     


def main():
    st.sidebar.warning('\
        Please upload SINGLE-person images. For best results, please also CENTER the person in the image.')
    st.sidebar.write(" ------ ")
    st.sidebar.title("Explore the Following")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)
    st.markdown("""<p>Before proceeding, please go to <a href='https://developer.spotify.com/console/get-recently-played/' target="_blank">this</a> website, click on get token and click on 'user-recently-played' and 'playlist-modify-private' to generate token. Paste the obtained OAuth Token here </p>""", unsafe_allow_html=True)
    token = st.text_input("Access token: ")
    if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write(" ------ ")
        st.sidebar.success("Project information showing on the right!")
        st.write('''
            # Mood based Music Recommender system

         This project helps the user to automatically play songs based on the emotions of the user. 
         It recognizes the facial emotions of the user and predicts the songs according to their mood.

         **Our goal is to build a song playlist for the individual in each picture.**

         👈 Please select **Select a Demo Image** in the sidebar to start.

         📸 Feel free to upload any image you want to get a song prediction under **Upload an Image**

         📞 Our team members are here to answer questions. Please refer to **Contact Information** under **Meet the Team**.''')
    

    elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        if not token:
            st.warning("Please enter access token")
        else:
            st.sidebar.write(" ------ ")

            directory = os.path.join(IMAGE_DIR)

            photos = []
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)

                # Find all valid images
                if imghdr.what(filepath) is not None:
                    photos.append(file)

            photos.sort()

            option = st.sidebar.selectbox('Please select a sample image, then click the button', photos)
            check = st.sidebar.checkbox("Add the playlist to my Spotify account")
            pressed = st.sidebar.button('Create playlist')
            
            if pressed:
                st.empty()
                st.sidebar.write('Please wait for the playlist to be created! This may take up to a few minutes.')

                pic = os.path.join(directory, option)

                ids = run_app(pic, token)
                if check:
                    st.empty()
                    auth.create_playlist(token, ids)
                    st.success("Playlist has been added to your Spotify account!")
                



    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
        if not token:
            st.warning("Please enter access token")
        else:
            #upload = st.empty()
            #with upload:
            st.sidebar.info('PRIVACY POLICY: Uploaded images are never saved or stored. They are held entirely within memory for prediction \
                and discarded after the final results are displayed. ')
            check = st.sidebar.checkbox("Add the playlist to my Spotify account")
            f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif', 'JPG'])
            if f is not None:
                tfile = tempfile.NamedTemporaryFile(delete=True)
                tfile.write(f.read())
                st.sidebar.write('Please wait for the playlist to be created! This may take up to a few minutes.')
                ids = run_app(tfile, token)
                if check:
                    st.empty()
                    auth.create_playlist(token, ids)
                    st.success("Playlist has been added to your Spotify account!")
      
    
   
    else:
        raise ValueError('Selected sidebar option is not implemented. Please open an issue on Github: LINK SOON')

main()
expander_faq = st.beta_expander("More About Our Project")
expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: LINK")


