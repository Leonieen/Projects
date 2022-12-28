# -*- coding: utf-8 -*-

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import random
from pedalboard.io import AudioFile
from PIL import Image
from scipy.io import wavfile
import librosa

# This function generates frequencies in Hertz from notes
def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440  # Frequency of Note A4
    keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end + 1]

    note_freqs = dict(zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0  # stop
    return note_freqs


# Make scale as specified by user
def makeScale(whichOctave, whichKey, whichScale):
    # Load note dictionary
    note_freqs = get_piano_notes()

    # Define tones. Upper case are white keys in piano. Lower case are black keys
    scale_intervals = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']

    # Find index of desired key
    index = scale_intervals.index(whichKey)

    # Redefine scale interval so that scale intervals begins with whichKey
    new_scale = scale_intervals[index:12] + scale_intervals[:index]

    # Choose scale
    if whichScale == 'AEOLIAN':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'BLUES':
        scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]
    elif whichScale == 'PHYRIGIAN':
        scale = [0, 1, 3, 5, 7, 8, 10]
    elif whichScale == 'CHROMATIC':
        scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif whichScale == 'DORIAN':
        scale = [0, 2, 3, 5, 7, 9, 10]
    elif whichScale == 'HARMONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 11]
    elif whichScale == 'LYDIAN':
        scale = [0, 2, 4, 6, 7, 9, 11]
    elif whichScale == 'MAJOR':
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif whichScale == 'MELODIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
    elif whichScale == 'MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'MIXOLYDIAN':
        scale = [0, 2, 4, 5, 7, 9, 10]
    elif whichScale == 'NATURAL_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'PENTATONIC':
        scale = [0, 2, 4, 7, 9]
    else:
        print('Invalid scale name')

    # Initialize arrays
    freqs = []
    for i in range(len(scale)):
        note = new_scale[scale[i]] + str(whichOctave)
        freqToAdd = note_freqs[note]
        freqs.append(freqToAdd)
    return freqs


# Convery Hue value to a frequency
def hue2freq(h, scale_freqs):
    thresholds = [26, 52, 78, 104, 128, 154, 180]
    # note = scale_freqs[0]
    if (h <= thresholds[0]):
        note = scale_freqs[0]
    elif (h > thresholds[0]) & (h <= thresholds[1]):
        note = scale_freqs[1]
    elif (h > thresholds[1]) & (h <= thresholds[2]):
        note = scale_freqs[2]
    elif (h > thresholds[2]) & (h <= thresholds[3]):
        note = scale_freqs[3]
    elif (h > thresholds[3]) & (h <= thresholds[4]):
        note = scale_freqs[4]
    elif (h > thresholds[4]) & (h <= thresholds[5]):
        note = scale_freqs[5]
    elif (h > thresholds[5]) & (h <= thresholds[6]):
        note = scale_freqs[6]
    else:
        note = scale_freqs[0]

    return note


# Make song from image!
def img2music(img, scale=[220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30],
              sr=22050, T=0.1, nPixels=60, useOctaves=True, randomPixels=False,
              harmonize='U0'):
    """
    Args:
        img    :     (array) image to process
        scale  :     (array) array containing frequencies to map H values to
        sr     :     (int) sample rate to use for resulting song
        T      :     (int) time in seconds for dutation of each note in song
        nPixels:     (int) how many pixels to use to make song
    Returns:
        song   :     (array) Numpy array of frequencies. Can be played by ipd.Audio(song, rate = sr)
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Get shape of image
    height, width, depth = img.shape

    i = 0;
    j = 0;
    k = 0
    # Initialize array the will contain Hues for every pixel in image
    hues = []
    if randomPixels == False:
        for val in range(nPixels):
            hue = abs(hsv[i][j][0])  # This is the hue value at pixel coordinate (i,j)
            hues.append(hue)
            i += 1
            j += 1
    else:
        for val in range(nPixels):
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            hue = abs(hsv[i][j][0])  # This is the hue value at pixel coordinate (i,j)
            hues.append(hue)

    # Make dataframe containing hues and frequencies
    pixels_df = pd.DataFrame(hues, columns=['hues'])
    pixels_df['frequencies'] = pixels_df.apply(lambda row: hue2freq(row['hues'], scale), axis=1)
    frequencies = pixels_df['frequencies'].to_numpy()

    # Convert frequency to a note
    pixels_df['notes'] = pixels_df.apply(lambda row: librosa.hz_to_note(row['frequencies']), axis=1)

    # Convert note to a midi number
    pixels_df['midi_number'] = pixels_df.apply(lambda row: librosa.note_to_midi(row['notes']), axis=1)

    # Make harmony dictionary
    # unison           = U0 ; semitone         = ST ; major second     = M2
    # minor third      = m3 ; major third      = M3 ; perfect fourth   = P4
    # diatonic tritone = DT ; perfect fifth    = P5 ; minor sixth      = m6
    # major sixth      = M6 ; minor seventh    = m7 ; major seventh    = M7
    # octave           = O8
    harmony_select = {'U0': 1,
                      'ST': 16 / 15,
                      'M2': 9 / 8,
                      'm3': 6 / 5,
                      'M3': 5 / 4,
                      'P4': 4 / 3,
                      'DT': 45 / 32,
                      'P5': 3 / 2,
                      'm6': 8 / 5,
                      'M6': 5 / 3,
                      'm7': 9 / 5,
                      'M7': 15 / 8,
                      'O8': 2
                      }
    harmony = np.array([])  # This array will contain the song harmony
    harmony_val = harmony_select[harmonize]  # This will select the ratio for the desired harmony

    # song_freqs = np.array([]) #This array will contain the chosen frequencies used in our song :]
    song = np.array([])  # This array will contain the song signal
    octaves = np.array([0.5, 1, 2])  # Go an octave below, same note, or go an octave above
    t = np.linspace(0, T, int(T * sr), endpoint=False)  # time variable
    # Make a song with numpy array :]
    # nPixels = int(len(frequencies))#All pixels in image
    for k in range(nPixels):
        if useOctaves:
            octave = random.choice(octaves)
        else:
            octave = 1

        if randomPixels == False:
            val = octave * frequencies[k]
        else:
            val = octave * random.choice(frequencies)

        # Make note and harmony note
        note = 0.5 * np.sin(2 * np.pi * val * t)
        h_note = 0.5 * np.sin(2 * np.pi * harmony_val * val * t)

        # Place notes into corresponfing arrays
        song = np.concatenate([song, note])
        harmony = np.concatenate([harmony, h_note])
        # song_freqs = np.concatenate([song_freqs, val])

    return song, pixels_df, harmony


# Adding an appropriate title for the test website
st.title("Geodata Sounds")

st.markdown(
    "Have you always wondered how geospatial data sounds? This app converts the color values of images into a sound. Many thanks go out to Victor Murcia for his great tutorial on converting images to songs (https://github.com/victormurcia/Making-Music-From-Images). You can choose some sample images on the left, they are from other projects from my portfolio or upload your own. Also feel free to play with the inputs to optimize and adjust the sound of your image. ")
# Making dropdown select box containing scale, key, and octave choices
df1 = pd.DataFrame({'Scale_Choice': ['AEOLIAN', 'BLUES', 'PHYRIGIAN', 'CHROMATIC', 'DORIAN', 'HARMONIC_MINOR', 'LYDIAN',
                                     'MAJOR', 'MELODIC_MINOR', 'MINOR', 'MIXOLYDIAN', 'NATURAL_MINOR', 'PENTATONIC']})
df2 = pd.DataFrame({'Keys': ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']})
df3 = pd.DataFrame({'Octaves': [1, 2, 3]})
df4 = pd.DataFrame({'Harmonies': ['U0', 'ST', 'M2', 'm3', 'M3', 'P4', 'DT', 'P5', 'm6', 'M6', 'm7', 'M7', 'O8']})

st.sidebar.markdown(
    "Here you can choose one of the images from my portfolio or upload your own:")
_radio = st.sidebar.radio("", ("Sample Image", "User Image"))

#sample_images = glob.glob('*.jpg')
d = {'Path': ["https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Choropleth1.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Choropleth2.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Choropleth3.jpg ",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/DEM.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Arial.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B02–B11–B12.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B04–B03–B02.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B08–B04–B03.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B11–B8A–B02.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B12–B11–B8A.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/B8A–B11–B02.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Heatmap.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Hillshade.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/MNDWI.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/NDVI.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/NilsHolgerssonMap.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/OsloContours.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/OsloMap.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/OsloMask.jpg ",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/SAVI.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Slope.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/Slope2.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/contour lines.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/map1.jpg",
                "https://raw.githubusercontent.com/Leonieen/Projects/main/Image2Music/SampleImages/sentinel2.jpg",
                ],
     'Images': ['Choropleth1.jpg', 'Choropleth2.jpg', 'Choropleth3.jpg',
              'DEM.jpg', 'Arial.jpg', 'B02–B11–B12.jpg', 'B04–B03–B02.jpg',
              'B08–B04–B03.jpg', 'B11–B8A–B02.jpg', 'B12–B11–B8A.jpg', 'B8A–B11–B02.jpg',
              'Heatmap.jpg', 'Hillshade.jpg', 'MNDWI.jpg', 'NDVI.jpg',
              'NilsHolgerssonMap.jpg', 'OsloContours.jpg', 'OsloMap.jpg', 'OsloMask.jpg',
              'SAVI.jpg', 'Slope.jpg', 'Slope2.jpg', 'contour lines.jpg', 'map1.jpg', 'sentinel2.jpg']}
samp_imgs_df = pd.DataFrame(data=d)
#samp_imgs_df = pd.DataFrame(sample_images, columns=['Images'])
samp_img = st.sidebar.selectbox('Choose a sample image from my portfolio', samp_imgs_df['Images'])

# Load image
user_data = st.sidebar.file_uploader(label="Upload your own Image")
if _radio == "Sample Image":
    img2load = samp_img
elif _radio == "User Image":
    img2load = user_data

# Display the image
st.sidebar.image(img2load)

col1, col2, col3, col4 = st.columns(4)

with col1:
    scale = st.selectbox('Choose a scale: ', df1['Scale_Choice'])

    'Selected scale: ' + scale
with col2:
    key = st.selectbox('Choose a key: ', df2['Keys'])

    'Selected key: ', key

with col3:
    octave = st.selectbox('Choose an octave: ', df3['Octaves'])

    'Selected octave: ', octave
with col4:
    harmony = st.selectbox('Choose a harmony: ', df4['Harmonies'])

    'Selected value: ', harmony

col5, col6 = st.columns(2)
with col5:
    # Ask user if they want to use random pixels
    random_pixels = st.checkbox('Use random pixels to build song?', value=True)
with col6:
    # Ask user to select song duration
    use_octaves = st.checkbox('Randomize note octaves while building song?', value=True)

col7, col8 = st.columns(2)
with col7:
    # Ask user to select song duration
    t_value = st.slider('Note duration [s]', min_value=0.01, max_value=1.0, step=0.01, value=0.2)

with col8:
    # Ask user to select song duration
    n_pixels = st.slider('How many pixels to use? (More pixels take longer)', min_value=12, max_value=320, step=1,
                         value=60)

# Making the required prediction
if img2load is not None:
    # Saves
    img = Image.open(img2load)
    img = img.save("img.jpg")

    # OpenCv Read
    img = cv2.imread("img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    st.image(img)

    # Make the scale from parameters above
    scale_to_use = makeScale(octave, key, scale)

    # Make the song!
    song, song_df, harmony = img2music(img, scale=scale_to_use, T=t_value, randomPixels=random_pixels,
                                       useOctaves=use_octaves, nPixels=n_pixels, harmonize=harmony)

    # Write the song into a file
    song_combined = np.vstack((song, harmony))
    wavfile.write('song.wav', rate=22050, data=song_combined.T.astype(np.float32))

    audio_file = open('song.wav', 'rb')
    audio_bytes = audio_file.read()

    # Read in a whole audio file:
    with AudioFile('song.wav', 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    # Play the processed song
    st.audio(audio_bytes, format='audio/wav')


    # @st.cache
    def convert_df_to_csv(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    # csv = song_df.to_csv('song.csv')
    st.download_button('Download the Song as CSV', data=convert_df_to_csv(song_df), file_name="song.csv", mime='text/csv',
                       key='download-csv')
# While no image is uploaded
else:
    st.write("Uploade your image...")
# st.markdown("# Main page 🎈")
# st.sidebar.markdown("# Main page 🎈")
