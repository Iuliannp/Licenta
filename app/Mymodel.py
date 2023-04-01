from pathlib import Path
from keras.models import load_model
import os
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings
from PIL import Image
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample_rate = 8000
image_width = 500
image_height = 128

classes = {
    0 : 'de',
    1 : 'en',
    2 : 'fr'
}

model = load_model("./model.h5")

def load_audio_file(audio_file_path):
    warnings.simplefilter('ignore', UserWarning)
    
    audio_segment, _ = lr.load(audio_file_path, sr=sample_rate)
    return audio_segment

    warnings.simplefilter('default', UserWarning)
    
def fix_audio_segment_to_10_seconds(audio_segment):
    target_len = 10 * sample_rate
    audio_segment = np.concatenate([audio_segment]*3, axis=0)
    audio_segment = audio_segment[0:target_len]
    
    return audio_segment

def to_integer(image_float):
    # range (0,1) -> (0,255)
    image_float_255 = image_float * 255.
    
    # Convert to uint8 in range [0:255]
    image_int = image_float_255.astype(np.uint8)
    
    return image_int

def spectrogram(audio_segment):
    # Compute mel-scaled spectrogram image
    hl = audio_segment.shape[0] // image_width
    spec = lr.feature.melspectrogram(y=audio_segment, n_mels=image_height, hop_length=int(hl))

    # Logarithmic amplitudes
    image = lr.core.power_to_db(spec)

    # Convert to numpy matrix
    image_np = np.asmatrix(image)

    # Normalize and scale
    image_np_scaled_temp = (image_np - np.min(image_np))
    
    image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)

    return image_np_scaled[:, 0:image_width]

def audio_to_image_file(audio_file):
    out_image_file = audio_file + '.png'
    audio = load_audio_file(audio_file)
    audio_fixed = fix_audio_segment_to_10_seconds(audio)
    if np.count_nonzero(audio_fixed) != 0:
        spectro = spectrogram(audio_fixed)
        spectro_int = to_integer(spectro)
        print(out_image_file)
        imageio.imwrite(out_image_file, spectro_int)
    else:
        print('WARNING! Detected an empty audio signal. Skipping...')
        
def makePrediction(file):
    audio_to_image_file(file)
    
    img = load_img(file + '.png', target_size=(image_height, image_width), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = img_array.astype('float32') / 255.0
     
    os.remove(file+'.png')
    
    prediction = model.predict(img_array)
    # predicted_class_index = np.argmax(prediction)
    # predicted_class_label = classes[predicted_class_index]
    # return predicted_class_label
    return prediction

        
if __name__ == '__main__':
    print(makePrediction(sys.argv[1]))
           