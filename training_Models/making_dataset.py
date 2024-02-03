import numpy as np
import argparse
import time
import cv2 as cv
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont
import re
import os

names = ['Agni Suktam', 'agnisUktam', 'Advaita Shatakam', 'A no bhadrAH Suktam',
         'UdakashAnti Mantra', 'Rigveda Mandala 1', 'Rigveda Mandala 2', 'Rigveda Mandala 3',
         'Rigveda Mandala 4', 'Rigveda Mandala 5', 'Rigveda Mandala 6', 'Rigveda Mandala 7',
         'Rigveda Mandala 8', 'Rigveda Mandala 9', 'Rigveda Mandala 10', 'Selected verses from Rigveda',
         'oShadhIsUktam', 'Kumara Suktam', 'khilas 1', 'Ganapati sUkta from Rigveda',
         'Shri Ganapati Atharvashirsha Upanishat or Ganapati Upanishat with Accents',
         'Gosthasukta', 'Go Samuha Suktam', 'First mantra of each Veda', 'Chamakaprashna',
         'Taittiriya AraNyaka', 'Taittiriya Brahmanam', 'Taittiriya Samhita 1', 'Taittiriya Samhita',
         'TaittiriyAranyakam aruNaprashnaH', 'Trisuparna Suktam', 'durgAsUktam', 'devIsukta (Rigveda)',
         'dhanurveda', 'Dhruvasuktam Rigveda', 'Vedokta Sabija Navagraha Mantra Japa Prayogah', 'naShTa dravya prApti sUktam',
         'NakShatrasukta', 'Narayanasukta', 'nAsadIya sUkta (Rigveda )', 'PavamAnasukta', 'Pitrisuktam', 'Purushasukta',
         'Purushasukta from Shuklayajurveda', 'Krityaapariharanasuktam or Bagalamukhisuktam', 'brahmaNaspatisUktam sasvara',
         'Bhagya Suktam or Pratah Suktam', 'shrIbhUsUktam', 'bhUsUktam', 'Mantrapushpa', 'mantrapuShpAnjali', 'Manyu Suktam',
         'Maha Sauram', 'Medhasukta', 'rakShoghna sUkta Rigveda Mandala 4 and 10', 'rAtrisUktam', 'Rashtra Suktam',
         'rudram (praise of Lord Shiva) namakam and chamakam', 'Rudrapanchakam', 'Rudraprashna',
         'Shri Shuklayajurvediya Sasvara Rudrashtadhyayi', 'Shri Shuklayajurvediya Rudrashtadhyayi', 'Varuna Suktam 1',
         'Varuna Suktam 2', 'Vastu Suktam', 'vishvakarmasUktam', 'Shri Vishnu Suktam 2', 'Vishnusuktam', 'Vedamantramanjari 1',
         'Vedamantramanjari 2', 'Vedamantramanjari 3', 'Praise of Vedas from Shrimad Bhagavata Purana Skandha 10 Adhyaya 87',
         'Shantipatha', 'Shasta Suktam', 'Shivapujana Vaidika Shodashopachara', 'Shraddha Suktam', 'shrI sUkta (Rigveda)',
         'Samvada or Akhyana sukta from Rigveda Samhita Mandala 10', 'sanj~nAnasUkta', 'Rigvediya Sandhya Vandana',
         'Shukla YajurvedIya SandhyA Morning-Noon-Evening', 'Samaveda Samhita Kauthuma ShAkha', 'Suryasukta from Rigveda',
         'SaubhagyalakShmi Upanishad', 'Svasti Suktam', 'hiraNyagarbhasUktam']

dir = '/home/ocr/teluguOCR'
text_file_paths = []
for x in names:
  text_file_paths.append(dir + '/html_transcriptions/' + x + '.txt')#path to text file in "html_transcriptions" folder
font_paths = [dir + '/fonts/Nirmala.ttf']

def get_text_dimensions(text_string, font):
    if font.getmask(text_string).getbbox() == None:
      return [0,0]
    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3]

    return [text_width, text_height]

def draw_telugu_text(text, font_path, font_size, text_color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)

    # Get the size of the text
    text_size = get_text_dimensions(text, font)

    height = text_size[1] + 100 #image height plus 10(buffer for clear image)
    width = text_size[0] + 20 #Image width plus 10(buffer for clear image)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Calculate the position to center the text
    x = (img.width - text_size[0]) // 2
    y = (img.height - text_size[1]) // 2

    draw.text((x, y), text, fill=text_color, font=font)
    
    # removing the excess white space on top and bottom
    image = np.array(img)
    non_zeros = []
    for i in range(image.shape[0]):
      if np.sum(image[i]) != 255*image.shape[1]*3:
        non_zeros.append(i)
    
    image = image[non_zeros[0]-5:non_zeros[-1]+5,:,:]
    img = Image.fromarray(image)
  
    return img

acchulu = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఌ', 'ౡ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః']
hallulu = ['క', 'ఖ', 'గ', 'ఘ', 'ఙ',
           'చ', 'ఛ', 'జ', 'ఝ', 'ఞ',
           'ట', 'ఠ', 'డ', 'ఢ', 'ణ',
           'త', 'థ', 'ద', 'ధ', 'న',
           'ప', 'ఫ', 'బ', 'భ', 'మ',
           'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ']
vallulu = ['ా', 'ి', 'ీ', 'ు' , 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', 'ఁ', 'ౕ', 'ౖ', 'ౢ' ]
connector = ['్']
numbers = ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯']

varnmala = acchulu + hallulu + vallulu + connector + numbers + [' '] 

# load the txt file and read the file line by line.
def read_file_lines(filename):
    lines = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())  # Remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return lines

def cleaning_the_text_2(string):
    string = string.strip()
    for x in string:
        if x in varnmala:
            continue
        else:
            string = string.replace(x, '')
    return string

lines = read_file_lines(dir + '/Dataset/labels.txt')
f_str = open(dir + '/Dataset/strings.txt', 'w')

for s in lines:
    s = cleaning_the_text_2(s)
    s = re.sub("\s\s+" , " ", s)
    if(s=='' or s == None or s == ' '):
        continue
    f_str.write(s + '\n')
f_str.close()

start = 0
lines = read_file_lines(dir + '/Dataset/strings.txt')
for ind_s in range(start, len(lines)):
  s = lines[ind_s]
  Img = draw_telugu_text(text = s, font_path = font_paths[0], font_size = 64)
  m = Img.size[1]//40
  Img = Img.resize((Img.size[0]//m, 40))
  cv.imwrite(dir + '/Dataset/Images/Image' + str(ind_s+1) + '.png', np.array(Img))
  del Img
  del m
  del s
  print(ind_s, end = '\r')