import re
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Paths in our code are designed according to relative path structure in our drive. This can be changed later.

# To sort video names in a directory alphanumerically
def sorted_alphanumeric(data):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  return sorted(data, key=alphanum_key)

# Extract image frames from videos
def extract_frames(video_list, output_dir): 
  os.chdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/JIGSAWS/video_encoded/')
  output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/img_frames/'
  
  for video in video_list:
  vid = video.split('.')[0]
  os.mkdir(output_dir+vid)
  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    success,image = vidcap.read()
    if not success:
      continue
    cv2.imwrite(output_dir + vid + '/frame%d.jpg' % count, image)
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
    count += 1
    
 # Generating color cues
 def generate_color_cue(video_list, output_dir):
   os.chdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/img_frames/')
   output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/color/'

   for video in video_list:
    vid = video.split('.')[0]
    os.mkdir(output_dir+vid)
    images = sorted_alphanumeric(os.listdir(vid))
    count = 0
    for image in images:
      img = cv2.imread(vid+'/'+image, cv2.IMREAD_GRAYSCALE)
      invert_img = abs(255-img)
      cv2.imwrite(output_dir + vid + '/frame%d.jpg' % count, invert_img)
      count += 1

# Generating location cues based on color
def generate_loc_cue(video_list, output_dir):
  os.chdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/color/')
  video_list = os.listdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/color/')
  output_dir = '../location/'
  for video in video_list:
    images = sorted_alphanumeric(os.listdir(video))
    image = cv2.imread(video + '/' + images[0])
    img_sum = np.zeros((image.shape[0], image.shape[1]))
    for image in images:
      img = cv2.imread(video + '/' + image, cv2.IMREAD_GRAYSCALE)
      img_sum += img

    img_sum /= len(images)
    count = 0
    os.mkdir(output_dir+video)
    for image in images:
      cv2.imwrite(output_dir + video + '/frame%d.jpg' % count, img_sum)
      count += 1

# Generating optical flow cue
def generate_of_cue(video_list, output_dir):
input_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/img_frames/'
output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/'
os.mkdir(output_dir + 'optical_flow/')

for video in video_list:
  vid = video.split('.')[0]
  os.mkdir(output_dir+'optical_flow/'+vid)
  images = sorted_alphanumeric(os.listdir(input_dir + vid))
  frame1 = cv2.imread(input_dir + vid +'/'+images[0])
  prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[..., 1] = 255
  count = 0
  for image in images[1:301]:
    frame2 = cv2.imread(input_dir + vid +'/'+image)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = np.where(hsv[..., 2] > 15, 1, 0) * 255
    cv2.imwrite(output_dir + 'optical_flow/' + vid + '/frame%d.jpg' % count, hsv[..., 2])
    prvs = next
    count += 1

# Generating location cue based on optical flow
def generate_loc_of_cue(video_list, output_dir):
  os.chdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/optical_flow/')
  video_list = os.listdir('/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/optical_flow/')
  output_dir = '../location_of/'
  for video in video_list:
    vid = video.split('.')[0]
    images = sorted_alphanumeric(os.listdir(vid))
    image = cv2.imread(video + '/' + images[0])
    img_sum = np.zeros((image.shape[0], image.shape[1]))
    for image in images:
      img = cv2.imread(vid + '/' + image, cv2.IMREAD_GRAYSCALE)
      img_sum += img

    img_sum /= len(images)
    count = 0
    os.mkdir(output_dir+vid)
    for image in images[:301]:
      cv2.imwrite(output_dir + video + '/frame%d.jpg' % count, img_sum)
      count += 1

# Generating anchors based on color and location based on color
def generate_anchors(video_list, output_dir):
input_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/'
output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/'

for video in video_list:
  vid = video.split('.')[0]
  color_cue = 'color/' + vid
  location_cue = 'location/' + vid
  #object_cue = 'objectness/' + vid
  os.mkdir(output_dir+'anchors/pos/'+vid)
  os.mkdir(output_dir+'anchors/neg/'+vid)
  images = sorted_alphanumeric(os.listdir(input_dir + color_cue))
  count = 0
  for image in images:
    color = cv2.imread(input_dir + color_cue +'/'+image, cv2.IMREAD_GRAYSCALE)
    color = np.where(color > 220, 1, 0)
    location = cv2.imread(input_dir+ location_cue +'/'+image, cv2.IMREAD_GRAYSCALE)
    location = np.where(location>180,0,location)
    location = np.where(location<100,0,location)
    location = np.where(location!=0, 1, 0)
    pos = 1.0 * color * location * 255
    neg = 255-pos
    cv2.imwrite(output_dir + 'anchors/pos/' + vid + '/frame%d.jpg' % count, pos)
    cv2.imwrite(output_dir + 'anchors/neg/' + vid + '/frame%d.jpg' % count, neg)
    count += 1

# Generating anchors based on color and location based on optical flow
def generate_anchors_of(video_list, output_dir):
  os.mkdir(output_dir+'anchors_of/')
  os.mkdir(output_dir+'anchors_of/pos')
  os.mkdir(output_dir+'anchors_of/neg')

  input_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/cues/'
  output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/'
  
  for video in video_list:
    vid = video.split('.')[0]
    color_cue = 'color/' + vid
    location_cue = 'location_of/' + vid
    #object_cue = 'objectness/' + vid
    os.mkdir(output_dir+'anchors_of/pos/'+vid)
    os.mkdir(output_dir+'anchors_of/neg/'+vid)
    images = sorted_alphanumeric(os.listdir(input_dir + location_cue))
    count = 0
    for image in images[:301]:                      #Keeping a cap of 300 frames for each video for now
      color = cv2.imread(input_dir + color_cue +'/'+image, cv2.IMREAD_GRAYSCALE)
      if color is None:
        continue
      color = color/255.0
      location = cv2.imread(input_dir+ location_cue +'/'+image, cv2.IMREAD_GRAYSCALE)
      if location is None:
        continue
      location = location/255.0
      pos = 1.0 * color * location * 255
      neg = 255-pos
      cv2.imwrite(output_dir + 'anchors_of/pos/' + vid + '/frame%d.jpg' % count, pos)
      cv2.imwrite(output_dir + 'anchors_of/neg/' + vid + '/frame%d.jpg' % count, neg)
      count += 1


if __name__ == '__main__':
  
  # Hard coded for now, can automate this part using argument parses and config files.
  video_list = ['Knot_Tying_B001_capture1.mp4', 'Knot_Tying_C003_capture1.mp4', 'Knot_Tying_D005_capture1.mp4', 'Knot_Tying_F002_capture2.mp4', 'Knot_Tying_G005_capture2.mp4',
               'Needle_Passing_B001_capture1.mp4', 'Needle_Passing_C003_capture2.mp4', 'Needle_Passing_D002_capture1.mp4', 'Needle_Passing_E005_capture2.mp4', 'Needle_Passing_I004_capture1.mp4',
               'Suturing_B001_capture1.mp4', 'Suturing_C003_capture1.mp4', 'Suturing_D002_capture1.mp4', 'Suturing_F003_capture2.mp4', 'Suturing_H003_capture2.mp4']

  extract_frames(video_list, None)
  
  # Experiment 1 - method according to the AGSD-Surgical-Instrument-Segmentation paper.
  generate_color_cue(video_list, None)
  generate_loc_cue(video_list, None)
  generate_anchors(video_list, None)
  
  #Experiment 2 - our method
  generate_loc_of_cue(video_list, None)
  generate_anchors_of(video_list, None)
    
    
    
    
