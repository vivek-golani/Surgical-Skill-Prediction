import re
import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from extract_frames import extract_frames



if __name__ == '__main__':
  
  # Hard coded for now, can automate this part using argument parses and config files.
  video_list = ['Knot_Tying_B001_capture1.mp4', 'Knot_Tying_C003_capture1.mp4', 'Knot_Tying_D005_capture1.mp4', 'Knot_Tying_F002_capture2.mp4', 'Knot_Tying_G005_capture2.mp4',
               'Needle_Passing_B001_capture1.mp4', 'Needle_Passing_C003_capture2.mp4', 'Needle_Passing_D002_capture1.mp4', 'Needle_Passing_E005_capture2.mp4', 'Needle_Passing_I004_capture1.mp4',
               'Suturing_B001_capture1.mp4', 'Suturing_C003_capture1.mp4', 'Suturing_D002_capture1.mp4', 'Suturing_F003_capture2.mp4', 'Suturing_H003_capture2.mp4']
