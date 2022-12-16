# Sample output dir and video_list example based on our code
#output_dir = '/content/drive/MyDrive/CV/SurgicalSkillAssesment/data/img_frames/'
"""video_list = ['Knot_Tying_B001_capture1.mp4', 'Knot_Tying_C003_capture1.mp4', 'Knot_Tying_D005_capture1.mp4', 'Knot_Tying_F002_capture2.mp4', 'Knot_Tying_G005_capture2.mp4', 
               'Needle_Passing_C003_capture2.mp4', 'Needle_Passing_D002_capture1.mp4', 'Needle_Passing_E005_capture2.mp4', 'Needle_Passing_I004_capture1.mp4', 
               'Suturing_B001_capture1.mp4', 'Suturing_C003_capture1.mp4', 'Suturing_D002_capture1.mp4', 'Suturing_F003_capture2.mp4', 'Suturing_H003_capture2.mp4']"""


def(output_dir, video_list): 

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
