Use "segmentation_info" folder to segment the videos into appropriate labels. 

Each xlsx file corresponds to a participant's video (same name in mp4 format)

xlsx file is in the following format:

Column A is genre label (1 = classical, 2 = neutral, 3 = pop)

Column B is song label (song 1-4 are classical, 5-8 are instrumental, 9-12 are pop)

Column C and D are start_time and end_time respectively, in unix time

How to segment the videos:

The video start time is same as the first start_time in corresponding xlsx file e.g. 1526964945 unix time in P01.xlsx is sec 1 of P01.mp4 

Every 4 frames of the video corresponds to 1 unix timestamp. e.g. in P01.mp4, frame 1-4 = 1526964945, frame 5-8 = 1526964946 and so on..

Use this info to segment the videos according to genre or song labels. 

The videos were constructed based on participants physiological response, details are in the paper. 