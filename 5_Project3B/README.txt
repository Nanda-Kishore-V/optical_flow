In order to run the code, it is required to install skimage-video and ffmpeg.

Use the following commands.

sudo apt-get install ffmpeg
pip3 install scikit-video

run the code with the following command:

python3 object_tracking.py


This command will output the easy video in the current state and save
the output to a file Easy.avi. Our current outputs of tracking for
easy and medium are saved to two files. One of the files is Easy_output.avi
and the second is Medium_output.avi. View those files to see the results of
our tracking. In the code, if you want to test the medium video, replace all
hardcoded values of Easy or easy with Medium or medium. This should be
3 total replacements of these words.
