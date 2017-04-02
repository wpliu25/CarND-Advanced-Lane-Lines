# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from P4 import *
import sys

def process_image(image):
    result = AdvancedLaneLines(image)
    return result

if __name__ == "__main__":

    if(len(sys.argv) < 3):
        print("Usage: ./P4Video.py <input_file> <output_file>\n")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]
    clip2 = VideoFileClip(input)
    output_clip = clip2.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)