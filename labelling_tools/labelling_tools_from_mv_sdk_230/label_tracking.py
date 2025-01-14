"""Copyright (c) Prophesee S.A.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License."""

import sys
import os
import datetime
import glob
import argparse
from labelling_bbox import cv2, np, FrameLabellingBBoxes, LabellingBBoxDrawingState, labelling_mouse_cb
import label
from bbox_txt2npy import bboxstr2array

# versioning info
__COMMITID__ = '416c4e5cff4b676f5d8f2da33368be32fbdbcecf'
__OSINFO__ = 'Ubuntu 18.04.3 LTS'
__DATE__ = '2020-01-16'
__ARCHINFO__ = 'x86_64'


def read_image(frame_index):
    """read image"""
    global image_dir
    frame = cv2.imread(image_dir[frame_index])
    return frame is not None, frame


def read_video_frame(frame_index):
    """read video frame"""
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = video.read()
    return ok, frame


def read_frame(frame_index):
    """read frame"""
    if video:
        return read_video_frame(frame_index)
    elif image_dir:
        return read_image(frame_index)
    else:
        raise ValueError(f"nothing to read in the frame:{frame_index}")


def usage(args):
    """print usage"""
    print("\n")
    print("--- Image Legend ---")
    print("\n")
    print("- Green: Manually set bbox")
    print("- Blue: BBox selected for edition or being edited")
    print("- [object_id, class_id]: Id couple on top left of a bounding box")
    print("\n")
    print("--- Functionalities ---")
    print("\n")
    print("- Hold left click and move the mouse on the scene image to draw a bbox.")
    print("- Left click on an existing bbox to select it for editing")
    print("- Hold left click on one of the corner an existing bbox to resize it")
    print("- Hold left click on one of the border an existing bbox to resize it")
    print("- Hold left click on an existing bbox then move the mouse to slide it")
    print("- Right click on an existing bbox to delete it")
    print("- Tab key to select next bbox for edition")
    print("- CTRL + Tab key to select previous bbox for edition")
    print("- D key to delete the selected bbox")
    print("- U key to delete all bbox in the sequence with the same object id as the selected bbox")
    print("- O key to overwrite all the future bbox with the same object id as the selected bbox.")
    print("- S key to stop tracking the selected bbox")
    print("- H key to hide bboxes")
    print("- V key to activate cursor visor")
    print("- Esc key to unselect the selected bbox")
    print("- Space key to run or pause autoplay mode")
    print("- R key to go to next frame. Pauses auto play mode")
    print("- E key to go to previous frame. Pauses autoplay mode")
    print("- C key to use precise cursor")
    print("- K key to print this")
    print("- I key to save current image as png")
    print("- + or - keys to respectively increase or decrease the time between each frame in auto play mode")
    print("\n")
    print("--- BBox edition with arrows (when bbox selected) ---")
    print("\n")
    print("- Left/Right Arrow               : move the left side of the BBox")
    print("- Up/Down Arrow                  : move the top side of the BBox")
    print("- CTRL + Left/CTRL + Right Arrow : move the right side of the BBox")
    print("- CTRL + Up/CTRL + Down Arrow    : move the bottom side of the BBox")
    print("\n")
    print("--- Change ID mode ---")
    print("\n")
    print("- When a bbox is selected, pressing a number key puts you in the change id mode.")
    print("- As soon as a number is pressed while the bbox is selected, you may change the object or class id of a bbox.")
    print("- A small window opens up with the first number you type")
    print("- As long as you keep pressing numbers, it will write them on this window.")
    print("- It represents the id that will replace the current one of the bbox")
    print("- Pressing the C key changes the id you will update: when you enter the mode, it is set to change the object_id.")
    print("- Pressing the C key will switch between changing the class_id and the object_id")
    print("- Changing the class id of a bbox will affect every bbox that has the same class id, in the future or in the past")
    print("- Changing the object id of a bbox will affect every time contiguous bbox that has the same object id, in the future or in the past")
    print("- You can't give a bbox the same object id as a bbox existing in the current frame. The script will prevent it.")
    print("- Pressing the Esc key cancel the id modification")
    print("- Pressing the Backspace key erase the last number entered")
    print("- Pressing the Enter key validates the id modification")
    print("\n\n")

    print("--- Input parameters ---")
    print("\n")
    print("Input video file: " + args.input)
    print("Output label file: " + args.output_file)
    if args.label_file != "":
        print("Input label file: " + args.label_file)
    if args.fps != 200:
        print("video fps - label frequency: " + str(args.fps) + " images per second")
    print("Video begins at frame " + str(args.frame_index))
    print("Minimum bbox size: " + str(args.minimum_size) + " pixels diagonal")


def parse_args(argv=None):
    """parse arguments"""
    parser = argparse.ArgumentParser(description='Label Tracking script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=f"PROPHESEE Label tracking tool - version {__COMMITID__} - created on {__OSINFO__} {__ARCHINFO__} - {__DATE__}")
    parser.add_argument("-i", "--input", required=True, help="Input video file or first image path of a directory.")
    parser.add_argument("-l", "--label_file", default="", help="Existing labeling file")
    parser.add_argument(
                        "-o",
                        "--output_file",
                        default="",
                        help="Output file where labels are written. By default, the output filename generated is "
                             "[input_filename]_labels.txt and then the labels will be converted to"
                             " a npy file [input_filename]_bbox.npy")
    parser.add_argument("-f", "--fps", type=int, default=200, help="Video fps. By default, 200")
    parser.add_argument("-s", "--frame_index", type=int, default=1,
                        help="Frame index to begin with. Default 1. Auto set in range [1, n_frames].")
    parser.add_argument("-m", "--minimum_size", type=int, default=20,
                        help="Bbox minimum diagonal size in pixels. Default 20")
    parser.add_argument("--overwrite", help="if set overwrite output file if exists", action="store_true")
    args = parser.parse_args(argv)

    return args



def main(args):
    """main function"""

    image_dir = None
    global video
    video = None

    assert os.path.isfile(args.input), f"{args.input} is not a file\n," \
                                       f" an AVI or JPG file of the first time frame is required!"
    input_data = os.path.splitext(args.input)
    assert input_data[1].lower() in [".jpg", ".avi"], "The labeling tool only works with AVI and JPG format"

    if input_data[1].lower() == ".avi":
        video_file = args.input
    elif input_data[1].lower() == ".jpg":
        image_dir = sorted(glob.glob(os.path.dirname(args.input) + "/*jpg"))
        assert len(image_dir) > 0, f"Could not initialize from image directory at: {args.input}"
        for img in image_dir:
            print(img)

    if not args.output_file:
        args.output_file = input_data[0] + "_labels.txt"

    if os.path.exists(args.output_file) and (not args.overwrite):
        args.output_file = os.path.splitext(
            args.output_file)[0] + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S") + ".txt"

    # Parameters
    zoom_width = 1500.
    zoom_height = 1200.

    # Read video
    if video_file:
        video = cv2.VideoCapture(video_file)

        # Exit if video not opened.
        assert video.isOpened(), f"Error: Could not open video: {video_file}"

        number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    elif image_dir:
        number_of_frames = len(image_dir)
        img = cv2.imread(image_dir[0])
        frame_width = img.shape[1]
        frame_height = img.shape[0]

    timestep_us = int(1000000 / args.fps)
    assert timestep_us >0, f"Computed frame timestep (us) is :{timestep_us}, which is not a correct timestep_us. " \
                           "Please set a correct frame rate manually with -f option." \
                           "Note: frame time interval is computed as: 1000000/fps"

    frame_index = args.frame_index - 1
    prev_index = frame_index - 1
    if frame_index > number_of_frames:
        frame_index = number_of_frames - 1
    if frame_index < 0:
        frame_index = 0

    args.frame_index = frame_index + 1

    if args.minimum_size < 0:
        args.minimum_size = 20

    # initilize variables
    save_image = False
    window_title = 'labelling: ' + args.input
    bboxes_container = {}

    leave_labelling = False
    frame = np.zeros(shape=(0, 0))
    frame_labelling_bboxes = FrameLabellingBBoxes()
    frame_labelling_bboxes.min_bbox_size = args.minimum_size
    min_autoplay_speed = 20
    autoplay_speed = min_autoplay_speed

    if frame_width > frame_height and zoom_width > frame_width:
        frame_labelling_bboxes.zoom = int(zoom_width / frame_width)
    elif frame_height > frame_width and zoom_height > frame_height:
        frame_labelling_bboxes.zoom = int(zoom_height / frame_height)

    frame_labelling_bboxes.tracker_name = "KCF"

    if args.label_file != "":
        assert os.path.exists(args.label_file), f"Could not open input label file: {args.label_file}. "

        # reading labelling file
        bboxes_container = label.read_bboxes(args.label_file, [], timestep_us)

        # changing output name if input and output are the same
        if ((args.label_file == args.output_file) and (not args.overwrite)):
            args.output_file = os.path.splitext(
                args.label_file)[0] + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S") + ".txt"

        bbox_next_available_id = 1
        for frame_time_idx, bboxes in bboxes_container.items():
            for bbox_id, bbox in bboxes.items():
                if bbox_id >= bbox_next_available_id:
                    bbox_next_available_id = bbox_id + 1

        # Read a new frame
        ok, frame = read_frame(frame_index)
        if not ok:
            print("Could not read first frame of input video")
            sys.exit(1)

        frame_labelling_bboxes.update_labelling_frame(frame)
        frame_labelling_bboxes.set_bbox_list_from_bboxes_container(bboxes_container, timestep_us)
        frame_labelling_bboxes.bbox_next_id = bbox_next_available_id
        frame_labelling_bboxes.unselect_selected()


    usage(args)

    # creating the windows
    cv2.namedWindow(window_title, cv2.WINDOW_GUI_NORMAL)

    # set the mouse callback
    cv2.setMouseCallback(window_title, labelling_mouse_cb, frame_labelling_bboxes)

    # begins processing
    frame_labelling_bboxes.update_bboxes_from_frame(bboxes_container, (frame_index + 1) * timestep_us)
    hide = False

    while not leave_labelling:

        # Read a new frame
        #     tracked = False
        if frame_index != prev_index:
            ok, frame = read_frame(frame_index)
            prev_index = frame_index
            if not ok:
                break

        frame_labelling_bboxes.update_labelling_frame(frame)
        # key bindings

        ret_code = cv2.waitKeyEx(autoplay_speed)
        key = ret_code & 0xFF

        # BBOXES key bindings
        if ret_code & 0xFFFF == 0xFF51:  # left
            if not ret_code & 0x040000:
                frame_labelling_bboxes.left_arrow_pressed()
            else:
                frame_labelling_bboxes.left_arrow_pressed(True)
        elif ret_code & 0xFFFF == 0xFF52:  # up
            if not ret_code & 0x040000:
                frame_labelling_bboxes.up_arrow_pressed()
            else:
                frame_labelling_bboxes.up_arrow_pressed(True)
        elif ret_code & 0xFFFF == 0xFF53:  # right
            if not ret_code & 0x040000:
                frame_labelling_bboxes.right_arrow_pressed()
            else:
                frame_labelling_bboxes.right_arrow_pressed(True)
        elif ret_code & 0xFFFF == 0xFF54:  # down
            if not ret_code & 0x040000:
                frame_labelling_bboxes.down_arrow_pressed()
            else:
                frame_labelling_bboxes.down_arrow_pressed(True)
        elif key == ord('D') or key == ord('d') or key == 127:  # 127 = del
            frame_labelling_bboxes.delete_selected()

        elif key == 9:  # tab key*
            if ret_code == 0x140009:
                frame_labelling_bboxes.change_selected(True)
            else:
                frame_labelling_bboxes.change_selected()
        elif key == 27:  # escape
            frame_labelling_bboxes.unselect_selected()

        elif (key >= ord('0') and key <= ord('9')) or (key >= 176 and key <= 185):  # num pad
            frame_labelling_bboxes.update_bbox_id_from_keys(
                key, bboxes_container, (frame_index + 1) * timestep_us, timestep_us)

        elif key == 32:  # space
            # doesnt allow auto play if drawing_state
            if frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.NONE:
                frame_labelling_bboxes.autoplay = not frame_labelling_bboxes.autoplay

        elif key == ord('I') or key == ord('i'):
            save_image = True

        elif key == 43 or key == 171:  # +
            autoplay_speed -= 10
            if autoplay_speed < min_autoplay_speed:
                autoplay_speed = min_autoplay_speed

        elif key == 45 or key == 173:  # -
            autoplay_speed += 10

        elif key == ord('Q') or key == ord('q'):
            leave_labelling = True

        elif key == ord('U') or key == ord('u'):
            frame_labelling_bboxes.delete_all_bbox_with_id_of_selected(bboxes_container)

        elif key == ord('O') or key == ord('o'):
            frame_labelling_bboxes.overwrite_all_futur_bbox_with_id_of_selected(
                bboxes_container, (frame_index + 1) * timestep_us)

        elif key == ord('S') or key == ord('s'):
            frame_labelling_bboxes.stop_tracking_selected_object(
                bboxes_container, (frame_index + 1) * timestep_us, timestep_us)

        elif key == ord('H') or key == ord('h'):
            hide = not hide
            frame_labelling_bboxes.hide_bboxes(hide)

        elif key == ord('K') or key == ord('k'):
            usage(args)

        elif key == ord('C') or key == ord('c'):
            frame_labelling_bboxes.set_cursor()

        elif key == ord('E') or key == ord('e'):
            frame_labelling_bboxes.autoplay = False
            frame_labelling_bboxes.save_current_bboxes(bboxes_container, (frame_index + 1) * timestep_us)
            frame_index -= frame_index > 0
            frame_labelling_bboxes.set_bbox_list_from_bboxes_container(bboxes_container,
                                                                       (frame_index + 1) * timestep_us)
            print(f'frame {frame_index + 1} out of {number_of_frames}')

        elif key == ord('R') or key == ord('r') or frame_labelling_bboxes.autoplay:
            if (key == ord('R') or key == ord('r')) and frame_labelling_bboxes.autoplay:
                frame_labelling_bboxes.autoplay = False
            frame_labelling_bboxes.save_current_bboxes(bboxes_container, (frame_index + 1) * timestep_us)
            frame_index += frame_index < number_of_frames - 1
            frame_labelling_bboxes.update_bboxes_from_frame(bboxes_container, (frame_index + 1) * timestep_us)
            print(f'frame {frame_index + 1} out of {number_of_frames}')

        # Display result
        frame_labelling_bboxes.draw_bboxes_on_frame()
        cv2.imshow(window_title, frame_labelling_bboxes.drawing_frame)

        if save_image:
            print(f'Saving image image{frame_index}.png')
            cv2.imwrite(f'image{frame_index}.png', frame)
            save_image = False

    frame_labelling_bboxes.save_current_bboxes(bboxes_container, (frame_index + 1) * timestep_us)
    label.write_bboxes(args.output_file, bboxes_container)
    print("exiting at frame " + str(frame_index))
    print("output label file in txt: " + args.output_file)

    # convert to NPY
    print(f'converting to NPY format: {args.output_file[:-4]}_bbox.npy')
    lines = open(args.output_file, "r").readlines()
    bboxes_array = bboxstr2array(lines)
    np.save(args.output_file[:-4] + "_bbox.npy", bboxes_array)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
