# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import cv2
import numpy as np
import copy


class LabellingBBoxState:
    TRACKS = (0, 0, 255)
    SET = (0, 255, 0)
    SELECTED = (255, 0, 0)
    END_OF_TRACK = (0, 215, 255)


class LabellingBBoxDrawingState:
    NONE = 0
    BBOX_CREATE = 1
    BBOX_SLIDES = 2
    BBOX_RESIZE = 3
    BBOX_KEYBOARD = 4


class BBoxBorder:
    NONE = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    LEFT = 5
    RIGHT = 6
    TOP = 7
    BOTTOM = 8


class LabellingBBox:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.id = -1
        self.status = LabellingBBoxState.SELECTED
        self.x_offset = 0
        self.y_offset = 0
        self.init_move_x = 0
        self.init_move_y = 0
        self.class_id = 0
        self.manually_created = True
        self.tracker = None
        self.end_of_track = -1
        self.hidden = False

    def copy(self, dest_bbox):
        self.x = dest_bbox.x
        self.y = dest_bbox.y
        self.width = dest_bbox.width
        self.height = dest_bbox.height
        self.id = dest_bbox.id
        self.status = dest_bbox.status
        self.x_offset = dest_bbox.x_offset
        self.y_offset = dest_bbox.y_offset
        self.init_move_x = dest_bbox.init_move_x
        self.init_move_y = dest_bbox.init_move_y
        self.class_id = dest_bbox.class_id
        self.manually_created = dest_bbox.manually_created
        self.tracker = dest_bbox.tracker
        self.end_of_track = dest_bbox.end_of_track
        self.hidden = dest_bbox.hidden

    def from_tracker_bbox(self, tracker_bbox):
        self.x = int(tracker_bbox[0])
        self.y = int(tracker_bbox[1])
        self.width = int(tracker_bbox[2])
        self.height = int(tracker_bbox[3])

    def print_bbox(self):
        print("bbox", self.id, self.class_id, "[", self.x, self.y, self.width, self.height, "]", self.end_of_track)

    def resize(self, x, y, corner):
        if corner == BBoxBorder.BOTTOM_LEFT:
            tr_x = self.x + self.width
            tr_y = self.y
            self.x = x
            self.width = tr_x - self.x
            self.height = y - tr_y

        elif corner == BBoxBorder.BOTTOM_RIGHT:
            update_bbox(self, x, y)

        elif corner == BBoxBorder.TOP_LEFT:
            br_x = self.x + self.width
            br_y = self.y + self.height
            self.x = x
            self.y = y
            self.width = br_x - self.x
            self.height = br_y - self.y

        elif corner == BBoxBorder.TOP_RIGHT:
            bl_x = self.x
            bl_y = self.y + self.height
            self.y = y
            self.width = x - bl_x
            self.height = bl_y - y

        elif corner == BBoxBorder.TOP:
            b_y = self.y + self.height
            self.y = y
            self.height = b_y - y

        elif corner == BBoxBorder.BOTTOM:
            t_y = self.y
            self.height = y - t_y

        elif corner == BBoxBorder.LEFT:
            r_x = self.x + self.width
            self.x = x
            self.width = r_x - x

        elif corner == BBoxBorder.RIGHT:
            l_x = self.x
            self.width = x - l_x

    def to_tracker_bbox(self):
        return tuple([self.x, self.y, self.width, self.height])

    def from_bbox_info(self, bbox_info):
        self.from_tracker_bbox(bbox_info["bbox"])
        self.class_id = bbox_info["class_id"]
        if "manually_created" not in bbox_info:
            self.manually_created = True
        else:
            self.manually_created = bbox_info["manually_created"]

        if "end_of_track" not in bbox_info:
            self.end_of_track = -1
        else:
            self.end_of_track = bbox_info["end_of_track"]

    def to_label_bbox_info(self):
        return {"class_id": self.class_id,
                "bbox": tuple([self.x,
                               self.y,
                               self.width,
                               self.height]),
                "manually_created": self.manually_created,
                "end_of_track": self.end_of_track}

    def is_null(self, min_size):
        if (self.width * self.width + self.height * self.height) < min_size * min_size:
            return True
        return False

    def self_scale(self, zoom):
        self.x = int(self.x / float(zoom))
        self.y = int(self.y / float(zoom))
        self.width = int(self.width / float(zoom))
        self.height = int(self.height / float(zoom))

    def get_scaled(self, zoom):
        bbox = copy.deepcopy(self)
        bbox.self_scale(zoom)
        return bbox


class FrameLabellingBBoxes:
    def __init__(self):
        self.drawing_state = LabellingBBoxDrawingState.NONE
        self.bbox_selected_idx = -1
        self.bbox_selected = LabellingBBox()
        self.bbox_save_selected = LabellingBBox()
        self.bbox_next_id = 0
        self.bbox_list = []
        self.editing_id = False
        self.zoom = 1
        self.drawing_frame = np.zeros([1, 1])
        self.input_frame = np.zeros([1, 1])
        self.prev_input_frame = np.zeros([1, 1])
        self.tracker_name = "KCF"
        self.autoplay = False
        self.current_class_id = 0
        self.resize_corner = BBoxBorder.NONE
        self.min_bbox_size = 20
        self.cursor = False
        self.cursor_x = 0
        self.cursor_y = 0

    def update_labelling_frame(self, input_frame):
        if not self.prev_input_frame.shape[0] == 1:
            self.prev_input_frame = input_frame
        else:
            self.prev_input_frame = np.copy(self.input_frame)

        self.input_frame = input_frame
        self.drawing_frame = cv2.resize(
            input_frame,
            (input_frame.shape[1] * self.zoom,
             input_frame.shape[0] * self.zoom),
            interpolation=cv2.INTER_AREA)

    def create_tracker_for_new_bboxes(self):
        for bbox in self.bbox_list:
            if not bbox.tracker:
                try:
                    tracker = cv2.Tracker_create(self.tracker_name)
                    if tracker.init(self.prev_input_frame, bbox.to_tracker_bbox()):
                        bbox.tracker = tracker
                        print("bbox " + str(bbox.id) + ": tracker initialized")
                    else:
                        print("Could not initialise tracker for bbox " + str(bbox.id) + ". Bbox removed.")
                except AttributeError:
                    tracker = None

    def find_bbox_list_idx_from_bbox_id(self, bbox_id_to_find):
        list_idx = 0
        for bbox in self.bbox_list:
            if bbox.id == bbox_id_to_find:
                return list_idx
            list_idx += 1
        return -1

    def save_current_bboxes(self, bboxes_container, frame_idx):
        bboxes_container[frame_idx] = {}
        for bbox in self.bbox_list:
            bboxes_container[frame_idx][bbox.id] = bbox.to_label_bbox_info()

    def update_bboxes_from_frame(self, bboxes_container, frame_idx):

        if self.bbox_selected_idx >= 0:
            self.unselect_selected()

        # create bbox container if not existing
        if frame_idx not in bboxes_container:
            bboxes_container[frame_idx] = {}

        # update with manually set bbox
        # remove current bboxes that will be erased by historical ones
        bbox_list_idx_to_delete = []
        for bbox_id, bbox_info in bboxes_container[frame_idx].items():

            if "manually_created" not in bbox_info:
                # bbox exists in the boxes container but is new or read from file i.e. must not be tracked
                bbox_info["manually_created"] = True

            if "end_of_track" not in bbox_info:
                bbox_info["end_of_track"] = -1  # no end of track associated to this bbox

            # if end of track passed: we must erase the bbox
            # if manually created: the manual must override the bbox tracked with the same id so we delete it
            if bbox_info["manually_created"] or (
                    frame_idx >= bbox_info["end_of_track"] and bbox_info["end_of_track"] >= 0):
                bbox_list_idx = self.find_bbox_list_idx_from_bbox_id(bbox_id)
                if bbox_list_idx >= 0:
                    bbox_list_idx_to_delete.append(bbox_list_idx)

        self.delete_bboxes(bbox_list_idx_to_delete)

        # create tracker for new bbox created at previous frame
        self.create_tracker_for_new_bboxes()

        bbox_list_idx = 0
        # reset the bbox list to append the updated frames
        bbox_list_idx_to_delete = []
        for bbox in self.bbox_list:
            if frame_idx >= bbox.end_of_track and bbox.end_of_track >= 0:
                bbox_list_idx_to_delete.append(bbox_list_idx)
                continue

            if bbox.tracker:
                tracked, tracker_bbox = bbox.tracker.update(self.input_frame)
                if not tracked:
                    bbox_list_idx_to_delete.append(bbox_list_idx)
                else:
                    # create the bbox
                    bbox.from_tracker_bbox(tracker_bbox)
                    correct_bbox(bbox, self.prev_input_frame.shape[1], self.prev_input_frame.shape[0])
                    bbox.status = LabellingBBoxState.TRACKS
                    bbox.manually_created = False
            bbox_list_idx += 1

        # delete untracked bbox
        self.delete_bboxes(bbox_list_idx_to_delete)

        # add the manually created bbox
        for bbox_id, bbox_info in bboxes_container[frame_idx].items():
            if bbox_info["manually_created"]:
                # create the bbox
                self.bbox_selected = LabellingBBox()
                self.bbox_selected.from_bbox_info(bbox_info)
                self.bbox_selected.id = bbox_id
                self.bbox_selected.status = LabellingBBoxState.SET
                self.add_selected_bbox(True)
                self.reset_selected()

            # else: bbox_list doesn't change

    def set_bbox_list_from_bboxes_container(self, bboxes_container, frame_idx):

        # delete current bbox on screen
        for bbox in self.bbox_list:
            del bbox

        self.bbox_list = []

        if frame_idx not in bboxes_container:
            bboxes_container[frame_idx] = {}

        for bbox_id, bbox_info in bboxes_container[frame_idx].items():
            if frame_idx >= bbox_info["end_of_track"] and bbox_info["end_of_track"] >= 0:
                continue

            # create the bbox
            self.bbox_selected = LabellingBBox()
            self.bbox_selected.from_bbox_info(bbox_info)
            self.bbox_selected.id = bbox_id
            self.add_selected_bbox(True)

            if self.bbox_selected.manually_created:
                self.bbox_selected.status = LabellingBBoxState.SET
            else:
                self.bbox_selected.status = LabellingBBoxState.TRACKS

            self.reset_selected()

    def update_bbox_id_from_keys(self, key, bboxes_container, frame_idx, frame_idx_step):

        if self.bbox_selected_idx < 0:
            return

        editing_class_id = False
        print("Input id for bbox id")

        if self.editing_id:
            return

        text_prefix = "new object id: "
        text_suffix = ""
        if self.bbox_selected_idx >= 0:
            self.editing_id = True
            id = ""

            if key >= ord('0') and key <= ord('9'):
                id = id + str(key - ord('0'))
            if key >= 176 and key <= 185:  # numpad
                id = id + str(key - 176)

            while self.editing_id:
                input_id_frame = np.zeros(shape=(100, 800))
                key = cv2.waitKey(1) & 0x7f
                if key >= ord('0') and key <= ord('9'):
                    if text_suffix != "":
                        text_suffix = ""
                        id = ""
                    id = id + str(key - ord('0'))

                elif key >= 176 and key <= 185:  # numpad
                    if text_suffix != "":
                        text_suffix = ""
                        id = ""
                    id = id + str(key - 176)

                elif key == ord('C') or key == ord('c'):
                    editing_class_id = not editing_class_id
                    text_suffix = ""
                    if editing_class_id:
                        text_prefix = "new class id: "
                        print("Input id for bbox class id")
                    else:
                        text_prefix = "new object id: "
                        print("Input id for bbox id")

                elif key == 27:  # escape
                    self.editing_id = False
                    self.unselect_selected()

                elif key == 8:  # backspace
                    if len(id) > 0:
                        id = id[:-1]

                elif key == 13 or key == 10:  # enter key
                    if not editing_class_id:
                        if not self.change_bbox_selected_id(int(id), bboxes_container, frame_idx, frame_idx_step):
                            text_suffix = " exists!"
                        else:
                            text_suffix = ""
                            self.approve_selected_bbox()
                            self.editing_id = False
                    else:
                        self.change_bbox_selected_class_id(int(id), bboxes_container)
                        if self.bbox_selected.tracker:
                            self.bbox_selected.status = LabellingBBoxState.TRACKS
                        else:
                            self.bbox_selected.status = LabellingBBoxState.SET
                        self.reset_selected()
                        self.editing_id = False

                cv2.putText(input_id_frame, text_prefix + id + text_suffix,tuple([0, 50]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                cv2.imshow("input new id", input_id_frame)

            cv2.destroyWindow("input new id")

    def change_bbox_selected_id(self, id, bboxes_container, frame_idx, frame_idx_step):

        # check if object already exists with same id in the current frame
        existing_bbox_idx_with_new_id = self.find_bbox_list_idx_from_bbox_id(id)
        if existing_bbox_idx_with_new_id >= 0:
            return 0

        # change contiguous bboxes with same id in future and past
        to_update_frame_idx = frame_idx - frame_idx_step

        while to_update_frame_idx in bboxes_container:
            if self.bbox_selected.id in bboxes_container[to_update_frame_idx]:
                bboxes_container[to_update_frame_idx][id] = copy.deepcopy(
                    bboxes_container[to_update_frame_idx][self.bbox_selected.id])
                del bboxes_container[to_update_frame_idx][self.bbox_selected.id]
                to_update_frame_idx -= frame_idx_step
            else:
                break

        to_update_frame_idx = frame_idx + frame_idx_step

        while to_update_frame_idx in bboxes_container:
            if self.bbox_selected.id in bboxes_container[to_update_frame_idx]:
                bboxes_container[to_update_frame_idx][id] = copy.deepcopy(
                    bboxes_container[to_update_frame_idx][self.bbox_selected.id])
                del bboxes_container[to_update_frame_idx][self.bbox_selected.id]
                to_update_frame_idx += frame_idx_step
            else:
                break

        self.bbox_selected.id = id

        # ensure not to use already used id
        if self.bbox_selected.id >= self.bbox_next_id:
            self.bbox_next_id = self.bbox_selected.id + 1

        return 1

    def change_bbox_selected_class_id(self, id, bboxes_container):
        # change all class id for the selected bbox, past and future
        for frame_idx, bboxes in bboxes_container.items():
            if self.bbox_selected.id in bboxes:
                bboxes[self.bbox_selected.id]["class_id"] = id

        self.current_class_id = id
        self.bbox_selected.class_id = id

    def delete_all_bbox_with_id_of_selected(self, bboxes_container):
        if self.bbox_selected_idx >= 0:
            bbox_id_to_del = self.bbox_selected.id

            # delete all bboxes with same class id as the selected bbox. Delete in past and future
            for frame_idx, bboxes in bboxes_container.items():
                if bbox_id_to_del in bboxes:
                    del bboxes[bbox_id_to_del]

            list_idx = self.find_bbox_list_idx_from_bbox_id(bbox_id_to_del)
            if list_idx >= 0:
                del self.bbox_list[list_idx]
                self.reset_selected()

    def overwrite_all_futur_bbox_with_id_of_selected(self, bboxes_container, frame_idx):
        if self.bbox_selected_idx >= 0:
            bbox_id_to_del = self.bbox_selected.id

            # overwrite all futur bboxes with same class id as the selected bbox i.e.
            # delete bboxes in all the futur frames
            for bboxes_frame_idx, bboxes in bboxes_container.items():
                if bboxes_frame_idx > frame_idx:
                    # deleting futur bbox with same object id so they are overwritten by the user
                    if bbox_id_to_del in bboxes:
                        del bboxes[bbox_id_to_del]

                else:  # bboxes_frame_idx <= frame_idx:
                    # set the end of track to none since it may change by the overwritting
                    if bbox_id_to_del in bboxes:
                        bboxes[bbox_id_to_del]["end_of_track"] = -1

            print("overwritting futur bbox with object id " + \
                str(bbox_id_to_del) + " from frame idx " + str(frame_idx) + ".")
            self.bbox_selected.end_of_track = -1
            self.unselect_selected()

    def stop_tracking_selected_object(self, bboxes_container, frame_idx, frame_idx_step):
        if self.bbox_selected_idx >= 0:
            bbox_id_to_update = self.bbox_selected.id

            # remove all futur contiguous bbox since the object is not tracked anymore until it may reappear
            to_update_frame_idx = frame_idx + frame_idx_step

            while to_update_frame_idx in bboxes_container:
                if bbox_id_to_update in bboxes_container[to_update_frame_idx]:
                    del bboxes_container[to_update_frame_idx][bbox_id_to_update]
                    print("deleting bbox", bbox_id_to_update, "in frame", to_update_frame_idx)
                    to_update_frame_idx += frame_idx_step
                else:
                    break

            to_update_frame_idx = frame_idx - frame_idx_step

            # update the end of track of the previous contiguous bboxes
            while to_update_frame_idx in bboxes_container:
                if bbox_id_to_update in bboxes_container[to_update_frame_idx]:
                    bboxes_container[to_update_frame_idx][bbox_id_to_update]["end_of_track"] = frame_idx
                    print("end of track for bbox", bbox_id_to_update, "in frame", to_update_frame_idx)
                    to_update_frame_idx -= frame_idx_step
                else:
                    break

            print("stopped tracking object id " + str(bbox_id_to_update) + " from frame idx " + str(frame_idx) + ".")
            self.bbox_selected.status = LabellingBBoxState.END_OF_TRACK
            self.bbox_selected.end_of_track = frame_idx
            self.reset_selected()

    def left_arrow_pressed(self, ctrl=False):
        if self.bbox_selected_idx < 0:
            return
        self.drawing_state = LabellingBBoxDrawingState.BBOX_KEYBOARD
        if not ctrl:
            self.bbox_selected.x = self.bbox_selected.x - 1
            self.bbox_selected.width = self.bbox_selected.width + 1
        else:
            self.bbox_selected.width = self.bbox_selected.width - 1

    def right_arrow_pressed(self, ctrl=False):
        if self.bbox_selected_idx < 0:
            return
        self.drawing_state = LabellingBBoxDrawingState.BBOX_KEYBOARD
        if not ctrl:
            self.bbox_selected.x = self.bbox_selected.x + 1
            self.bbox_selected.width = self.bbox_selected.width - 1
        else:
            self.bbox_selected.width = self.bbox_selected.width + 1

    def up_arrow_pressed(self, ctrl=False):
        if self.bbox_selected_idx < 0:
            return
        self.drawing_state = LabellingBBoxDrawingState.BBOX_KEYBOARD
        if not ctrl:
            self.bbox_selected.y = self.bbox_selected.y - 1
            self.bbox_selected.height = self.bbox_selected.height + 1
        else:
            self.bbox_selected.height = self.bbox_selected.height - 1

    def down_arrow_pressed(self, ctrl=False):
        if self.bbox_selected_idx < 0:
            return
        self.drawing_state = LabellingBBoxDrawingState.BBOX_KEYBOARD
        if not ctrl:
            self.bbox_selected.y = self.bbox_selected.y + 1
            self.bbox_selected.height = self.bbox_selected.height - 1
        else:
            self.bbox_selected.height = self.bbox_selected.height + 1

    def change_selected(self, revert=False):

        if self.drawing_state != LabellingBBoxDrawingState.NONE and self.drawing_state != LabellingBBoxDrawingState.BBOX_KEYBOARD:
            return

        if len(self.bbox_list) == 0:
            return

        if self.drawing_state == LabellingBBoxDrawingState.BBOX_KEYBOARD:
            print("save BBOX")
            if not self.approve_selected_bbox():
                self.unselect_selected()
            self.drawing_state = LabellingBBoxDrawingState.NONE

        # purpose here is to not select any bbox after the last one
        if not revert and self.bbox_selected_idx == len(self.bbox_list) - 1:
            self.unselect_selected()
        elif revert and self.bbox_selected_idx == 0:
            self.unselect_selected()
        else:
            if self.bbox_selected_idx < 0:
                if not revert:
                    self.bbox_selected_idx = 0
                else:
                    self.bbox_selected_idx = len(self.bbox_list) - 1
            else:
                if not revert:
                    to_select_id = (self.bbox_selected_idx + 1) % len(self.bbox_list)
                else:
                    to_select_id = (self.bbox_selected_idx - 1) % len(self.bbox_list)
                self.unselect_selected()
                self.bbox_selected_idx = to_select_id

            self.init_bbox_edition(self.bbox_selected_idx)

    def unselect_selected(self):
        if self.bbox_selected_idx >= 0:
            self.bbox_selected.copy(self.bbox_save_selected)
            self.reset_selected()

    def delete_selected(self):
        if self.bbox_selected_idx >= 0:
            print("deleteting bbox " + str(self.bbox_list[self.bbox_selected_idx].id))
            del self.bbox_list[self.bbox_selected_idx]
            self.reset_selected()

    def delete_bbox(self, bbox_idx):
        if len(self.bbox_list) > bbox_idx and bbox_idx >= 0:
            print("deleteting bbox " + str(self.bbox_list[bbox_idx].id))
            del self.bbox_list[bbox_idx]
            self.reset_selected()

    def delete_bboxes(self, bbox_list_idx):
        for bbox_idx in sorted(bbox_list_idx, reverse=True):
            self.delete_bbox(bbox_idx)

    def reset_selected(self):
        self.bbox_selected = LabellingBBox()
        self.bbox_selected_idx = -1
        self.bbox_save_selected = copy.deepcopy(self.bbox_selected)

    def create_new_bbox(self):
        self.bbox_selected.id = self.bbox_next_id
        self.bbox_selected.class_id = self.current_class_id
        self.bbox_selected.status = LabellingBBoxState.SELECTED

    def add_selected_bbox(self, from_existing=False):
        # if bb has a non null size append it
        if not self.bbox_selected.is_null(self.min_bbox_size):
            # deep copy the bounding box to append it to the list
            bbox = copy.deepcopy(self.bbox_selected)

            # append the bbox
            self.bbox_list.append(bbox)
            self.bbox_selected = bbox

            # self_scale bbox
            if not from_existing:
                print("bbox " + str(self.bbox_selected.id) + " created")
                self.approve_selected_bbox()
                self.bbox_next_id += 1
            else:
                correct_bbox(self.bbox_selected, self.input_frame.shape[1], self.input_frame.shape[0])
                print("bbox " + str(self.bbox_selected.id) + " loaded")

    def approve_selected_bbox(self):
        if not self.bbox_selected.is_null(self.min_bbox_size):
            self.bbox_selected.status = LabellingBBoxState.SET
            self.bbox_selected.manually_created = True
            correct_bbox(self.bbox_selected, self.input_frame.shape[1], self.input_frame.shape[0])
            if self.bbox_selected.tracker:
                del self.bbox_selected.tracker
                self.bbox_selected.tracker = None
            self.reset_selected()
            return True
        else:
            return False

    def init_bbox_edition(self, bbox_idx):
        self.bbox_save_selected.copy(self.bbox_list[bbox_idx])
        self.bbox_selected = self.bbox_list[bbox_idx]
        self.bbox_selected.status = LabellingBBoxState.SELECTED
        self.bbox_selected_idx = bbox_idx  # set selected to current bbox

    def hide_bboxes(self, state):
        for bbox in self.bbox_list:
            if bbox.id != self.bbox_selected.id:
                bbox.hidden = state

    def set_cursor(self):
        self.cursor = not self.cursor

    def set_cursor_pos(self, x, y):
        self.cursor_x = x
        self.cursor_y = y

    def draw_bboxes_on_frame(self):

        if self.cursor:
            cv2.line(self.drawing_frame, tuple([0, self.cursor_y]), tuple(
                [self.drawing_frame.shape[1], self.cursor_y]), (255, 255, 255), 1)
            cv2.line(self.drawing_frame, tuple([self.cursor_x, 0]), tuple(
                [self.cursor_x, self.drawing_frame.shape[0]]), (255, 255, 255), 1)

        for bbox in self.bbox_list:
            bbox_to_draw = bbox.get_scaled(1. / float(self.zoom))
            if bbox_to_draw.hidden:
                status_color = tuple([color_val * 1 / 2 for color_val in bbox_to_draw.status])
            else:
                status_color = bbox_to_draw.status

            cv2.rectangle(self.drawing_frame,
                          tuple([bbox_to_draw.x,
                                 bbox_to_draw.y]),
                          (bbox_to_draw.x + bbox_to_draw.width,
                           bbox_to_draw.y + bbox_to_draw.height),
                          status_color,
                          2)
            cv2.putText(self.drawing_frame, "[" +
                        str(bbox_to_draw.id) +
                        "," +
                        str(bbox_to_draw.class_id) +
                        "]", tuple([bbox_to_draw.x, bbox_to_draw.y -
                                    20]), cv2.FONT_HERSHEY_SIMPLEX, 2, status_color)

        if self.drawing_state:
            bbox_to_draw = self.bbox_selected.get_scaled(1. / float(self.zoom))
            cv2.rectangle(self.drawing_frame,
                          tuple([bbox_to_draw.x,
                                 bbox_to_draw.y]),
                          (bbox_to_draw.x + bbox_to_draw.width,
                           bbox_to_draw.y + bbox_to_draw.height),
                          bbox_to_draw.status,
                          2)
            cv2.putText(self.drawing_frame, "[" +
                        str(bbox_to_draw.id) +
                        "," +
                        str(bbox_to_draw.class_id) +
                        "]", tuple([bbox_to_draw.x, bbox_to_draw.y -
                                    20]), cv2.FONT_HERSHEY_SIMPLEX, 2, bbox_to_draw.status)


def is_click_in_bbox(bbox, x, y):
    if bbox.hidden:
        return False

    if x >= bbox.x and x <= bbox.x + bbox.width:
        if y >= bbox.y and y <= bbox.y + bbox.height:
            return True

    return False


def is_click_on_border(bbox, x, y):

    if bbox.hidden:
        return False

    on_left = abs(bbox.x - x) < 3
    on_right = abs(bbox.x + bbox.width - x) < 3
    on_top = abs(bbox.y - y) < 3
    on_bottom = abs(bbox.y + bbox.height - y) < 3

    # corner
    if on_left and on_top:
        return BBoxBorder.TOP_LEFT
    if on_left and on_bottom:
        return BBoxBorder.BOTTOM_LEFT
    if on_right and on_top:
        return BBoxBorder.TOP_RIGHT
    if on_right and on_bottom:
        return BBoxBorder.BOTTOM_RIGHT

    # vertex
    if on_right:
        return BBoxBorder.RIGHT
    if on_left:
        return BBoxBorder.LEFT
    if on_top:
        return BBoxBorder.TOP
    if on_bottom:
        return BBoxBorder.BOTTOM

    return BBoxBorder.NONE


def get_bbox_idx_from_click(bbox_list, x, y):

    bbox_idx = 0
    for bbox in bbox_list:
        if is_click_in_bbox(bbox, x, y):
            return bbox_idx
        bbox_idx += 1
    return -1


def correct_bbox(bbox, frame_width, frame_height):

    if bbox.width < 0:
        bbox.width *= -1
        bbox.x -= bbox.width

    if bbox.height < 0:
        bbox.height *= -1
        bbox.y -= bbox.height

    if bbox.x + bbox.width > (frame_width - 1):
        bbox.width = frame_width - 1 - bbox.x

    if bbox.y + bbox.height > (frame_height - 1):
        bbox.height = frame_height - 1 - bbox.y

    if bbox.x < 0:
        bbox.width = bbox.width - abs(bbox.x)
        bbox.x = 0
    if bbox.y < 0:
        bbox.height = bbox.height - abs(bbox.y)
        bbox.y = 0


def update_bbox(bbox, x, y):
    bbox.width = x - bbox.x
    bbox.height = y - bbox.y


def init_move_bbox(bbox, x, y):
    if not is_click_in_bbox(bbox, x, y):
        bbox.x_offset = 0
        bbox.y_offset = 0
        return

    bbox.init_move_x = x
    bbox.init_move_y = y
    bbox.x_offset = bbox.x - x
    bbox.y_offset = bbox.y - y


def move_bbox(bbox, x, y):
    bbox.x = x + bbox.x_offset
    bbox.y = y + bbox.y_offset


def labelling_mouse_cb(event, x, y, flags, frame_labelling_bboxes):

    if frame_labelling_bboxes.editing_id:
        return

    if event:
        frame_labelling_bboxes.autoplay = False

    frame_labelling_bboxes.set_cursor_pos(x, y)

    x = x / frame_labelling_bboxes.zoom
    y = y / frame_labelling_bboxes.zoom

    if event == cv2.EVENT_MOUSEMOVE:
        # change br corner of bbox and update the bounding box accordingly
        if frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_CREATE:
            update_bbox(frame_labelling_bboxes.bbox_selected, x, y)

        elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_RESIZE:
            frame_labelling_bboxes.bbox_selected.resize(x, y, frame_labelling_bboxes.resize_corner)

        # slide bbox if in slide mode
        elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_SLIDES:
            move_bbox(frame_labelling_bboxes.bbox_selected, x, y)

    elif event == cv2.EVENT_LBUTTONDOWN and not frame_labelling_bboxes.drawing_state:
        # check if click is inside an already created bbox
        bbox_idx = get_bbox_idx_from_click(frame_labelling_bboxes.bbox_list, x, y)

        # if no sleection and click outside: creating new bbox
        if bbox_idx < 0:
            if frame_labelling_bboxes.bbox_selected_idx == -1:
                # create new bbox
                frame_labelling_bboxes.create_new_bbox()

            frame_labelling_bboxes.drawing_state = LabellingBBoxDrawingState.BBOX_CREATE

            # set new bbox position
            frame_labelling_bboxes.bbox_selected.width = 0
            frame_labelling_bboxes.bbox_selected.height = 0
            frame_labelling_bboxes.bbox_selected.x = x
            frame_labelling_bboxes.bbox_selected.y = y

        else:
            if frame_labelling_bboxes.bbox_selected_idx >= 0 and frame_labelling_bboxes.bbox_selected_idx != bbox_idx:
                frame_labelling_bboxes.unselect_selected()

            corner = is_click_on_border(frame_labelling_bboxes.bbox_list[bbox_idx], x, y)
            if corner != BBoxBorder.NONE:
                frame_labelling_bboxes.init_bbox_edition(bbox_idx)
                frame_labelling_bboxes.drawing_state = LabellingBBoxDrawingState.BBOX_RESIZE
                frame_labelling_bboxes.resize_corner = corner

            elif frame_labelling_bboxes.bbox_selected_idx != bbox_idx:
                frame_labelling_bboxes.init_bbox_edition(bbox_idx)
                frame_labelling_bboxes.drawing_state = LabellingBBoxDrawingState.BBOX_SLIDES
                init_move_bbox(frame_labelling_bboxes.bbox_selected, x, y)
            else:
                frame_labelling_bboxes.drawing_state = LabellingBBoxDrawingState.BBOX_CREATE

                # set new bbox position
                frame_labelling_bboxes.bbox_selected.width = 0
                frame_labelling_bboxes.bbox_selected.height = 0
                frame_labelling_bboxes.bbox_selected.x = x
                frame_labelling_bboxes.bbox_selected.y = y

    elif event == cv2.EVENT_LBUTTONUP:

        # check if click is inside an already created bbox
        bbox_idx = get_bbox_idx_from_click(frame_labelling_bboxes.bbox_list, x, y)
        # if click is inside an existing bbox
        if bbox_idx >= 0 and bbox_idx != frame_labelling_bboxes.bbox_selected_idx:
            # no bbox selected: we select the bounding box for editing
            if frame_labelling_bboxes.bbox_selected_idx == -1:
                if frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_CREATE:
                    update_bbox(frame_labelling_bboxes.bbox_selected, x, y)

                    # new bounding box
                    if frame_labelling_bboxes.bbox_selected_idx == -1:
                        frame_labelling_bboxes.add_selected_bbox()
                else:
                    frame_labelling_bboxes.init_bbox_edition(bbox_idx)
                    print("selecting bbox " + str(frame_labelling_bboxes.bbox_selected.id))
            else:
                # is sliding.
                if frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_SLIDES:
                    if frame_labelling_bboxes.bbox_selected.init_move_x != x and frame_labelling_bboxes.bbox_selected.init_move_y != y:
                        frame_labelling_bboxes.approve_selected_bbox()
                    else:
                        print("selecting bbox " + str(frame_labelling_bboxes.bbox_selected.id))

                elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_RESIZE:
                    if not frame_labelling_bboxes.approve_selected_bbox():
                        frame_labelling_bboxes.unselect_selected()

                else:
                    frame_labelling_bboxes.unselect_selected()
                    frame_labelling_bboxes.init_bbox_edition(bbox_idx)
                    print("selecting bbox " + str(frame_labelling_bboxes.bbox_selected.id))

        elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_CREATE:
            update_bbox(frame_labelling_bboxes.bbox_selected, x, y)

            # new bounding box
            if frame_labelling_bboxes.bbox_selected_idx == -1:
                frame_labelling_bboxes.add_selected_bbox()
            else:
                frame_labelling_bboxes.approve_selected_bbox()
                # is sliding.
        elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_SLIDES:
            if frame_labelling_bboxes.bbox_selected.init_move_x != x or frame_labelling_bboxes.bbox_selected.init_move_y != y:
                frame_labelling_bboxes.approve_selected_bbox()
            else:
                print("selecting bbox " + str(frame_labelling_bboxes.bbox_selected.id))

        elif frame_labelling_bboxes.drawing_state == LabellingBBoxDrawingState.BBOX_RESIZE:
            frame_labelling_bboxes.bbox_selected.resize(x, y, frame_labelling_bboxes.resize_corner)
            if not frame_labelling_bboxes.approve_selected_bbox():
                frame_labelling_bboxes.unselect_selected()

        frame_labelling_bboxes.drawing_state = LabellingBBoxDrawingState.NONE
        frame_labelling_bboxes.resize_corner = BBoxBorder.NONE

    elif event == cv2.EVENT_RBUTTONUP:

        if frame_labelling_bboxes.bbox_selected_idx >= 0:
            return

        # check if click is inside an already created bbox
        bbox_idx = get_bbox_idx_from_click(frame_labelling_bboxes.bbox_list, x, y)

        # if no bbox is currently selected and click is inside an existing bbox: we delete the selcting bbox
        if bbox_idx >= 0:
            frame_labelling_bboxes.delete_bbox(bbox_idx)
