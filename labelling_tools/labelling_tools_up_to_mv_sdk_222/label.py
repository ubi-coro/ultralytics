# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import os
import numpy
import warnings


def write_bboxes(filename, bboxes, redundant=True, full_protocol=False,
                 tracked=True, delta_t=100000, default_class_id=0, header=""):
    """ This function assumes that bboxes is a dictionary of dictionaries
    of dictionaries, where the first key is the timestamp the second key
    is the object_id and each dictionary is of the form:

    {class_id": id, "bbox": (x,y,width,height)}

    an example of bboxes might be:

    {1000: {1: {"class_id": 0, "bbox": (20,20,10,10)}},
     2000: {1: {"class_id": 0, "bbox": (20,20,10,10)},
            2: {"class_id": 0, "bbox": (40,20,10,10)}},
     3000: {2: {"class_id": 0, "bbox": (50,20,10,10)}},
     .
     .
     .
     }

    If there are sequences of frames that do not contain bboxes, the first
    frame of the sequence should be marked as an empty dictionary

    if tracked is set to false a BB_DELETE is generated after a delta_t amount of milliseconds
    """
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    file = open(filename, "w")
    if header:
        file.write(header)

    last_bbox = {}
    bbox_buffer = {}  # dictionary mapping timestamps to corresponding bb_delete in untracked format
    for i in sorted(bboxes):

        if not bboxes[i]:
            if last_bbox:
                for id_obj in last_bbox:
                    command = str(i).zfill(10) + " " + str(id_obj) + " BB_DELETE\n"
                    file.write(command)
            last_bbox.clear()
        else:
            if not tracked:
                # we clear the BB_DELETE that come before timestamp (=i)
                bbox_to_delete = list(filter(lambda k: k < i, sorted(bbox_buffer.keys())))
                for ts in bbox_to_delete:
                    for id_ in bbox_buffer[ts]:
                        file.write("{:010d} {:d} BB_DELETE\n".format(ts, id_))
                    bbox_buffer.pop(ts)
            bbox_buffer[int(i + delta_t)] = {}
            for id_obj in bboxes[i]:
                command = str(i).zfill(10) + " " + str(id_obj) + " BB_"

                new_bbox = (0, 0, 0, 0)
                if 'bbox' in bboxes[i][id_obj]:
                    new_bbox = tuple(map(float, bboxes[i][id_obj]['bbox']))
                elif len(bboxes[i][id_obj]) == 4:
                    new_bbox = tuple(map(float, bboxes[i][id_obj]))
                else:
                    continue

                probability = 1
                if 'probability' in bboxes[i][id_obj]:
                    probability = float(bboxes[i][id_obj]['probability'])
                class_id = default_class_id
                if 'class_id' in bboxes[i][id_obj]:
                    class_id = bboxes[i][id_obj]['class_id']

                if id_obj in last_bbox:
                    bbox_diff = numpy.subtract(new_bbox, last_bbox[id_obj])
                    if (bbox_diff[0] != 0 or bbox_diff[1] != 0) and \
                            (bbox_diff[2] != 0 or bbox_diff[3] != 0):
                        command += "MOVE_AND_RESIZE {:f} {:f} {:f} {:f} {:f}\n".format(*(new_bbox + (probability, )))
                    elif (bbox_diff[0] != 0 or bbox_diff[1] != 0):
                        if full_protocol:
                            command += "MOVE {:f} {:f} {:f}\n".format(*(new_bbox[0:2] + (probability, )))
                        else:
                            command += "MOVE_AND_RESIZE {:f} {:f} {:f} {:f} {:f}\n".format(
                                *(new_bbox + (probability, )))
                    elif (bbox_diff[2] != 0 or bbox_diff[3] != 0):
                        if full_protocol:
                            command += "RESIZE {:f} {:f} {:f}\n".format(*(new_bbox[2:4] + (probability, )))
                        else:
                            command += "MOVE_AND_RESIZE {:f} {:f} {:f} {:f} {:f}\n".format(
                                *(new_bbox + (probability, )))
                    elif redundant:
                        command += "MOVE_AND_RESIZE {:f} {:f} {:f} {:f} {:f}\n".format(*(new_bbox + (probability, )))
                    else:
                        continue

                    last_bbox[id_obj] = new_bbox
                else:
                    command += "CREATE " + str(class_id) + \
                        " {:f} {:f} {:f} {:f} {:f}\n".format(*(new_bbox + (probability, )))
                    if tracked:
                        last_bbox[id_obj] = new_bbox
                    # if not tracked bboxes are saved at timestamp + delta_t
                    # to emit the BB_DELETE
                    else:
                        bbox_buffer[int(i + delta_t)][id_obj] = bboxes[i]

                file.write(command)
            to_delete = []
            for last_id in sorted(last_bbox.keys()):
                if last_id not in bboxes[i]:
                    command = str(i).zfill(10) + " " + str(last_id) + " BB_DELETE\n"
                    file.write(command)
                    to_delete.append(last_id)

            for id_to_delete in to_delete:
                del last_bbox[id_to_delete]
    if not tracked:
        # at the end of the loop we send the last bbdelete
        for ts in bbox_buffer:
            for id_ in bbox_buffer[ts]:
                file.write("{:010d} {:d} BB_DELETE\n".format(ts, id_))
    file.close()


def read_bboxes(filename, keep_command_labels=[], timestep_us=5000):
    """ This reads a txt file with labeled bboxes and produces
    a dictionary with the information about the bounding boxes:

    an example of bboxes might be:

    {1000: {1: {"class_id": 0, "bbox": (20,20,10,10), "probability":0.6}},
     2000: {1: {"class_id": 0, "bbox": (20,20,10,10)},
            2: {"class_id": 0, "bbox": (40,20,10,10)}},
     3000: {2: {"class_id": 0, "bbox": (50,20,10,10)}},
     .
     .
     .
     }
     You can optionally specify a list of command labels you would like to keep,
     the dict returned will skip any command labels not in this list.
    """
    file = open(filename, "r")
    class_id = 0
    bboxes = {}
    created_bboxes = {}
    for line in file:
        if line[0] == "%":
            continue
        probability = 1
        end_of_track = -1
        tokens = line.split(" ")
        tokens[2] = tokens[2].rstrip()

        if keep_command_labels and (not tokens[2] in keep_command_labels):
            continue

        if tokens[2] == "BB_CREATE":
            created_bboxes[int(tokens[1])] = (float(tokens[4]), float(tokens[5]),
                                              float(tokens[6]), float(tokens[7]))
            if len(tokens) == 9:
                probability = tokens[8]
            class_id = tokens[3]
        elif tokens[2] == "BB_MOVE":
            warnings.warn("In this protocol we assume that all trackers are created (BB_CREATE) in this file")
            prev_box = created_bboxes[int(tokens[1])]
            created_bboxes[int(tokens[1])] = (float(tokens[3]), float(tokens[4]),
                                              prev_box[2], prev_box[3])
            if len(tokens) == 6:
                probability = tokens[5]
        elif tokens[2] == "BB_RESIZE":
            warnings.warn("In this protocol we assume that all trackers are created (BB_CREATE) in this file")
            prev_box = created_bboxes[int(tokens[1])]
            created_bboxes[int(tokens[1])] = (prev_box[0], prev_box[1],
                                              float(tokens[3]), float(tokens[4]))
            if len(tokens) == 6:
                probability = tokens[5]
        elif tokens[2] == "BB_MOVE_AND_RESIZE":
            created_bboxes[int(tokens[1])] = (float(tokens[3]), float(tokens[4]),
                                              float(tokens[5]), float(tokens[6]))
            if len(tokens) == 8:
                probability = tokens[7]
        elif tokens[2] == "BB_DELETE":
            end_of_track = int(tokens[0])
        else:
            del created_bboxes[int(tokens[1])]

        bbox_id = int(tokens[1])
        frame_idx = int(tokens[0])
        if bbox_id in created_bboxes:
            # end_of_track to be updated for the given object
            if end_of_track >= 0:
                to_update_frame_idx = frame_idx - timestep_us
                while to_update_frame_idx in bboxes:
                    if bbox_id in bboxes[to_update_frame_idx]:
                        bboxes[to_update_frame_idx][bbox_id]["end_of_track"] = frame_idx
                        to_update_frame_idx -= timestep_us
                    else:
                        break
                continue

            if frame_idx not in bboxes:
                bboxes[frame_idx] = {bbox_id: {"class_id": class_id,
                                               "bbox": created_bboxes[bbox_id],
                                               "probability": probability,
                                               "end_of_track": end_of_track}}
            else:
                bboxes[frame_idx][bbox_id] = {"class_id": class_id,
                                              "bbox": created_bboxes[bbox_id],
                                              "probability": probability,
                                              "end_of_track": end_of_track}

        else:
            bboxes[frame_idx] = {}

    return bboxes


def get_number_of_bboxes(filename):
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    bboxes = set()
    for line in lines:
        if line[0] == "%":
            continue
        tokens = line.split(" ")
        if tokens[2] == "BB_CREATE":
            bboxes.add(tokens[1])
    return len(bboxes)
