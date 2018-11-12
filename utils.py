import numpy as np
import cv2
import ctypes
import os, inspect, sys
import json

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './leap_lib'
# Mac
#arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap


def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients



def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

def get_raw_image(image):
    image_buffer_ptr = image.data_pointer
    ctype_array_def = ctypes.c_ubyte * image.width * image.height
    as_ctype_array = ctype_array_def.from_address(int(image_buffer_ptr))
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    return as_numpy_array



#TODO
#right hand from frame.hands
def frame2json_struct(frame):

    j_frame = {}
    f = None
    if frame.is_valid:
        f = frame
    else:
        j_frame['frame'] = 'invalid'

    h = None
    for hand in frame.hands:
        if hand.is_right and hand.is_valid:
            h = hand

    if h is None:
        j_frame['frame'] = 'invalid'
        return j_frame

    fingers_list = []
    for i, finger in enumerate(h.fingers):
        fingers_list.append(finger)

    bones = {
        't': {
            'metacarpal': fingers_list[0].bone(0),
            'proximal':  fingers_list[0].bone(1),
            'intermediate': fingers_list[0].bone(2),
            'distal': fingers_list[0].bone(3)
        },
        'i': {
            'metacarpal': fingers_list[1].bone(0),
            'proximal': fingers_list[1].bone(1),
            'intermediate': fingers_list[1].bone(2),
            'distal': fingers_list[1].bone(3)
        },
        'm': {
            'metacarpal': fingers_list[2].bone(0),
            'proximal': fingers_list[2].bone(1),
            'intermediate': fingers_list[2].bone(2),
            'distal': fingers_list[2].bone(3)
        },
        'r': {
            'metacarpal': fingers_list[3].bone(0),
            'proximal': fingers_list[3].bone(1),
            'intermediate': fingers_list[3].bone(2),
            'distal': fingers_list[3].bone(3)
        },
        'p': {
            'metacarpal': fingers_list[4].bone(0),
            'proximal': fingers_list[4].bone(1),
            'intermediate': fingers_list[4].bone(2),
            'distal': fingers_list[4].bone(3)
        }
    }

    # costruzione json

    j_frame['frame'] = {
        'id': f.id,
        'timestamp': f.timestamp,
        'right_hand': {
            'id': h.id,
            'palm_position': [h.palm_position.x, h.palm_position.y, h.palm_position.z, h.palm_position.pitch, h.palm_position.yaw, h.palm_position.roll],
            'palm_normal': [h.palm_normal.x, h.palm_normal.y, h.palm_normal.z, h.palm_normal.pitch, h.palm_normal.yaw, h.palm_normal.roll],
            'direction': [h.palm_direction.x, h.palm_direction.y, h.palm_direction.z, h.palm_direction.pitch, h.palm_direction.yaw, h.palm_direction.roll],
            'direction_pitch': h.direction.pitch * Leap.RAD_TO_DEG,
            'normal_roll': h.palm_normal.roll * Leap.RAD_TO_DEG,
            'direction_yaw': h.direction.yaw * Leap.RAD_TO_DEG,
            'fingers': {
                'thumb': {
                    'id': fingers_list[0].id,
                    'length': fingers_list[0].length,
                    'width': fingers_list[0].width,
                    'bones': {
                        'metacarpal': {
                            'prev_joint': bones['t']['metacarpal'].prev_joint,
                            'next_joint': bones['t']['metacarpal'].next_joint,
                            'direction': bones['t']['metacarpal'].direction
                        },
                        'proximal': {
                            'prev_joint': bones['t']['proximal'].prev_joint,
                            'next_joint': bones['t']['proximal'].next_joint,
                            'direction': bones['t']['proximal'].direction
                        },
                        'intermediate': {
                            'prev_joint': bones['t']['intermediate'].prev_joint,
                            'next_joint': bones['t']['intermediate'].next_joint,
                            'direction': bones['t']['intermediate'].direction
                        },
                        'distal': {
                            'prev_joint': bones['t']['distal'].prev_joint,
                            'next_joint': bones['t']['distal'].next_joint,
                            'direction': bones['t']['distal'].direction
                        }
                    }
                },
                'index': {
                    'id': fingers_list[1].id,
                    'length': fingers_list[1].length,
                    'width': fingers_list[1].width,
                    'bones': {
                        'metacarpal': {
                            'prev_joint': bones['i']['metacarpal'].prev_joint,
                            'next_joint': bones['i']['metacarpal'].next_joint,
                            'direction': bones['i']['metacarpal'].direction
                        },
                        'proximal': {
                            'prev_joint': bones['i']['proximal'].prev_joint,
                            'next_joint': bones['i']['proximal'].next_joint,
                            'direction': bones['i']['proximal'].direction
                        },
                        'intermediate': {
                            'prev_joint': bones['i']['intermediate'].prev_joint,
                            'next_joint': bones['i']['intermediate'].next_joint,
                            'direction': bones['i']['intermediate'].direction
                        },
                        'distal': {
                            'prev_joint': bones['i']['distal'].prev_joint,
                            'next_joint': bones['i']['distal'].next_joint,
                            'direction': bones['i']['distal'].direction
                        }
                    }
                },
                'middle': {
                    'id': fingers_list[2].id,
                    'length': fingers_list[2].length,
                    'width': fingers_list[2].width,
                    'bones': {
                        'metacarpal': {
                            'prev_joint': bones['m']['metacarpal'].prev_joint,
                            'next_joint': bones['m']['metacarpal'].next_joint,
                            'direction': bones['m']['metacarpal'].direction
                        },
                        'proximal': {
                            'prev_joint': bones['m']['proximal'].prev_joint,
                            'next_joint': bones['m']['proximal'].next_joint,
                            'direction': bones['m']['proximal'].direction
                        },
                        'intermediate': {
                            'prev_joint': bones['m']['intermediate'].prev_joint,
                            'next_joint': bones['m']['intermediate'].next_joint,
                            'direction': bones['m']['intermediate'].direction
                        },
                        'distal': {
                            'prev_joint': bones['m']['distal'].prev_joint,
                            'next_joint': bones['m']['distal'].next_joint,
                            'direction': bones['m']['distal'].direction
                        }
                    }
                },
                'ring': {
                    'id': fingers_list[3].id,
                    'length': fingers_list[3].length,
                    'width': fingers_list[3].width,
                    'bones': {
                        'metacarpal': {
                            'prev_joint': bones['r']['metacarpal'].prev_joint,
                            'next_joint': bones['r']['metacarpal'].next_joint,
                            'direction': bones['r']['metacarpal'].direction
                        },
                        'proximal': {
                            'prev_joint': bones['r']['proximal'].prev_joint,
                            'next_joint': bones['r']['proximal'].next_joint,
                            'direction': bones['r']['proximal'].direction
                        },
                        'intermediate': {
                            'prev_joint': bones['r']['intermediate'].prev_joint,
                            'next_joint': bones['r']['intermediate'].next_joint,
                            'direction': bones['r']['intermediate'].direction
                        },
                        'distal': {
                            'prev_joint': bones['r']['distal'].prev_joint,
                            'next_joint': bones['r']['distal'].next_joint,
                            'direction': bones['r']['distal'].direction
                        }
                    }
                },
                'pinky': {
                    'id': fingers_list[4].id,
                    'length': fingers_list[4].length,
                    'width': fingers_list[4].width,
                    'bones': {
                        'metacarpal': {
                            'prev_joint': bones['p']['metacarpal'].prev_joint,
                            'next_joint': bones['p']['metacarpal'].next_joint,
                            'direction': bones['p']['metacarpal'].direction
                        },
                        'proximal': {
                            'prev_joint': bones['p']['proximal'].prev_joint,
                            'next_joint': bones['p']['proximal'].next_joint,
                            'direction': bones['p']['proximal'].direction
                        },
                        'intermediate': {
                            'prev_joint': bones['p']['intermediate'].prev_joint,
                            'next_joint': bones['p']['intermediate'].next_joint,
                            'direction': bones['p']['intermediate'].direction
                        },
                        'distal': {
                            'prev_joint': bones['p']['distal'].prev_joint,
                            'next_joint': bones['p']['distal'].next_joint,
                            'direction': bones['p']['distal'].direction
                        }
                    }
                }

            },
            'arm': {
                'direction': [h.arm.direction.x, h.arm.direction.y, h.arm.direction.z, h.arm.direction.pitch, h.arm.direction.yaw, h.arm.direction.roll],
                'wrist_position': [h.arm.wrist_position.x, h.arm.wrist_position.y, h.arm.wrist_position.z, h.arm.wrist_position.pitch, h.arm.wrist_position.yaw, h.arm.wrist_position.roll],
                'elbow_position': [h.arm.elbow_position.x, h.arm.elbow_position.y, h.arm.elbow_position.z, h.arm.elbow_position.pitch, h.arm.elbow_position.yaw, h.arm.elbow_position.roll],
            },


        }
    }

    return j_frame



