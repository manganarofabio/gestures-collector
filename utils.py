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
            'palm_velocity': [h.palm_velocity.x, h.palm_velocity.y, h.palm_velocity.z, h.palm_velocity.pitch, h.palm_velocity.yaw, h.palm_velocity.roll],
            'palm_width': h.palm_width,
            'pinch_strength': h.pinch_strength,
            'grab_strength': h.grab_strength,
            'direction': [h.direction.x, h.direction.y, h.direction.z, h.direction.pitch, h.direction.yaw, h.direction.roll],
            'sphere_center': [h.sphere_center.x, h.sphere_center.y, h.sphere_center.z, h.sphere_center.pitch, h.sphere_center.yaw, h.sphere_center.roll],
            'sphere_radius': h.sphere_radius,
            'wrist_position': [h.wrist_position.x, h.wrist_position.y, h.wrist_position.z, h.wrist_position.pitch, h.wrist_position.yaw, h.wrist_position.roll],
            'fingers': {
                'thumb': {
                    'id': fingers_list[0].id,
                    'length': fingers_list[0].length,
                    'width': fingers_list[0].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['t']['metacarpal'].center.x, bones['t']['metacarpal'].center.y, bones['t']['metacarpal'].center.z,
                                       bones['t']['metacarpal'].center.pitch, bones['t']['metacarpal'].center.yaw, bones['t']['metacarpal'].center.roll],
                            'direction': [bones['t']['metacarpal'].direction.x, bones['t']['metacarpal'].direction.y, bones['t']['metacarpal'].direction.z,
                                       bones['t']['metacarpal'].direction.pitch, bones['t']['metacarpal'].direction.yaw, bones['t']['metacarpal'].direction.roll],
                            'length':  bones['t']['metacarpal'].length,
                            'width':  bones['t']['metacarpal'].width,
                            'prev_joint': [bones['t']['metacarpal'].prev_joint.x,bones['t']['metacarpal'].prev_joint.y, bones['t']['metacarpal'].prev_joint.z,
                                           bones['t']['metacarpal'].prev_joint.pitch, bones['t']['metacarpal'].prev_joint.yaw, bones['t']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['t']['metacarpal'].next_joint.x, bones['t']['metacarpal'].next_joint.y, bones['t']['metacarpal'].next_joint.z,
                                           bones['t']['metacarpal'].next_joint.pitch, bones['t']['metacarpal'].next_joint.yaw, bones['t']['metacarpal'].next_joint.roll]

                        },

                        'proximal': {
                            'center': [bones['t']['proximal'].center.x, bones['t']['proximal'].center.y,
                                       bones['t']['proximal'].center.z,
                                       bones['t']['proximal'].center.pitch, bones['t']['proximal'].center.yaw,
                                       bones['t']['proximal'].center.roll],
                            'direction': [bones['t']['proximal'].direction.x, bones['t']['proximal'].direction.y,
                                          bones['t']['proximal'].direction.z,
                                          bones['t']['proximal'].direction.pitch,
                                          bones['t']['proximal'].direction.yaw,
                                          bones['t']['proximal'].direction.roll],
                            'length': bones['t']['proximal'].length,
                            'width': bones['t']['proximal'].width,
                            'prev_joint': [bones['t']['proximal'].prev_joint.x, bones['t']['proximal'].prev_joint.y,
                                           bones['t']['proximal'].prev_joint.z,
                                           bones['t']['proximal'].prev_joint.pitch,
                                           bones['t']['proximal'].prev_joint.yaw,
                                           bones['t']['proximal'].prev_joint.roll],
                            'next_joint': [bones['t']['proximal'].next_joint.x, bones['t']['proximal'].next_joint.y,
                                           bones['t']['proximal'].next_joint.z,
                                           bones['t']['proximal'].next_joint.pitch,
                                           bones['t']['proximal'].next_joint.yaw,
                                           bones['t']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['t']['intermediate'].center.x, bones['t']['intermediate'].center.y,
                                       bones['t']['intermediate'].center.z,
                                       bones['t']['intermediate'].center.pitch, bones['t']['intermediate'].center.yaw,
                                       bones['t']['intermediate'].center.roll],
                            'direction': [bones['t']['intermediate'].direction.x, bones['t']['intermediate'].direction.y,
                                          bones['t']['intermediate'].direction.z,
                                          bones['t']['intermediate'].direction.pitch,
                                          bones['t']['intermediate'].direction.yaw,
                                          bones['t']['intermediate'].direction.roll],
                            'length': bones['t']['intermediate'].length,
                            'width': bones['t']['intermediate'].width,
                            'prev_joint': [bones['t']['intermediate'].prev_joint.x, bones['t']['intermediate'].prev_joint.y,
                                           bones['t']['intermediate'].prev_joint.z,
                                           bones['t']['intermediate'].prev_joint.pitch,
                                           bones['t']['intermediate'].prev_joint.yaw,
                                           bones['t']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['t']['intermediate'].next_joint.x, bones['t']['intermediate'].next_joint.y,
                                           bones['t']['intermediate'].next_joint.z,
                                           bones['t']['intermediate'].next_joint.pitch,
                                           bones['t']['intermediate'].next_joint.yaw,
                                           bones['t']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['t']['distal'].center.x, bones['t']['distal'].center.y,
                                       bones['t']['distal'].center.z,
                                       bones['t']['distal'].center.pitch, bones['t']['distal'].center.yaw,
                                       bones['t']['distal'].center.roll],
                            'direction': [bones['t']['distal'].direction.x, bones['t']['distal'].direction.y,
                                          bones['t']['distal'].direction.z,
                                          bones['t']['distal'].direction.pitch,
                                          bones['t']['distal'].direction.yaw,
                                          bones['t']['distal'].direction.roll],
                            'length': bones['t']['distal'].length,
                            'width': bones['t']['distal'].width,
                            'prev_joint': [bones['t']['distal'].prev_joint.x, bones['t']['distal'].prev_joint.y,
                                           bones['t']['distal'].prev_joint.z,
                                           bones['t']['distal'].prev_joint.pitch,
                                           bones['t']['distal'].prev_joint.yaw,
                                           bones['t']['distal'].prev_joint.roll],
                            'next_joint': [bones['t']['distal'].next_joint.x, bones['t']['distal'].next_joint.y,
                                           bones['t']['distal'].next_joint.z,
                                           bones['t']['distal'].next_joint.pitch,
                                           bones['t']['distal'].next_joint.yaw,
                                           bones['t']['distal'].next_joint.roll]
                        }
                    }

                },

                'index': {
                    'id': fingers_list[1].id,
                    'length': fingers_list[1].length,
                    'width': fingers_list[1].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['i']['metacarpal'].center.x, bones['i']['metacarpal'].center.y,
                                       bones['i']['metacarpal'].center.z,
                                       bones['i']['metacarpal'].center.pitch, bones['i']['metacarpal'].center.yaw,
                                       bones['i']['metacarpal'].center.roll],
                            'direction': [bones['i']['metacarpal'].direction.x, bones['i']['metacarpal'].direction.y,
                                          bones['i']['metacarpal'].direction.z,
                                          bones['i']['metacarpal'].direction.pitch,
                                          bones['i']['metacarpal'].direction.yaw,
                                          bones['i']['metacarpal'].direction.roll],
                            'length': bones['i']['metacarpal'].length,
                            'width': bones['i']['metacarpal'].width,
                            'prev_joint': [bones['i']['metacarpal'].prev_joint.x, bones['i']['metacarpal'].prev_joint.y,
                                           bones['i']['metacarpal'].prev_joint.z,
                                           bones['i']['metacarpal'].prev_joint.pitch,
                                           bones['i']['metacarpal'].prev_joint.yaw,
                                           bones['i']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['i']['metacarpal'].next_joint.x, bones['i']['metacarpal'].next_joint.y,
                                           bones['i']['metacarpal'].next_joint.z,
                                           bones['i']['metacarpal'].next_joint.pitch,
                                           bones['i']['metacarpal'].next_joint.yaw,
                                           bones['m']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['i']['proximal'].center.x, bones['i']['proximal'].center.y,
                                       bones['i']['proximal'].center.z,
                                       bones['i']['proximal'].center.pitch, bones['i']['proximal'].center.yaw,
                                       bones['i']['proximal'].center.roll],
                            'direction': [bones['i']['proximal'].direction.x, bones['i']['proximal'].direction.y,
                                          bones['i']['proximal'].direction.z,
                                          bones['i']['proximal'].direction.pitch,
                                          bones['i']['proximal'].direction.yaw,
                                          bones['i']['proximal'].direction.roll],
                            'length': bones['i']['proximal'].length,
                            'width': bones['i']['proximal'].width,
                            'prev_joint': [bones['i']['proximal'].prev_joint.x, bones['i']['proximal'].prev_joint.y,
                                           bones['i']['proximal'].prev_joint.z,
                                           bones['i']['proximal'].prev_joint.pitch,
                                           bones['i']['proximal'].prev_joint.yaw,
                                           bones['i']['proximal'].prev_joint.roll],
                            'next_joint': [bones['i']['proximal'].next_joint.x, bones['i']['proximal'].next_joint.y,
                                           bones['i']['proximal'].next_joint.z,
                                           bones['i']['proximal'].next_joint.pitch,
                                           bones['i']['proximal'].next_joint.yaw,
                                           bones['i']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['i']['intermediate'].center.x, bones['i']['intermediate'].center.y,
                                       bones['i']['intermediate'].center.z,
                                       bones['i']['intermediate'].center.pitch, bones['i']['intermediate'].center.yaw,
                                       bones['i']['intermediate'].center.roll],
                            'direction': [bones['i']['intermediate'].direction.x, bones['i']['intermediate'].direction.y,
                                          bones['i']['intermediate'].direction.z,
                                          bones['i']['intermediate'].direction.pitch,
                                          bones['i']['intermediate'].direction.yaw,
                                          bones['i']['intermediate'].direction.roll],
                            'length': bones['i']['intermediate'].length,
                            'width': bones['i']['intermediate'].width,
                            'prev_joint': [bones['i']['intermediate'].prev_joint.x, bones['i']['intermediate'].prev_joint.y,
                                           bones['i']['intermediate'].prev_joint.z,
                                           bones['i']['intermediate'].prev_joint.pitch,
                                           bones['i']['intermediate'].prev_joint.yaw,
                                           bones['i']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['i']['intermediate'].next_joint.x, bones['i']['intermediate'].next_joint.y,
                                           bones['i']['intermediate'].next_joint.z,
                                           bones['i']['intermediate'].next_joint.pitch,
                                           bones['i']['intermediate'].next_joint.yaw,
                                           bones['i']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['i']['distal'].center.x, bones['i']['distal'].center.y,
                                       bones['i']['distal'].center.z,
                                       bones['i']['distal'].center.pitch, bones['i']['distal'].center.yaw,
                                       bones['i']['distal'].center.roll],
                            'direction': [bones['i']['distal'].direction.x, bones['i']['distal'].direction.y,
                                          bones['i']['distal'].direction.z,
                                          bones['i']['distal'].direction.pitch,
                                          bones['i']['distal'].direction.yaw,
                                          bones['i']['distal'].direction.roll],
                            'length': bones['i']['distal'].length,
                            'width': bones['i']['distal'].width,
                            'prev_joint': [bones['i']['distal'].prev_joint.x, bones['i']['distal'].prev_joint.y,
                                           bones['i']['distal'].prev_joint.z,
                                           bones['i']['distal'].prev_joint.pitch,
                                           bones['i']['distal'].prev_joint.yaw,
                                           bones['i']['distal'].prev_joint.roll],
                            'next_joint': [bones['i']['distal'].next_joint.x, bones['i']['distal'].next_joint.y,
                                           bones['i']['distal'].next_joint.z,
                                           bones['i']['distal'].next_joint.pitch,
                                           bones['i']['distal'].next_joint.yaw,
                                           bones['i']['distal'].next_joint.roll]
                        }
                    }
                },
                'middle': {
                    'id': fingers_list[2].id,
                    'length': fingers_list[2].length,
                    'width': fingers_list[2].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['m']['metacarpal'].center.x, bones['m']['metacarpal'].center.y,
                                       bones['m']['metacarpal'].center.z,
                                       bones['m']['metacarpal'].center.pitch, bones['m']['metacarpal'].center.yaw,
                                       bones['m']['metacarpal'].center.roll],
                            'direction': [bones['m']['metacarpal'].direction.x, bones['m']['metacarpal'].direction.y,
                                          bones['m']['metacarpal'].direction.z,
                                          bones['m']['metacarpal'].direction.pitch,
                                          bones['m']['metacarpal'].direction.yaw,
                                          bones['m']['metacarpal'].direction.roll],
                            'length': bones['m']['metacarpal'].length,
                            'width': bones['m']['metacarpal'].width,
                            'prev_joint': [bones['m']['metacarpal'].prev_joint.x, bones['m']['metacarpal'].prev_joint.y,
                                           bones['m']['metacarpal'].prev_joint.z,
                                           bones['m']['metacarpal'].prev_joint.pitch,
                                           bones['m']['metacarpal'].prev_joint.yaw,
                                           bones['m']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['m']['metacarpal'].next_joint.x, bones['m']['metacarpal'].next_joint.y,
                                           bones['m']['metacarpal'].next_joint.z,
                                           bones['m']['metacarpal'].next_joint.pitch,
                                           bones['m']['metacarpal'].next_joint.yaw,
                                           bones['m']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['m']['proximal'].center.x, bones['m']['proximal'].center.y,
                                       bones['m']['proximal'].center.z,
                                       bones['m']['proximal'].center.pitch, bones['m']['proximal'].center.yaw,
                                       bones['m']['proximal'].center.roll],
                            'direction': [bones['m']['proximal'].direction.x, bones['m']['proximal'].direction.y,
                                          bones['m']['proximal'].direction.z,
                                          bones['m']['proximal'].direction.pitch,
                                          bones['m']['proximal'].direction.yaw,
                                          bones['m']['proximal'].direction.roll],
                            'length': bones['m']['proximal'].length,
                            'width': bones['m']['proximal'].width,
                            'prev_joint': [bones['m']['proximal'].prev_joint.x, bones['m']['proximal'].prev_joint.y,
                                           bones['m']['proximal'].prev_joint.z,
                                           bones['m']['proximal'].prev_joint.pitch,
                                           bones['m']['proximal'].prev_joint.yaw,
                                           bones['m']['proximal'].prev_joint.roll],
                            'next_joint': [bones['m']['proximal'].next_joint.x, bones['m']['proximal'].next_joint.y,
                                           bones['m']['proximal'].next_joint.z,
                                           bones['m']['proximal'].next_joint.pitch,
                                           bones['m']['proximal'].next_joint.yaw,
                                           bones['m']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['m']['intermediate'].center.x, bones['m']['intermediate'].center.y,
                                       bones['m']['intermediate'].center.z,
                                       bones['m']['intermediate'].center.pitch, bones['m']['intermediate'].center.yaw,
                                       bones['m']['intermediate'].center.roll],
                            'direction': [bones['m']['intermediate'].direction.x,
                                          bones['m']['intermediate'].direction.y,
                                          bones['m']['intermediate'].direction.z,
                                          bones['m']['intermediate'].direction.pitch,
                                          bones['m']['intermediate'].direction.yaw,
                                          bones['m']['intermediate'].direction.roll],
                            'length': bones['m']['intermediate'].length,
                            'width': bones['m']['intermediate'].width,
                            'prev_joint': [bones['m']['intermediate'].prev_joint.x,
                                           bones['m']['intermediate'].prev_joint.y,
                                           bones['m']['intermediate'].prev_joint.z,
                                           bones['m']['intermediate'].prev_joint.pitch,
                                           bones['m']['intermediate'].prev_joint.yaw,
                                           bones['m']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['m']['intermediate'].next_joint.x,
                                           bones['m']['intermediate'].next_joint.y,
                                           bones['m']['intermediate'].next_joint.z,
                                           bones['m']['intermediate'].next_joint.pitch,
                                           bones['m']['intermediate'].next_joint.yaw,
                                           bones['m']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['m']['distal'].center.x, bones['m']['distal'].center.y,
                                       bones['m']['distal'].center.z,
                                       bones['m']['distal'].center.pitch, bones['m']['distal'].center.yaw,
                                       bones['m']['distal'].center.roll],
                            'direction': [bones['m']['distal'].direction.x, bones['m']['distal'].direction.y,
                                          bones['m']['distal'].direction.z,
                                          bones['m']['distal'].direction.pitch,
                                          bones['m']['distal'].direction.yaw,
                                          bones['m']['distal'].direction.roll],
                            'length': bones['m']['distal'].length,
                            'width': bones['m']['distal'].width,
                            'prev_joint': [bones['m']['distal'].prev_joint.x, bones['m']['distal'].prev_joint.y,
                                           bones['m']['distal'].prev_joint.z,
                                           bones['m']['distal'].prev_joint.pitch,
                                           bones['m']['distal'].prev_joint.yaw,
                                           bones['m']['distal'].prev_joint.roll],
                            'next_joint': [bones['m']['distal'].next_joint.x, bones['m']['distal'].next_joint.y,
                                           bones['m']['distal'].next_joint.z,
                                           bones['m']['distal'].next_joint.pitch,
                                           bones['m']['distal'].next_joint.yaw,
                                           bones['m']['distal'].next_joint.roll]
                        }
                    }
                },
                'ring': {
                    'id': fingers_list[3].id,
                    'length': fingers_list[3].length,
                    'width': fingers_list[3].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['r']['metacarpal'].center.x, bones['r']['metacarpal'].center.y,
                                       bones['r']['metacarpal'].center.z,
                                       bones['r']['metacarpal'].center.pitch, bones['r']['metacarpal'].center.yaw,
                                       bones['r']['metacarpal'].center.roll],
                            'direction': [bones['r']['metacarpal'].direction.x, bones['r']['metacarpal'].direction.y,
                                          bones['r']['metacarpal'].direction.z,
                                          bones['r']['metacarpal'].direction.pitch,
                                          bones['r']['metacarpal'].direction.yaw,
                                          bones['r']['metacarpal'].direction.roll],
                            'length': bones['r']['metacarpal'].length,
                            'width': bones['r']['metacarpal'].width,
                            'prev_joint': [bones['r']['metacarpal'].prev_joint.x, bones['r']['metacarpal'].prev_joint.y,
                                           bones['r']['metacarpal'].prev_joint.z,
                                           bones['r']['metacarpal'].prev_joint.pitch,
                                           bones['r']['metacarpal'].prev_joint.yaw,
                                           bones['r']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['r']['metacarpal'].next_joint.x, bones['r']['metacarpal'].next_joint.y,
                                           bones['r']['metacarpal'].next_joint.z,
                                           bones['r']['metacarpal'].next_joint.pitch,
                                           bones['r']['metacarpal'].next_joint.yaw,
                                           bones['r']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['r']['proximal'].center.x, bones['r']['proximal'].center.y,
                                       bones['r']['proximal'].center.z,
                                       bones['r']['proximal'].center.pitch, bones['r']['proximal'].center.yaw,
                                       bones['r']['proximal'].center.roll],
                            'direction': [bones['r']['proximal'].direction.x, bones['r']['proximal'].direction.y,
                                          bones['r']['proximal'].direction.z,
                                          bones['r']['proximal'].direction.pitch,
                                          bones['r']['proximal'].direction.yaw,
                                          bones['r']['proximal'].direction.roll],
                            'length': bones['r']['proximal'].length,
                            'width': bones['r']['proximal'].width,
                            'prev_joint': [bones['r']['proximal'].prev_joint.x, bones['r']['proximal'].prev_joint.y,
                                           bones['r']['proximal'].prev_joint.z,
                                           bones['r']['proximal'].prev_joint.pitch,
                                           bones['r']['proximal'].prev_joint.yaw,
                                           bones['r']['proximal'].prev_joint.roll],
                            'next_joint': [bones['r']['proximal'].next_joint.x, bones['r']['proximal'].next_joint.y,
                                           bones['r']['proximal'].next_joint.z,
                                           bones['r']['proximal'].next_joint.pitch,
                                           bones['r']['proximal'].next_joint.yaw,
                                           bones['r']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['r']['intermediate'].center.x, bones['r']['intermediate'].center.y,
                                       bones['r']['intermediate'].center.z,
                                       bones['r']['intermediate'].center.pitch, bones['r']['intermediate'].center.yaw,
                                       bones['r']['intermediate'].center.roll],
                            'direction': [bones['r']['intermediate'].direction.x,
                                          bones['r']['intermediate'].direction.y,
                                          bones['r']['intermediate'].direction.z,
                                          bones['r']['intermediate'].direction.pitch,
                                          bones['r']['intermediate'].direction.yaw,
                                          bones['r']['intermediate'].direction.roll],
                            'length': bones['r']['intermediate'].length,
                            'width': bones['r']['intermediate'].width,
                            'prev_joint': [bones['r']['intermediate'].prev_joint.x,
                                           bones['r']['intermediate'].prev_joint.y,
                                           bones['r']['intermediate'].prev_joint.z,
                                           bones['r']['intermediate'].prev_joint.pitch,
                                           bones['r']['intermediate'].prev_joint.yaw,
                                           bones['r']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['r']['intermediate'].next_joint.x,
                                           bones['r']['intermediate'].next_joint.y,
                                           bones['r']['intermediate'].next_joint.z,
                                           bones['r']['intermediate'].next_joint.pitch,
                                           bones['r']['intermediate'].next_joint.yaw,
                                           bones['r']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['r']['distal'].center.x, bones['r']['distal'].center.y,
                                       bones['r']['distal'].center.z,
                                       bones['r']['distal'].center.pitch, bones['r']['distal'].center.yaw,
                                       bones['r']['distal'].center.roll],
                            'direction': [bones['r']['distal'].direction.x, bones['r']['distal'].direction.y,
                                          bones['r']['distal'].direction.z,
                                          bones['r']['distal'].direction.pitch,
                                          bones['r']['distal'].direction.yaw,
                                          bones['r']['distal'].direction.roll],
                            'length': bones['r']['distal'].length,
                            'width': bones['r']['distal'].width,
                            'prev_joint': [bones['r']['distal'].prev_joint.x, bones['r']['distal'].prev_joint.y,
                                           bones['r']['distal'].prev_joint.z,
                                           bones['r']['distal'].prev_joint.pitch,
                                           bones['r']['distal'].prev_joint.yaw,
                                           bones['r']['distal'].prev_joint.roll],
                            'next_joint': [bones['r']['distal'].next_joint.x, bones['r']['distal'].next_joint.y,
                                           bones['r']['distal'].next_joint.z,
                                           bones['r']['distal'].next_joint.pitch,
                                           bones['r']['distal'].next_joint.yaw,
                                           bones['r']['distal'].next_joint.roll]
                        }
                    }
                },
                'pinky': {
                    'id': fingers_list[4].id,
                    'length': fingers_list[4].length,
                    'width': fingers_list[4].width,
                    'bones': {
                        'metacarpal': {
                            'center': [bones['p']['metacarpal'].center.x, bones['p']['metacarpal'].center.y,
                                       bones['p']['metacarpal'].center.z,
                                       bones['p']['metacarpal'].center.pitch, bones['p']['metacarpal'].center.yaw,
                                       bones['p']['metacarpal'].center.roll],
                            'direction': [bones['p']['metacarpal'].direction.x, bones['p']['metacarpal'].direction.y,
                                          bones['p']['metacarpal'].direction.z,
                                          bones['p']['metacarpal'].direction.pitch,
                                          bones['p']['metacarpal'].direction.yaw,
                                          bones['p']['metacarpal'].direction.roll],
                            'length': bones['p']['metacarpal'].length,
                            'width': bones['p']['metacarpal'].width,
                            'prev_joint': [bones['p']['metacarpal'].prev_joint.x, bones['p']['metacarpal'].prev_joint.y,
                                           bones['p']['metacarpal'].prev_joint.z,
                                           bones['p']['metacarpal'].prev_joint.pitch,
                                           bones['p']['metacarpal'].prev_joint.yaw,
                                           bones['p']['metacarpal'].prev_joint.roll],
                            'next_joint': [bones['p']['metacarpal'].next_joint.x, bones['p']['metacarpal'].next_joint.y,
                                           bones['p']['metacarpal'].next_joint.z,
                                           bones['p']['metacarpal'].next_joint.pitch,
                                           bones['p']['metacarpal'].next_joint.yaw,
                                           bones['p']['metacarpal'].next_joint.roll]
                        },
                        'proximal': {
                            'center': [bones['p']['proximal'].center.x, bones['p']['proximal'].center.y,
                                       bones['p']['proximal'].center.z,
                                       bones['p']['proximal'].center.pitch, bones['p']['proximal'].center.yaw,
                                       bones['p']['proximal'].center.roll],
                            'direction': [bones['p']['proximal'].direction.x, bones['p']['proximal'].direction.y,
                                          bones['p']['proximal'].direction.z,
                                          bones['p']['proximal'].direction.pitch,
                                          bones['p']['proximal'].direction.yaw,
                                          bones['p']['proximal'].direction.roll],
                            'length': bones['p']['proximal'].length,
                            'width': bones['p']['proximal'].width,
                            'prev_joint': [bones['p']['proximal'].prev_joint.x, bones['p']['proximal'].prev_joint.y,
                                           bones['p']['proximal'].prev_joint.z,
                                           bones['p']['proximal'].prev_joint.pitch,
                                           bones['p']['proximal'].prev_joint.yaw,
                                           bones['p']['proximal'].prev_joint.roll],
                            'next_joint': [bones['p']['proximal'].next_joint.x, bones['p']['proximal'].next_joint.y,
                                           bones['p']['proximal'].next_joint.z,
                                           bones['p']['proximal'].next_joint.pitch,
                                           bones['p']['proximal'].next_joint.yaw,
                                           bones['p']['proximal'].next_joint.roll]
                        },
                        'intermediate': {
                            'center': [bones['p']['intermediate'].center.x, bones['p']['intermediate'].center.y,
                                       bones['p']['intermediate'].center.z,
                                       bones['p']['intermediate'].center.pitch, bones['p']['intermediate'].center.yaw,
                                       bones['p']['intermediate'].center.roll],
                            'direction': [bones['p']['intermediate'].direction.x,
                                          bones['p']['intermediate'].direction.y,
                                          bones['p']['intermediate'].direction.z,
                                          bones['p']['intermediate'].direction.pitch,
                                          bones['p']['intermediate'].direction.yaw,
                                          bones['p']['intermediate'].direction.roll],
                            'length': bones['p']['intermediate'].length,
                            'width': bones['p']['intermediate'].width,
                            'prev_joint': [bones['p']['intermediate'].prev_joint.x,
                                           bones['p']['intermediate'].prev_joint.y,
                                           bones['p']['intermediate'].prev_joint.z,
                                           bones['p']['intermediate'].prev_joint.pitch,
                                           bones['p']['intermediate'].prev_joint.yaw,
                                           bones['p']['intermediate'].prev_joint.roll],
                            'next_joint': [bones['p']['intermediate'].next_joint.x,
                                           bones['p']['intermediate'].next_joint.y,
                                           bones['p']['intermediate'].next_joint.z,
                                           bones['p']['intermediate'].next_joint.pitch,
                                           bones['p']['intermediate'].next_joint.yaw,
                                           bones['p']['intermediate'].next_joint.roll]
                        },
                        'distal': {
                            'center': [bones['p']['distal'].center.x, bones['p']['distal'].center.y,
                                       bones['p']['distal'].center.z,
                                       bones['p']['distal'].center.pitch, bones['p']['distal'].center.yaw,
                                       bones['p']['distal'].center.roll],
                            'direction': [bones['p']['distal'].direction.x, bones['p']['distal'].direction.y,
                                          bones['p']['distal'].direction.z,
                                          bones['p']['distal'].direction.pitch,
                                          bones['p']['distal'].direction.yaw,
                                          bones['p']['distal'].direction.roll],
                            'length': bones['p']['distal'].length,
                            'width': bones['p']['distal'].width,
                            'prev_joint': [bones['p']['distal'].prev_joint.x, bones['p']['distal'].prev_joint.y,
                                           bones['p']['distal'].prev_joint.z,
                                           bones['p']['distal'].prev_joint.pitch,
                                           bones['p']['distal'].prev_joint.yaw,
                                           bones['p']['distal'].prev_joint.roll],
                            'next_joint': [bones['p']['distal'].next_joint.x, bones['p']['distal'].next_joint.y,
                                           bones['p']['distal'].next_joint.z,
                                           bones['p']['distal'].next_joint.pitch,
                                           bones['p']['distal'].next_joint.yaw,
                                           bones['p']['distal'].next_joint.roll]
                        }
                    }
                }
            },
            'arm': {
                'width': h.arm.width,
                'direction': [h.arm.direction.x, h.arm.direction.y, h.arm.direction.z, h.arm.direction.pitch, h.arm.direction.yaw, h.arm.direction.roll],
                'wrist_position': [h.arm.wrist_position.x, h.arm.wrist_position.y, h.arm.wrist_position.z, h.arm.wrist_position.pitch, h.arm.wrist_position.yaw, h.arm.wrist_position.roll],
                'elbow_position': [h.arm.elbow_position.x, h.arm.elbow_position.y, h.arm.elbow_position.z, h.arm.elbow_position.pitch, h.arm.elbow_position.yaw, h.arm.elbow_position.roll],
            },
        }
    }
    return j_frame



