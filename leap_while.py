import os, inspect, sys
import utils
import cv2
import numpy as np
import json
import argparse
import time
import queue
import roypy
from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper

asd = 4

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './leap_lib'

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

bottomLeftCornerOfText = (0, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 1

DIM_RGB_IMAGE = (400, 400)

file_info = "session_info.json"
gestures = ['pinch', 'closing_fist']
threads = []


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q

    def onNewData(self, data):
        z_values = []
        gray_values = []
        for i in range(data.getNumPoints()):
            z_values.append(data.getZ(i))
            gray_values.append(data.getGrayValue(i))

        z_array = np.array(z_values)
        gray_array = np.array(gray_values)

        z_p = z_array.reshape(-1, data.width)
        gray_p = gray_array.reshape(-1, data.width)

        self.queue.put((z_p, gray_p))


class Gesture:

    def __init__(self, gesture_id, gesture_dir, controller, cam, queue, id_session, maps_initialized=False):
        self.gesture_id = gesture_id
        self.controller = controller
        self.cam = cam
        self.q = queue
        self.gesture_dir = gesture_dir
        self.session = id_session
        self.maps_initialized = maps_initialized
        # directories
        self.directory_rr = "./data/{0}/{1}_{2}/R/raw".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_lr = "./data/{0}/{1}_{2}/L/raw".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_ru = "./data/{0}/{1}_{2}/R/undistorted".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_lu = "./data/{0}/{1}_{2}/L/undistorted".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_leap_info = "./data/{0}/{1}_{2}/leap_motion_json".format(self.session, self.gesture_id,
                                                                           self.gesture_dir)
        self.directory_rgb = "./data/{0}/{1}_{2}/rgb".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_z = "./data/{0}/{1}_{2}/depth/z".format(self.session, self.gesture_id, self.gesture_dir)
        self.directory_ir = "./data/{0}/{1}_{2}/depth/ir".format(self.session, self.gesture_id, self.gesture_dir)

        if not os.path.exists(self.directory_rr)and not os.path.exists(self.directory_lr) and \
                not os.path.exists(self.directory_lr)and not os.path.exists(self.directory_lu) \
                and not os.path.exists(self.directory_leap_info) and not os.path.exists(self.directory_rgb) \
                and not os.path.exists(self.directory_z) and not os.path.exists(self.directory_ir):
            os.makedirs(self.directory_rr)
            os.makedirs(self.directory_lr)
            os.makedirs(self.directory_ru)
            os.makedirs(self.directory_lu)
            os.makedirs(self.directory_leap_info)
            os.makedirs(self.directory_rgb)
            os.makedirs(self.directory_z)
            os.makedirs(self.directory_ir)

        else:
            print("error on loading session info")
            exit(-1)

    def record(self):

        list_img_rr = []
        list_img_ru = []
        list_img_lr = []
        list_img_lu = []
        list_json = []
        list_img_rgb = []
        list_img_z = []
        list_img_ir = []

        record_if_valid = True
        frame_counter = 0
        print("recording")
        # open rgb camera
        cap = cv2.VideoCapture(0)
        if not cap:
            print("error rgb cam")
            exit(-1)

        while True:

            if cv2.waitKey(1) == ord('s'):
                break

            frame = self.controller.frame()

            # controllo di validità per inizio registrazione (OPZIONALE)
            # inizia a registrare i frame solo se leap motion rileva correttamente la mano

            # if utils.hand_is_valid(frame) and not record_if_valid:
            #     print('check ok')
            #     record_if_valid = True

            if record_if_valid:

                # print('hand is valid')
                # Leap Motion
                image_l = frame.images[0]
                image_r = frame.images[1]

                if image_l.is_valid and image_r.is_valid:
                    if not self.maps_initialized:
                        left_coordinates, left_coefficients = utils.convert_distortion_maps(image_l)
                        right_coordinates, right_coefficients = utils.convert_distortion_maps(image_r)
                        self.maps_initialized = True

                    raw_img_l = utils.get_raw_image(image_l)
                    raw_img_r = utils.get_raw_image(image_r)
                    # undistorted images
                    undistorted_left = utils.undistort(image_l, left_coordinates, left_coefficients, 400, 400)
                    undistorted_right = utils.undistort(image_r, right_coordinates, right_coefficients, 400, 400)

                    # json
                    json_obj = utils.frame2json_struct(frame)

                    # RGB CAM
                    # get rgb image
                    ret, img_rgb = cap.read()
                    # resize dim img rgb
                    if ret:
                        img_rgb = cv2.resize(img_rgb, DIM_RGB_IMAGE)
                    else:
                        print("rgb cam not working")
                        exit(-1)

                    # PICOFLEXX
                    z, ir = utils.get_images_from_picoflexx(self.q)

                    utils.draw_ui(text="recording - press S to stop", circle=True, thickness=-1)
                    cv2.imshow('img', undistorted_right)

                    if args.on_disk:

                        thr = utils.ThreadOnDisk(raw_img_r, undistorted_right, raw_img_l, undistorted_left, json_obj,
                                                 img_rgb, z, ir, frame_counter, self.directory_rr, self.directory_ru,
                                                 self.directory_lr,
                                                 self.directory_lu,
                                                 self.directory_leap_info,
                                                 self.directory_rgb,
                                                 self.directory_z,
                                                 self.directory_ir)

                        thr.start()
                    else:
                        list_img_rr.append(raw_img_r.copy())
                        list_img_ru.append(undistorted_right.copy())
                        list_img_lr.append(raw_img_l.copy())
                        list_img_lu.append(undistorted_left.copy())
                        list_img_rgb.append(img_rgb.copy())
                        list_json.append(json_obj)
                        list_img_z.append(z.copy())
                        list_img_ir.append(ir.copy())

                    frame_counter += 1

                else:
                    print('hand not valid')

        # release rgb camera
        cap.release()

        print('record completed')
        record_if_valid = False
        # scrittura su disco
        if not args.on_disk:

            # thread
            threads.append(utils.ThreadWritingGesture(list_img_rr, list_img_ru, list_img_lr, list_img_lu, list_json, list_img_rgb,
                                     list_img_z, list_img_ir,
                                     self.directory_rr, self.directory_ru, self.directory_lr, self.directory_lu,
                                     self.directory_leap_info, self.directory_rgb, self.directory_z, self.directory_ir))

            threads[-1].start()



class Session:

    controller = 0
    id_session = 1
    dir = 0
    gest_counter = 0

    def __init__(self, id_session, controller, cam, queue):
        self.id_session = id_session
        self.controller = controller
        self.cam = cam
        self.q = queue

    def run_session(self):

        for i in range(0, len(gestures)):

            utils.draw_ui(text="press S to start recording {0}: {1}".format(i, gestures[i]))
            while cv2.waitKey() != ord('s'):
                pass

            g = Gesture(i, gestures[i], self.controller, self.cam, self.q, self.id_session)
            g.record()

        # self.cam.stopCapture()
        print("recording session ended")


def run(controller, cam):

    # inizializzazione picoflexx
    q = queue.Queue()
    listener = MyListener(q)
    cam.registerDataListener(listener)
    cam.startCapture()

    if not os.path.exists("./data"):
        session_counter = 0
        session_start = 0
        utils.save_session_info(session_id=session_counter)
        os.makedirs("./data")
    elif os.path.exists("./data") and not os.path.exists(file_info):
        print("json file has to be present - check utils.save_session_info()")
        exit()
    else:
        session_start = utils.load_session_info() + 1
        session_counter = session_start
    while True:

        if session_counter == session_start:

            utils.draw_ui(text="press E to start new session of recording")
        else:

            utils.draw_ui(text="press E to start new session of recording or Q to quit")
        k = cv2.waitKey()
        if k == ord('e'):
            pass
        elif k == ord('q'):
            print("end collection")
            utils.save_session_info(session_id=session_counter - 1)
            break
        sess = Session(id_session=session_counter, controller=controller, cam=cam, queue=q)

        # creazione directory per sessione
        directory = "./data/{}".format(sess.id_session)
        if not os.path.exists(directory):
            os.makedirs(directory)
       
        sess.dir = directory
        print("session {} started".format(sess.id_session))
        sess.run_session()
        session_counter += 1

        # join of threads
        for th in threads:
            th.join()

    cam.stopCapture()


def str2bool(value):
    return value.lower() == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--save_on_disk_frame_by_frame', dest='on_disk', default=False, type=str2bool)
args = parser.parse_args()


def main():
    # PICOFLEXX

    parser1 = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser1)

    # parser1.add_argument("--seconds", type=int, default=15, help="duration to capture data")
    options = parser1.parse_args()
    opener = CameraOpener(options)
    cam = opener.open_camera()

    cam.setUseCase("MODE_5_45FPS_500")
    print_camera_info(cam)
    # print("isConnected", cam.isConnected())
    # print("getFrameRate", cam.getFrameRate())

    # LEAP MOTION
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    run(controller, cam)


if __name__ == '__main__':
    main()

