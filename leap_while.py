import os, inspect, sys
import utils
import cv2
import numpy as np
import json
import argparse
import time

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


class Gesture:

    def __init__(self, gesture_id, gesture_dir, controller, id_session, maps_initialized=False):
        self.gesture_id = gesture_id
        self.controller = controller
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

        if not os.path.exists(self.directory_rr)and not os.path.exists(self.directory_lr) and \
                not os.path.exists(self.directory_lr)and not os.path.exists(self.directory_lu) \
                and not os.path.exists(self.directory_leap_info) and not os.path.exists(self.directory_rgb):
            os.makedirs(self.directory_rr)
            os.makedirs(self.directory_lr)
            os.makedirs(self.directory_ru)
            os.makedirs(self.directory_lu)
            os.makedirs(self.directory_leap_info)
            os.makedirs(self.directory_rgb)
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


        # directory_rr = "./data/{0}/{1}_{2}/R/raw".format(self.session, self.gesture_id, self.gesture_dir)
        # directory_lr = "./data/{0}/{1}_{2}/L/raw".format(self.session, self.gesture_id, self.gesture_dir)
        # directory_ru = "./data/{0}/{1}_{2}/R/undistorted".format(self.session, self.gesture_id, self.gesture_dir)
        # directory_lu = "./data/{0}/{1}_{2}/L/undistorted".format(self.session, self.gesture_id, self.gesture_dir)
        # directory_leap_info = "./data/{0}/{1}_{2}/leap_motion_json".format(self.session, self.gesture_id, self.gesture_dir)

        # if not os.path.exists(directory_rr)and not os.path.exists(directory_lr) and \
        #         not os.path.exists(directory_lr)and not os.path.exists(directory_lu) and not os.path.exists(directory_leap_info):
        #     os.makedirs(directory_rr)
        #     os.makedirs(directory_lr)
        #     os.makedirs(directory_ru)
        #     os.makedirs(directory_lu)
        #     os.makedirs(directory_leap_info)
        # else:
        #     print("error on loading session info")
        #     exit(-1)

        record_if_valid = False
        frame_counter = 0
        print("recording")
        # open rgb camera
        cap = cv2.VideoCapture(1)

        while True:

            # start = time.clock()
            if cv2.waitKey(1) == ord('s'):
                break

            frame = self.controller.frame()

            # controllo di validità per inizio registrazione (OPZIONALE)
            # inizia a registrare i frame solo se leap motion rileva correttamente la mano
            if utils.hand_is_valid(frame) and not record_if_valid:
                print('check ok')
                record_if_valid = True

            if record_if_valid:

                # print('hand is valid')
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

                    # get rgb image
                    ret, img_rgb = cap.read()
                    # resize dim img rgb
                    img_rgb = cv2.resize(img_rgb, DIM_RGB_IMAGE)

                    # cv2.imshow('rgb', img_rgb)
                    cv2.imshow('img', undistorted_right)

                    if args.on_disk:

                        thr = utils.ThreadOnDisk(raw_img_r, undistorted_right, raw_img_l, undistorted_left, json_obj,
                                                 img_rgb, frame_counter, self.directory_rr, self.directory_ru,
                                                 self.directory_lr,
                                                 self.directory_lu,
                                                 self.directory_leap_info,
                                                 self.directory_rgb)

                        thr.start()
                    else:
                        list_img_rr.append(raw_img_r.copy())
                        list_img_ru.append(undistorted_right.copy())
                        list_img_lr.append(raw_img_l.copy())
                        list_img_lu.append(undistorted_left.copy())
                        list_img_rgb.append(img_rgb.copy())
                        list_json.append(json_obj)


                    img = np.zeros((400, 1000))
                    cv2.putText(img, "recording - press S to stop",
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                    cv2.circle(img, (700, 100), 50, color=(255, 0, 0), thickness=-1)

                    cv2.imshow('', img)
                    frame_counter += 1
                    # print(time.clock() - start)
                else:
                    print('hand not valid')

        # release rgb camera
        cap.release()

        print('record completed')
        record_if_valid = False
        # scrittura su disco
        if not args.on_disk:

            # thread
            th = utils.ThreadWriting(list_img_rr, list_img_ru, list_img_lr, list_img_lu, list_json, list_img_rgb,
                                     self.directory_rr, self.directory_ru, self.directory_lr, self.directory_lu,
                                     self.directory_leap_info, self.directory_rgb)
            th.start()
            # time.sleep(1)


class Session:

    controller = 0

    id_session = 1
    dir = 0
    gest_counter = 0

    def __init__(self, id_session, controller):
        self.id_session = id_session
        self.controller = controller

    def run_session(self):

        for i in range(0, len(gestures)):
            img = np.zeros((400, 1000))
            cv2.putText(img, "press S to start recording {0}: {1}".format(i, gestures[i]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.circle(img, (700, 100), 50, color=(255, 0, 0), thickness=2)

            cv2.imshow('', img)
            while cv2.waitKey() != ord('s'):
                pass

            g = Gesture(i, gestures[i], self.controller, self.id_session)
            g.record()

        print("recording session ended")

def run(controller):

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
            img = np.zeros((400, 1000))
            cv2.putText(img, "press E to start new session of recording",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            print("press E to register new session of recording")
        else:
            img = np.zeros((400, 1000))
            cv2.putText(img, "press E to start new session of recording or Q to quit",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            print("press E to start new session of recording or Q to quit")
        cv2.imshow('', img)
        k = cv2.waitKey()
        if k == ord('e'):
            pass
        elif k == ord('q'):
            print("end collection")
            utils.save_session_info(session_id=session_counter - 1)
            break
        sess = Session(id_session=session_counter, controller=controller)

        # creazione directory per sessione
        directory = "./data/{}".format(sess.id_session)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # else:
        #
        #     print "insert new session_id and press enter"
        #     # id = raw_input()
        #     img = np.zeros((400, 1000))
        #     cv2.putText(img, "insert new session_id",
        #                 bottomLeftCornerOfText,
        #                 font,
        #                 fontScale,
        #                 fontColor,
        #                 lineType)
        #
        #     cv2.imshow('', img)
        #     while True:
        #         sess.id_session = chr(cv2.waitKey())
        #         while not sess.id_session.isnumeric():
        #             print("insert digit")
        #             sess.id_session = chr(cv2.waitKey())
        #
        #         directory = "./data/{}".format(sess.id_session)
        #         session_counter = int(sess.id_session)
        #         if not os.path.exists(directory):
        #             os.makedirs(directory)
        #             break

        sess.dir = directory
        print("session {} started".format(sess.id_session))
        sess.run_session()
        session_counter += 1


def str2bool(value):
    return value.lower() == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--save_on_disk_frame_by_frame', dest='on_disk', default=False, type=str2bool)
args = parser.parse_args()


def main():

    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    run(controller)


if __name__ == '__main__':
    main()

