import os, inspect, sys, thread
import utils
import cv2
import numpy as np
import json
import time

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './leap_lib'
# Mac
#arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap



bottomLeftCornerOfText = (0, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 1

gestures = ['pinch', 'closing_fist']

class Gesture:

    controller = 0
    maps_initialized = False
    gesture_dir = 0
    gesture_id = 0

    def __init__(self, gesture_id, gesture_dir, controller):
        self.gesture_id = gesture_id
        self.controller = controller
        self.gesture_dir = gesture_dir



    def record(self):

        frame_counter = 0
        print "recording"
        directory = "./{}".format(self.gesture_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)
        else:
            exit(10)

        while True:


            if cv2.waitKey(1) == ord('s'):
                break

            frame = self.controller.frame()
            # previous = self.controller.frame(1)

            image = frame.images[0]

            frame_shown = False

            if image.is_valid:
                if not self.maps_initialized:
                    left_coordinates, left_coefficients = utils.convert_distortion_maps(frame.images[0])
                    right_coordinates, right_coefficients = utils.convert_distortion_maps(frame.images[1])
                    self.maps_initialized = True

                undistorted_left = utils.undistort(image, left_coordinates, left_coefficients, 400, 400)
                undistorted_right = utils.undistort(image, right_coordinates, right_coefficients, 400, 400)

                cv2.imshow('img', undistorted_right)
                cv2.imwrite("{}.jpg".format(frame_counter), undistorted_right)
                #scrittura file
                with open("{}.txt".format(frame_counter), 'w') as outfile:
                    json.dump(utils.frame2json_struct(frame), outfile)

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


        print('record ended')



class Session:

    controller = 0

    id_session = 1
    dir = 0
    gest_counter = 0

    def __init__(self, id_session, controller):
        self.id_session = id_session
        self.controller = controller



    def run_session(self):

        os.chdir(self.dir)

        #ciclo di tutte le gesture - prova 2 gesture
        for i in range(0, len(gestures)):
            img = np.zeros((400, 1000))
            cv2.putText(img, "press S to start recording {}".format(gestures[i]),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.circle(img, (700, 100), 50, color=(255, 0, 0), thickness=2)

            cv2.imshow('', img)
            while cv2.waitKey() != ord('s'):
                pass
            # while c != 's':
            #     print "press s to start recording gesture {}".format(i)
            #     c = raw_input()

            g = Gesture(i, gestures[i], self.controller)
            g.record()
            os.chdir('..')

        print "recording session ended"









def run(controller):

    session_counter = 1
    while True:

        if session_counter == 1:
            img = np.zeros((400, 1000))
            cv2.putText(img, "press E to start new session of recording",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            print "press E to register new session of recording"
        else:
            img = np.zeros((400, 1000))
            cv2.putText(img, "press E to start new session of recording or Q to quit",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            print "press E to start new session of recording or Q to quit"
        cv2.imshow('', img)
        k = cv2.waitKey()
        if k == ord('e'):
            pass
        elif k == ord('q'):
            print "end collection"
            break
        sess = Session(id_session=session_counter, controller=controller)

        # creazione directory per sessione
        directory = "./{}".format(sess.id_session)
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:

            print "insert new session_id and press enter"
            # id = raw_input()
            img = np.zeros((400, 1000))
            cv2.putText(img, "insert new session_id",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.imshow('', img)
            while True:
                sess.id_session = chr(cv2.waitKey())
                directory = "./{}".format(sess.id_session)
                session_counter = int(sess.id_session)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    break

        sess.dir = directory
        print "session {} started".format(sess.id_session)
        sess.run_session()
        os.chdir('..')
        session_counter += 1




def main():

    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    if not os.path.exists("./data"):
        os.makedirs("./data")
    os.chdir("./data")
    # thread.start_new_thread(start_session, ())
    run(controller)

if __name__ == '__main__':
    main()