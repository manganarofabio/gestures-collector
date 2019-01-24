import os, inspect, sys
import utils
import cv2
import numpy as np
import argparse
import queue
import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options


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

# DIM_RGB_IMAGE = (400, 400)


file_info = "session_info.json"
gestures = ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12_test']
NUMBER_OF_SESSIONS_PER_PERSON = 3
NUMBER_OF_RECORDS_PER_GESTURE = 3
NUMBER_OF_MINIMUM_FRAMES_PER_RECORD = 35


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q, recording):
        super(MyListener, self).__init__()
        self.queue = q
        self.recording = recording

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

        if self.recording:
            self.queue.put((z_p, gray_p))

    def setRecording(self, value):
        self.recording = value


class Gesture:

    def __init__(self, gesture_id, gesture_dir, record_number, controller, cam, queue, listener, cap, id_session, maps_initialized,
                 left_coord, right_coord, left_coeff, right_coeff, rewrite=False):

        self.gesture_id = gesture_id
        self.controller = controller
        self.record_number = record_number
        self.cam = cam
        self.q = queue
        self.listener = listener
        self.cap = cap
        self.gesture_dir = gesture_dir
        self.session = id_session
        self.maps_initialized = maps_initialized
        self.left_coord = left_coord
        self.right_coord = right_coord
        self.left_coeff = left_coeff
        self.right_coeff = right_coeff
        self.rewrite = rewrite
        # directories
        self.directory_rr = "./data/{:03d}/{}/{:02d}/R/raw".format(self.session, self.gesture_dir, self.record_number)
        self.directory_lr = "./data/{:03d}/{}/{:02d}/L/raw".format(self.session, self.gesture_dir, self.record_number)
        self.directory_ru = "./data/{:03d}/{}/{:02d}/R/undistorted".format(self.session, self.gesture_dir,
                                                                           self.record_number)
        self.directory_lu = "./data/{:03d}/{}/{:02d}/L/undistorted".format(self.session, self.gesture_dir,
                                                                           self.record_number)
        self.directory_leap_info = "./data/{:03d}/{}/{:02d}/leap_motion_json".format(self.session, self.gesture_dir,
                                                                                     self.record_number)
        self.directory_rgb = "./data/{:03d}/{}/{:02d}/rgb".format(self.session, self.gesture_dir, self.record_number)
        self.directory_z = "./data/{:03d}/{}/{:02d}/depth/z".format(self.session, self.gesture_dir, self.record_number)
        self.directory_ir = "./data/{:03d}/{}/{:02d}/depth/ir".format(self.session, self.gesture_dir,
                                                                      self.record_number)

    def record(self):

        list_img_rr = []
        list_img_ru = []
        list_img_lr = []
        list_img_lu = []
        list_json = []
        list_img_rgb = []
        list_img_z = []
        list_img_ir = []

        record_if_valid = False
        frame_counter = 0
        # print("ready to go")
        # open rgb camera
        # cap = cv2.VideoCapture(1)
        #
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap:
            print("error rgb cam")
            exit(-1)

        self.cam.startCapture()
        while True:
            # print(frame_counter)
            if cv2.waitKey(1) == ord('s') and record_if_valid and frame_counter > NUMBER_OF_MINIMUM_FRAMES_PER_RECORD:
                break

            frame = self.controller.frame()

            # controllo di validità per inizio registrazione (OPZIONALE)
            # inizia a registrare i frame solo se leap motion rileva correttamente la mano

            # print(self.listener.recording)
            if utils.hand_is_valid(frame) and not record_if_valid:
                print('\nhand is valid -> ready to start')
                record_if_valid = True
                # print(self.listener.recording)
                print("start gesture")
                self.listener.setRecording(True)
                # print(self.listener.recording)

            if record_if_valid:
                print("\rrecord valid -> showing {}".format(frame_counter), end="")
                utils.draw_ui(text="recording - press S to stop", circle=True, thickness=-1)
                # RGB CAM
                # get rgb image
                # print(1)
                ret, img_rgb = self.cap.read()
                # print(2)
                # resize dim img rgb
                if not ret:
                    print("\nrgb cam not working")
                    exit(-1)
                # cv2.imshow('img_rgb', img_rgb)
                # cv2.waitKey(1)

                # Leap Motion
                if frame.is_valid:
                    image_l = frame.images[0]
                    image_r = frame.images[1]
                    # print(3)
                else:
                    print("\rframe {} not valid".format(frame_counter), end="")
                    continue

                if image_l.is_valid and image_r.is_valid:
                    # print(4)
                    raw_img_l = utils.get_raw_image(image_l)
                    raw_img_r = utils.get_raw_image(image_r)
                    # undistorted images
                    undistorted_left = utils.undistort(image_l, self.left_coord, self.left_coeff, 400, 400)
                    undistorted_right = utils.undistort(image_r, self.right_coord, self.right_coeff, 400, 400)
                    # print(5)

                    # show images
                    # previous position cv2.imshow()
                    # cv2.imshow('img_leap', undistorted_right)

                    # json
                    json_obj = utils.frame2json_struct(frame)
                    # print(6)
                    # PICOFLEXX
                    z, ir = utils.get_images_from_picoflexx(self.q)
                    if z is None or ir is None:
                        print("\rpico image not valid", end="")
                        continue
                    cv2.moveWindow('img_rgb', -700, 325)
                    cv2.moveWindow('img_leap', -1150, 400)
                    cv2.moveWindow('img_ir', -1500, 600)
                    cv2.imshow('img_leap', undistorted_right)
                    cv2.imshow('img_rgb', img_rgb)
                    cv2.imshow('img_ir', ir)

                    # print(7)
                    list_img_rr.append(raw_img_r.copy())
                    list_img_ru.append(undistorted_right.copy())
                    list_img_lr.append(raw_img_l.copy())
                    list_img_lu.append(undistorted_left.copy())
                    list_img_rgb.append(img_rgb.copy())
                    list_json.append(json_obj)
                    list_img_z.append(z.copy())
                    list_img_ir.append(ir.copy())

                    frame_counter += 1
                    # print(8)
                else:
                    print('image not valid')

            else:
                print("\rerror in getting valid leap motion frame", end="")

        # print(self.listener.recording)
        self.listener.setRecording(False)
        self.cam.stopCapture()

        # print(self.listener.recording)
        # release rgb camera
        # cap.release()

        print('\nrecord {}/2 of g{} completed'.format(self.record_number, self.gesture_id))
        record_if_valid = False
        # scrittura su disco
        # if not args.on_disk:
        if self.rewrite:
            rewrite = True
        else:
            rewrite = False

        return utils.GestureData(self.gesture_id, list_img_rr, list_img_ru, list_img_lr, list_img_lu, list_json,
                                 list_img_rgb,
                                 list_img_z, list_img_ir,
                                 self.directory_rr, self.directory_ru, self.directory_lr, self.directory_lu,
                                 self.directory_leap_info, self.directory_rgb, self.directory_z, self.directory_ir,
                                 rewrite=rewrite)


class Session:
    controller = 0
    id_session = 1
    dir = 0
    gest_counter = 0

    def __init__(self, id_session, controller, cam, queue, listener):
        self.id_session = id_session
        self.controller = controller
        self.cam = cam
        self.q = queue
        self.listener = listener

    def run_session(self):

        # init leap motion

        print("waiting for maps initialization...")
        while True:
            frame = self.controller.frame()
            image_l = frame.images[0]
            image_r = frame.images[1]

            if image_l.is_valid and image_r.is_valid:

                left_coordinates, left_coefficients = utils.convert_distortion_maps(image_l)
                right_coordinates, right_coefficients = utils.convert_distortion_maps(image_r)
                maps_initialized = True
                print('maps initialized')

                break
            else:
                print('\rinvalid leap motion frame', end="")

        # initialize video capture
        while True:
            cap = cv2.VideoCapture(1)
            print(cap)
            if cap:
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)
                # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                break
            else:
                print("\rerror rgb cam", end="")

        print("ready to go")

        list_of_gestures = []
        counter_gesture = 0
        rewrite = False
        for i in range(0, len(gestures)):

            # if (i == len(gestures) - 1) and self.last_session is False:
            #     break

            while True:

                if counter_gesture == NUMBER_OF_RECORDS_PER_GESTURE:
                    counter_gesture = 0
                    break
                # if (i == len(gestures) - 1) and counter_gesture > 0:
                #     #counter_gesture = 0

                if counter_gesture == 0:
                    utils.draw_ui(text="press S to start record {0} of {1}".format(counter_gesture, gestures[i]))
                elif (i == len(gestures) - 1) and counter_gesture > 0:
                    utils.draw_ui(text="press R to repeat last record or Q to quit")
                else:
                    utils.draw_ui(text="press S to start record {}/{} of {}/{:02d}"
                                       " or press R to repeat record {}".format(counter_gesture,
                                                                                NUMBER_OF_RECORDS_PER_GESTURE - 1,
                                                                                gestures[i], len(gestures) - 1,
                                                                                counter_gesture - 1))

                while True:
                        k = cv2.waitKey()
                        if k == ord('s'):
                            break
                        elif k == ord('r') and counter_gesture > 0 and i == len(gestures) - 1:
                            counter_gesture = 0
                            rewrite = True
                            break

                        elif k == ord('r') and counter_gesture > 0:
                            counter_gesture -= 1
                            rewrite = True
                            break
                        elif k == ord('q') and i == len(gestures) - 1:
                            counter_gesture = NUMBER_OF_RECORDS_PER_GESTURE
                            break

                if counter_gesture != NUMBER_OF_RECORDS_PER_GESTURE:
                    g = Gesture(i, gestures[i], counter_gesture, self.controller, self.cam, self.q, self.listener, cap,
                                self.id_session, maps_initialized,
                                left_coord=left_coordinates, left_coeff=left_coefficients,
                                right_coord=right_coordinates, right_coeff=right_coefficients, rewrite=rewrite)
                    record = g.record()

                    if not rewrite:
                        list_of_gestures.append(record)
                        counter_gesture += 1
                    else:
                        list_of_gestures.pop(-1)
                        list_of_gestures.append(record)
                        counter_gesture += 1
                        rewrite = False

        # release videocapture
        cap.release()

        list_of_thread = []
        for x in list_of_gestures:
            utils.draw_ui("Saving Session...")
            cv2.waitKey(1)
            list_of_thread.append(x.saveGestureData())

        for th in list_of_thread:
            th.join()
        print("Recording session saved")


def run(controller, cam):

    # inizializzazione picoflexx
    q = queue.Queue()
    listener = MyListener(q, recording=False)
    cam.registerDataListener(listener)
    # cam.startCapture()

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
            print("press E to start new session of recording")
            utils.draw_ui(text="press E to start new session of recording")

        else:
            print("press E to record new session {} or Q to quit".format(session_counter))
            utils.draw_ui(text="press E to record new session {} or Q to quit".format(session_counter))

        k = cv2.waitKey()
        if k == ord('e'):
            pass
        elif k == ord('q'):
            print("end collection")
            utils.save_session_info(session_id=session_counter - 1)
            break

        sess = Session(id_session=session_counter, controller=controller, cam=cam, queue=q, listener=listener)

        # creazione directory per sessione
        directory = "./data/{:03d}".format(sess.id_session)
        if not os.path.exists(directory):
            os.makedirs(directory)

        sess.dir = directory
        print("session {} started".format(sess.id_session))
        sess.run_session()
        session_counter += 1

        # # end collection
        # print("end collection")
        # utils.save_session_info(session_id=session_counter - 1)

        # cam.stopCapture()


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
    # print_camera_info(cam)
    # print("isConnected", cam.isConnected())
    # print("getFrameRate", cam.getFrameRate())

    # LEAP MOTION
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    run(controller, cam)


if __name__ == '__main__':
    main()

