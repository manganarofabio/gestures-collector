import os, inspect, sys
import utils
import cv2
import numpy as np
import queue
import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
import argparse
import shutil


asd = 4

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './leap_lib'

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

parser = argparse.ArgumentParser(description='Gesture collector')
parser.add_argument('--file_info', type=str, default="session_info.json", help="config session file")
parser.add_argument('--n_records_gesture', type=int, default=3, help="number of records per gesture")
parser.add_argument('--n_min_frames', type=int, default=40, help="minimum namber of frames to record per gesture")


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


def init_setup(controller):
    # init leap motion

    print("\nwaiting for maps initialization...")
    while True:
        frame = controller.frame()
        image_l = frame.images[0]
        image_r = frame.images[1]

        if image_l.is_valid and image_r.is_valid:

            left_coordinates, left_coefficients = utils.convert_distortion_maps(image_l)
            right_coordinates, right_coefficients = utils.convert_distortion_maps(image_r)
            maps_initialized = True
            print('maps initialized\n')

            break
        else:
            print('invalid leap motion frame\n', end="")

    # initialize video capture
    while True:
        cap = cv2.VideoCapture(1)
        print(cap)
        if cap:
            return cap, left_coordinates, left_coefficients, right_coordinates, right_coefficients
        else:
            print("\rerror rgb cam", end="")
            return None


def start_single_record(counter, cap, cam, controller, listener, q, left_coord, left_coeff, right_coord, right_coeff):
    directory_rr = "./single_data/{}/R/raw".format(counter)
    directory_lr = "./single_data/{}/L/raw".format(counter)
    directory_ru = "./single_data/{}/R/undistorted".format(counter)
    directory_lu = "./single_data/{}/L/undistorted".format(counter)
    directory_leap_info = "./single_data/{}/leap_motion/tracking_data".format(counter)
    directory_rgb = "./single_data/{}/rgb".format(counter)
    directory_z = "./single_data/{}/depth/z".format(counter)
    directory_ir = "./single_data/{}/depth/ir".format(counter)

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

    if not cap:
        print("error rgb cam")
        exit(-1)
    error_queue = False
    cam.startCapture()
    while True:
        # print(frame_counter)
        if (cv2.waitKey(1) == ord('s') and record_if_valid and frame_counter > args.n_min_frames) \
                or error_queue:
            # print(error_queue)
            break

        frame = controller.frame()

        # controllo di validità per inizio registrazione (OPZIONALE)
        # inizia a registrare i frame solo se leap motion rileva correttamente la mano

        # print(self.listener.recording)
        if utils.hand_is_valid(frame) and not record_if_valid:
            print('\nhand is valid -> ready to start')
            record_if_valid = True
            # print(self.listener.recording)
            print("start gesture")
            listener.setRecording(True)
            # print(self.listener.recording)

        if record_if_valid:
            print("\rrecord valid -> showing {}".format(frame_counter), end="")
            utils.draw_ui(text="recording - press S to stop", circle=True, thickness=-1)
            # RGB CAM
            # get rgb image
            # print(1)
            ret, img_rgb = cap.read()
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
                undistorted_left = utils.undistort(image_l, left_coord, left_coeff, 400, 400)
                undistorted_right = utils.undistort(image_r, right_coord, right_coeff, 400, 400)
                # print(5)

                # show images
                # previous position cv2.imshow()
                # cv2.imshow('img_leap', undistorted_right)

                # json
                json_obj = utils.frame2json_struct(frame)
                # print(6)
                # PICOFLEXX
                # imgs == (z, ir)
                ret_pico, imgs = utils.get_images_from_picoflexx(q)
                # print("ret_pico, z, ir", ret_pico, imgs[0], imgs[1])
                if not ret_pico:
                    print("pico image not valid")
                    error_queue = True
                    continue

                cv2.moveWindow('img_rgb', -700, 325)
                cv2.moveWindow('img_leap', -1150, 400)
                cv2.moveWindow('img_ir', -1500, 600)
                cv2.imshow('img_leap', undistorted_right)
                cv2.imshow('img_rgb', img_rgb)
                cv2.imshow('img_ir', imgs[1])

                # print(7)
                list_img_rr.append(raw_img_r.copy())
                list_img_ru.append(undistorted_right.copy())
                list_img_lr.append(raw_img_l.copy())
                list_img_lu.append(undistorted_left.copy())
                list_img_rgb.append(img_rgb.copy())
                list_json.append(json_obj)
                list_img_z.append(imgs[0].copy())
                list_img_ir.append(imgs[1].copy())

                # list_img_z.append(z.copy())
                # list_img_ir.append(ir.copy())

                frame_counter += 1
                # print(8)
            else:
                print('image not valid')

        else:
            print("\rerror in getting valid leap motion frame", end="")

    # print(self.listener.recording)
    listener.setRecording(False)
    cam.stopCapture()
    cap.release()


    #write single record
    print("saving record")
    utils.draw_ui(text="Saving session...")

    utils.save_single_record(list_img_rr, list_img_ru, list_img_lr, list_img_lu, list_json, list_img_rgb, list_img_z,
                             list_img_ir, directory_rr, directory_ru, directory_lr, directory_lu, directory_leap_info,
                             directory_rgb, directory_z, directory_ir)


def run(controller, cam):

    # inizializzazione picoflexx
    q = queue.Queue()
    listener = MyListener(q, recording=False)
    cam.registerDataListener(listener)
    # cam.startCapture()

    if not os.path.exists("./single_data"):
        os.makedirs("./single_data")
    else:
        shutil.rmtree("./single_data")

    counter = 0
    while True:

        # setup camere

        cap, lcd, lcf, rcd, rcf = init_setup(controller)
        if cap:
            print("setup initialized")
        else:
            return -1

        print("press E to record single record {} or Q to quit".format(counter))
        utils.draw_ui(text="press E to record single record {} or Q to quit".format(counter))

        k = cv2.waitKey()
        if k == ord('e'):
            pass
        elif k == ord('q'):
            print("end single recording")
            break

        start_single_record(counter, cap, cam, controller, listener, q, lcd, lcf, rcd, rcf)
        print("record saved")
        counter += 1


def str2bool(value):
    return value.lower() == 'true'


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

