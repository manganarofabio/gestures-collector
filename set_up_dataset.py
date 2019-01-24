import csv
import os.path
import os
import glob
import argparse

# parser = argparse.ArgumentParser
# parser.add_argument('--mode', default='ir', help='type of data (eg. ir)')
# args = parser.args()

# -3 = mode (depth, L, R)


# Dipende dal sistema in cui eseguiamo e dove è messa la cartella data
def get_data_from_img(img_path):
    p = os.path.normpath(img_path)
    return p.split(os.sep)

def check_first_image(relative_name_img):
    s = relative_name_img.split('_')
    return int(s[0]) == 0





def main():

    l = [os.path.abspath(os.path.join(dp, f)) for dp, dn, fn in os.walk(os.path.expanduser("data")) for f in fn]

    # list_sessions, num_of_sessions = os.listdir("data"), len(os.listdir("data"))
    # list_gestures = ['g0', 'g01', 'g02', 'g02', 'g02']

    with open('csv_dataset', 'w', newline="") as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')

        file_writer.writerow(['img_path', 'session_id', 'gesture_id', 'record', 'mode', 'label', 'first'])
        for img in l:
            data = get_data_from_img(img)
            first = check_first_image(data[-1])
            if data[-3] == 'depth' or data[-3] == 'L' or data[-3] == 'R':
                file_writer.writerow([img, data[-6], data[-5], data[-4], data[-2], data[-5], first])
            else:
                file_writer.writerow([img, data[-5], data[-4], data[-3], data[-2], data[4], first])








if __name__ == '__main__':
    main()





