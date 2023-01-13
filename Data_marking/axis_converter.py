from sympy import Line2D, Point2D, Segment3D, Point3D, Line3D, Symbol, solve
from sympy.plotting import plot as symplot

from sympy import symbols
from numpy import linspace
from sympy import lambdify
import numpy as np
import cv2 as cv
import pandas as pd
import os
import csv
import time
import tqdm
import threading as thr
from shutil import copyfile
import multiprocessing as mp
# from visualiser import Visualiser
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PIL import Image as im

def draw_point(pt, img, img_size, divider, K):
    new_K = np.resize(K, (3, 3))
    X = np.matmul(new_K, [pt[0], pt[1], pt[2]])
    x1 = int(X[0] / X[2] / divider)
    y1 = int(X[1] / X[2] / divider)
    if y1 > img_size[0] or x1 > img_size[1] or y1 < 0 or x1 < 0:
        return False
    image = np.zeros(img_size, float)
    image = cv.circle(image, (x1, y1), radius=1, color=255, thickness=-1)
    # image = cv.flip(image, 1)

    img[np.where(image==255)] = pt[2]
    return True

def get_pixel(pt, img_size, divider, K):
    new_K = np.resize(K, (3, 3))
    X = np.matmul(new_K, [pt[0], pt[1], pt[2]])
    x1 = int(X[0] / X[2] / divider)
    y1 = int(X[1] / X[2] / divider)
    if y1 > img_size[0] or x1 > img_size[1] or y1 < 0 or x1 < 0:
        return
    return x1, y1, pt[2]

def in_picture(pt, img_size, divider, K):
    new_K = np.resize(K, (3, 3))
    X = np.matmul(new_K, [pt[0], pt[1], pt[2]])
    x1 = int(X[0] / X[2] / divider)
    y1 = int(X[1] / X[2] / divider)
    # if not (-5 <= y1 < img_size[0] and -5 <= x1 < img_size[1]):
    #     # print(x1, y1)
    return ((0 <= y1 < img_size[0]) and (0 <= x1 < img_size[1]))


def calc_xy_for_z(pt1, pt2, z):
    t = Symbol('t')
    z_eq = pt1[2] + (pt2[2] - pt1[2]) * t - z
    t = solve(z_eq, t)[0]
    x = pt1[0] + (pt2[0] - pt1[0]) * t
    y = pt1[1] + (pt2[1] - pt1[1]) * t
    return [x,y,z]


def calc_xz_for_y(pt1, pt2, y):
    t = (y - pt1[1])/(pt2[1] - pt1[1])
    x = pt1[0] + (pt2[0] - pt1[0]) * t
    z = pt1[2] + (pt2[2] - pt1[2]) * t
    # if x> 2 or x < -2:
    #     print(x, y, z)
    # print([x, y, z])
    return [x,y,z]

def calc_yz_for_x(pt1, pt2, x):
    # t = Symbol('t')
    t = (x - pt1[0])/(pt2[0] - pt1[0])
    y = pt1[1] + (pt2[1] - pt1[1]) * t
    z = pt1[2] + (pt2[2] - pt1[2]) * t
    # print([x, y, z])
    return [x,y,z]


def Point3D_2_float(point):
    return [float(point.x), float(point.y), float(point.z)]


def list_2_Point3D(point):
    return Point3D(point[0], point[1], point[2])


def perpendicular_point(point1, point2, point0=None):
    if point0 is None:
        point0 = [0., 0., 0.]
    p1 = Point3D(point1[0], point1[1], point1[2])
    p2 = Point3D(point2[0], point2[1], point2[2])
    p0 = Point3D(point0[0], point0[1], point0[2])
    l1 = Line3D(p1, p2)
    l2 = l1.perpendicular_line(p0)
    return Point3D_2_float(l2.p2)


def tensor_from_perpendicular_point(perp_point, point1, point2):
    p1 = Point3D(point1[0], point1[1], point1[2])
    p2 = Point3D(point2[0], point2[1], point2[2])
    p0 = Point3D(perp_point[0], perp_point[1], perp_point[2])
    s1 = Segment3D(p2, p1)
    mid_point = s1.midpoint
    vector = mid_point - p0
    vector = Point3D_2_float(vector)
    return normalize_vector(vector)


def normalize_vector(vector):
    a = np.array(vector)
    mod = np.sqrt(a.dot(a))
    mod = np.mean(mod)
    return vector / mod


def process_axis(point1, point2):
    perp_pt = perpendicular_point(point1, point2)
    tensor = tensor_from_perpendicular_point(perp_pt, point1, point2)
    return perp_pt, tensor


def split_for_threads(dirs, cores=4):
    l = len(dirs)/cores
    out = []
    for i in range(cores):
        a = dirs[round(l*i):round(l*(i+1))]
        out.append(a)
    return out


def change_path(path: str,
                f='/media/kamil/HDD21',
                t='/mnt/9277-709F'):

    return path.replace(f, t)


def convert_csv(dir, core, lock, visualise=False, reconvert=False):
    dir = ['/home/el_zlociako/Documents/AoR_CNN/Segmentacja/files/data.csv']
    for csv_file in dir:
        DIVIDER = 1
        if csv_file.find('validation') != -1:
            continue
        IMG_SIZE = (480//DIVIDER, 640//DIVIDER)
        data = pd.read_csv(csv_file)
        # timestamp = data['time']
        # with lock:
            # length = range(len(data['time']))
            # folder_name = csv_file[csv_file.find('final_dataset')+len('final_dataset')+1:csv_file.find('/rot_axis.csv')]
            # progress = tqdm.tqdm(length, 'Core ' + str(core)+': '+ folder_name + ' ')
        for i, row in data.iterrows():
            # out_dir_ax = csv_file[:csv_file.rfind('/') + 1] + 'axis/'
            # if not os.path.isdir(out_dir_ax):
            #     os.mkdir(out_dir_ax)
            # out_path_ax = out_dir_ax + str(i) + '.npy'
            #
            # out_dir_rgb = csv_file[:csv_file.rfind('/') + 1] + 'rgb/'
            # if not os.path.isdir(out_dir_rgb):
            #     os.mkdir(out_dir_rgb)
            # out_path_rgb = out_dir_rgb + str(timestamp[i]) + '.png'
            #
            # out_dir_dep = csv_file[:csv_file.rfind('/') + 1] + 'depth/'
            # if not os.path.isdir(out_dir_dep):
            #     os.mkdir(out_dir_dep)
            # out_path_dep = out_dir_dep + str(timestamp[i]) + '.npy'
            #
            # if not reconvert and os.path.exists(out_path_ax) and \
            #         os.path.exists(out_path_rgb) and os.path.exists(out_path_dep):
            #     continue
            
            img = np.zeros(IMG_SIZE, float)

            pt1 = [data['x1'][i], data['y1'][i], data['z1'][i]]
            pt2 = [data['x2'][i], data['y2'][i], data['z2'][i]]

            K = [data['K'+str(k)][i] for k in range(9)]

            if pt1 == pt2:
                continue

            switch_to_Y = abs(pt2[0]-pt1[0]) < abs(pt2[1]-pt1[1])

            min_ran = None
            max_ran = None

            if not switch_to_Y:
                x_i = 0.0000001
                while min_ran is None or max_ran is None:
                    [x, y, z] = calc_yz_for_x(pt1, pt2, x_i)
                    if in_picture([x, y, z], IMG_SIZE, 5, K):
                        x_min = x_i
                        while min_ran is None:
                            [x, y, z] = calc_yz_for_x(pt1, pt2, x_min)
                            if not in_picture([x, y, z], IMG_SIZE, 5, K):
                                min_ran = x_min
                                #print(x_min)
                            else:
                                x_min -= 0.01

                        x_max = x_i
                        while max_ran is None:
                            # print(x_max)
                            [x, y, z] = calc_yz_for_x(pt1, pt2, x_max)
                            if not in_picture([x, y, z], IMG_SIZE, 5, K):
                                max_ran = x_max
                                #print(x_max)
                            else:
                                x_max += 0.01

                    else:
                        x_i = (abs(x_i)+0.1)*(x_i/abs(x_i))


            else:
                y_i = 0.0000001
                while min_ran is None or max_ran is None:
                    [x, y, z] = calc_xz_for_y(pt1, pt2, y_i)
                    if in_picture([x, y, z], IMG_SIZE, 5, K):
                        y_min = y_i
                        while min_ran is None:
                            [x, y, z] = calc_xz_for_y(pt1, pt2, y_min)
                            if not in_picture([x, y, z], IMG_SIZE, 5, K):
                                min_ran = y_min
                            else:
                                y_min -= 0.01

                        y_max = y_i
                        while max_ran is None:
                            [x, y, z] = calc_xz_for_y(pt1, pt2, y_max)
                            if not in_picture([x, y, z], IMG_SIZE, 5, K):
                                max_ran = y_max
                            else:
                                y_max += 0.01

                    else:
                        y_i = (abs(y_i) + 0.1) * (y_i / abs(y_i))


            if min_ran is None:
                csv_name = csv_file[:csv_file.find('/rot_axis')]
                print("Not found for " + csv_name)
                continue

            if switch_to_Y:
                step = (max_ran - min_ran) / (IMG_SIZE[1])
            else:
                step = (max_ran - min_ran) / (IMG_SIZE[0])
            ran = np.arange(min_ran, max_ran, step)
            a = []
            if switch_to_Y:
                for y_i in ran:
                    a.append(calc_xz_for_y(pt1, pt2, y_i))
            else:
                for x_i in ran:
                    a.append(calc_yz_for_x(pt1, pt2, x_i))

            a_px = [get_pixel(px, IMG_SIZE, DIVIDER, K) for px in a]
            a_px = [np.array([int(i[0]), int(i[1]), float(i[2])]) for i in a_px if i is not None]

            linspace = np.linalg.norm(a_px[0] - a_px[-1])
            points_on_line = np.linspace(a_px[0], a_px[-1], int(linspace))
            for pt in points_on_line:
                tmp_img = np.zeros(IMG_SIZE, float)
                depth_img = np.load(data['depth_img_II'].iloc[i])
                # depth_img = cv.imread(data['depth_img_I'].iloc[i], cv.IMREAD_UNCHANGED)
                cv.circle(tmp_img, (int(pt[0]), int(pt[1])), radius=1, color=pt[2], thickness=-1)
                # if np.sum(tmp_img*depth_img):
                img[np.where(tmp_img!=0)] = tmp_img[np.where(tmp_img!=0)]
                    # img = tmp_img

            # plt.imshow(img)
            # plt.show()

            np.save(f'/home/el_zlociako/Documents/AoR_CNN/Segmentacja/files/axis/AX{str(i).zfill(5)}', img)
            # np.save(out_path_ax, img)

            # with lock:
            #     progress.update()
        # with lock:
        #     progress.close()
        print("FINISHED "+ csv_file)

#   PARAMS
# parent_dir = '/mnt/9277-709F/RBO Dataset/final_dataset'
cores = 1
RECONVERT = True

threads = []
lock = mp.Lock()
# dirs = os.listdir(parent_dir)
# dirs = [os.path.join(parent_dir, dirn, 'rot_axis.csv') for dirn in dirs if
#         os.path.isdir(os.path.join(parent_dir, dirn))]

# print(dirs)
# dirs = split_for_threads(dirs, cores)

for i in range(cores):
    threads.append(mp.Process(target=convert_csv, args=(['a'], i, lock, False, RECONVERT)))

for thread in threads:
    thread.start()
print("All " + str(cores)+ " threads started")

for thread in threads:
    thread.join()
