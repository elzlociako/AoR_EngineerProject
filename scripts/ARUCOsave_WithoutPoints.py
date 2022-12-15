#!/usr/bin/env python3
from __future__ import print_function
from ctypes import sizeof
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import os
import pandas as pd 
import numpy as np
import open3d as o3d

# Saving without data loss
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

ArUco_positon = [0.0,0.0,0.0]
ArUco_orientation = [0.0,0.0,0.0,0.0]

part_num = 1
counter = 0

def CreatePointCloud(RGB_PATH, DEPTH_PATH):
    color_raw = o3d.io.read_image(RGB_PATH) # Reads RGB image
    depth_npy= np.load(DEPTH_PATH) # Reads Depth data
    depth_raw  = o3d.geometry.Image(depth_npy) # Converts depth data into image format
    print(color_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw) # Creates RGBD image using TUM format
    PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)) # Creates Point Cloud from rgbd image
    PointCloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
    return PointCloud

def PickPoints(pcd): # Allows to pick axis of rotation points 
  vis = o3d.visualization.VisualizerWithEditing()
  vis.create_window()
  vis.add_geometry(pcd)
  vis.run()
  vis.destroy_window()
  pc_arr = np.asarray(pcd.points)
  point_id = vis.get_picked_points()
  if(len(point_id) < 2):
    return [None, None], False
  else:
    return [pc_arr[point_id[0]],pc_arr[point_id[1]]], True

class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    self.sub_RGB = rospy.Subscriber("/camera/color/image_raw",Image, self.callbackRGB)
    self.sub_DEPTH = rospy.Subscriber("/camera/depth/image_raw", Image, self.callbackDEPTH)
    self.subINFO_RGB = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.callbackINFO_RGB)
    self.subINFO_DEPTH = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.callbackINFO_DEPTH)
    
    self.sub_ArUcoPosition = rospy.Subscriber("/aruco_single/pose", PoseStamped, self.callback_ArUcoPose)
    self.sub_ArUcoRGB = rospy.Subscriber("/aruco_single/result", Image, self.callbackArUcoRGB)

  def callbackDEPTH(self, data):
    global img_DPH
    try:
      img_DPH = self.bridge.imgmsg_to_cv2(data, "32FC1")
    except CvBridgeError as e:
      print(e)

  def callbackRGB(self, data):
    global img_RGB
    try:
      img_RGB = self.bridge.imgmsg_to_cv2(data, "bgr8")
      img_RGB_copy = img_RGB.copy()
    except CvBridgeError as e:
      print(e)

  def callback_ArUcoPose(self, pose_stamped_msg):
      global ArUco_positon, ArUco_orientation
      try:
        ArUco_positon = [pose_stamped_msg.pose.position.x, pose_stamped_msg.pose.position.y, pose_stamped_msg.pose.position.z]
        ArUco_orientation = [pose_stamped_msg.pose.orientation.x, pose_stamped_msg.pose.orientation.y, pose_stamped_msg.pose.orientation.z, pose_stamped_msg.pose.orientation.w]

        # print("X: %f"%ArUco_positon[0],"Y: %f"%ArUco_positon[1], "Z: %f"%ArUco_positon[2])
        # print("OX: %f"%ArUco_orientation[0],"OY: %f"%ArUco_orientation[1],
        #       "OZ: %f"%ArUco_orientation[2],"OW: %f"%ArUco_orientation[3])
      except CvBridgeError as e:
        print(e)


  def callbackArUcoRGB(self, data):
    global img_ArUcoRGB, img_ArUcoRGB_copy
    try:
      img_ArUcoRGB = self.bridge.imgmsg_to_cv2(data, "bgr8")
      img_ArUcoRGB_copy = img_ArUcoRGB.copy()
    except CvBridgeError as e:
      print(e)

    cv2.putText(img_ArUcoRGB_copy, 'IMAGE: %d'%counter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_ArUcoRGB_copy, 'PART: %d'%part_num, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img_ArUcoRGB_copy, ("X: %03f | Y: %03f | Z: %03f"%(ArUco_positon[0], ArUco_positon[1], ArUco_positon[2])), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(img_ArUcoRGB_copy, ("OX: %03f | OY: %03f | OZ: %03f | OW: %0ff"%(ArUco_orientation[0], ArUco_orientation[1], ArUco_orientation[2], ArUco_orientation[3])), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 100), 1, cv2.LINE_AA)    

    key = cv2.waitKey(1)
    cv2.imshow('AruCo is Cool!', img_ArUcoRGB_copy)

    if key == ord('s'):
      SaveImg()
    if key == ord('p'):
      rospy.signal_shutdown("Exit")  

  def callbackINFO_RGB(self, data):
    global imgI_RGB
    try:
      imgI_RGB = data
    except CvBridgeError as e:
      print(e)
      
  def callbackINFO_DEPTH(self, data):
    global imgI_DPH
    try:
      imgI_DPH = data
    except CvBridgeError as e:
      print(e)

def SaveImg():
  global counter, part_num, axis_points, object_name, aruco_pos1, aruco_orient1, aruco_pos2, aruco_orient2
  axis_points = [[0,0,0] , [0,0,0]]
  if(part_num == 1):
    cv2.imwrite('files_ArUco/images/rgb_img_I/arucoRGB%05d.png'%counter, img_RGB)
    np.save('files_ArUco/images/depth_img_I/arucoD%05d'%counter, img_DPH)
    part_num = 2
    PC = CreatePointCloud('files_ArUco/images/rgb_img_I/arucoRGB%05d.png'%counter,
     'files_ArUco/images/depth_img_I/arucoD%05d.npy'%counter)
    aruco_pos1 = ArUco_positon
    aruco_orient1 = ArUco_orientation
    o3d.visualization.draw_geometries([PC])
    print(aruco_pos1)
    print(aruco_orient1)
    #   axis_points_1, correctly_picked = PickPoints(PC)

    print("First image was taken")
  else:
    cv2.imwrite('files_ArUco/images/rgb_img_II/arucoRGB%05d.png'%counter, img_RGB)
    np.save('files_ArUco/images/depth_img_II/arucoD%05d'%counter, img_DPH)
    np.save('files_ArUco/cam_info/arucoINFO%05d'%counter, imgI_RGB)
    part_num = 1

    PC = CreatePointCloud('files_ArUco/images/rgb_img_II/arucoRGB%05d.png'%counter,
     'files_ArUco/images/depth_img_II/arucoD%05d.npy'%counter)
    aruco_pos2 = ArUco_positon
    aruco_orient2 = ArUco_orientation
    o3d.visualization.draw_geometries([PC])
    print(aruco_pos2)
    print(aruco_orient2)
    print("Second image was taken")

    object_name = input("Instert object name: ")
    CollectData()
    counter += 1

def CollectData():
  global counter
  df = pd.DataFrame(
    [
      [
        object_name,
        'files_ArUco/images/rgb_img_I/arucoBGR%05d.png'%counter, 
        'files_ArUco/images/rgb_img_II/arucoBGR%05d.png'%counter, 
        'files_ArUco/images/depth_img_I/arucoD%05d.npy'%counter,
        'files_ArUco/images/depth_img_II/arucoD%05d.npy'%counter,
        'files_ArUco/cam_info/arucoINFO%05d.npy'%counter, 
        np.asarray(axis_points[0][0]),
        np.asarray(axis_points[0][1]),
        np.asarray(axis_points[0][2]),
        np.asarray(axis_points[1][0]),
        np.asarray(axis_points[1][1]),
        np.asarray(axis_points[1][2]),
        np.asarray(aruco_pos1[0]),
        np.asarray(aruco_pos1[1]),
        np.asarray(aruco_pos1[2]),
        np.asarray(aruco_orient1[0]),
        np.asarray(aruco_orient1[1]),
        np.asarray(aruco_orient1[2]),
        np.asarray(aruco_orient1[3]),
        np.asarray(aruco_pos2[0]),
        np.asarray(aruco_pos2[1]),
        np.asarray(aruco_pos2[2]),
        np.asarray(aruco_orient2[0]),
        np.asarray(aruco_orient2[1]),
        np.asarray(aruco_orient2[2]),
        np.asarray(aruco_orient2[3]),

      ]
    ],
    columns=
    [
      "object_name", "rgb_img_I", "rgb_img_II",
       "depth_img_I", "depth_img_II", "camera_info",
        "x1", "y1", "z1", "x2", "y2", "z2",
        "aurco_posX1", "aruco_posY1", "aruco_posZ1", "aurco_orntX1", "aurco_orntY1", "aurco_orntZ1", "aurco_orntW1",
        "aurco_posX2", "aruco_posY2", "aruco_posZ2", "aurco_orntX2", "aurco_orntY2", "aurco_orntZ2", "aurco_orntW2"
    ]
  )
 

  # create data.csv file if not exists
  if not os.path.isfile('files_ArUco/data_ArUco.csv'):
    df.to_csv('files_ArUco/data_ArUco.csv', index=False)
  else: # else it exists so append without header
    df.to_csv('files_ArUco/data_ArUco.csv', mode='a', header=False, index=False)

def SET_counter():
  global counter
  csv_file = pd.read_csv('files_ArUco/data_ArUco.csv')
  last_element = csv_file["rgb_img_I"].iloc[-1]
  num = ""
  for c in last_element:
      if c.isdigit():
          num = num + c
  counter = int(num) + 1

def main(args):
  if os.path.isfile('files_ArUco/data_ArUco.csv'):
    SET_counter()

  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)

