# Imports
import open3d as o3d
import pandas as pd
import numpy as np
import sys

# PATH = str(sys.argv[1])
FIRST_SECELTED_IMAGE = int(sys.argv[1])
LAST_SELECTED_IMAGE = int(sys.argv[2])

ROOT_DIR = '/home/el_zlociako/Documents/Praca_inzynierska/Dataset/'
PATH = f'{ROOT_DIR}files_ArUco/data_ArUco.csv'
# FIRST_SECELTED_IMAGE = 6
# LAST_SELECTED_IMAGE = 6


# Functions
def CreatePointCloud(RGB_PATH, DEPTH_PATH):
    color_raw = o3d.io.read_image(RGB_PATH) # Reads RGB image
    depth_npy= np.load(DEPTH_PATH) # Reads Depth data
    depth_raw  = o3d.geometry.Image(depth_npy) # Converts depth data into image format
    print(color_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 10000) # Creates RGBD image using TUM format
    PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)) # Creates Point Cloud from rgbd image
    # PointCloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
    return PointCloud

def PickPoints(pcd): # Allows to pick axis of rotation points 
  vis = o3d.visualization.VisualizerWithEditing()
  vis.create_window()
  vis.add_geometry(pcd)
  vis.run()
  vis.destroy_window()
  pc_arr = np.asarray(pcd.points)
  point_id = vis.get_picked_points()
  if(len(point_id) != 2):
    return [None, None], False
  else: 
    return [pc_arr[point_id[0]],pc_arr[point_id[1]]], True

# # Read CSV
Perfect_datafile = pd.read_csv(PATH)



AxisPoints = [[0,0,0],[0,0,0]]
points_1 = [0,0,0]
points_2 = [0,0,0]
two_points = False

PC = CreatePointCloud(ROOT_DIR+Perfect_datafile.loc[FIRST_SECELTED_IMAGE,'rgb_img_I'], ROOT_DIR+Perfect_datafile.loc[FIRST_SECELTED_IMAGE,'depth_img_I'])
while two_points == False:
  points_1, two_points = PickPoints(PC)
two_points = False
PC = CreatePointCloud(ROOT_DIR+Perfect_datafile.loc[FIRST_SECELTED_IMAGE,'rgb_img_II'], ROOT_DIR+Perfect_datafile.loc[FIRST_SECELTED_IMAGE,'depth_img_II'])
while two_points == False:
  points_2, two_points = PickPoints(PC)

# for row in range(Perfect_datafile.shape[0]):
for row in range(FIRST_SECELTED_IMAGE,LAST_SELECTED_IMAGE+1):
    print(f'-----> Image number: {row} <------')

    for i in range(2):
      for j in range(3):
        AxisPoints[i][j] = (points_1[i][j] + points_2[i][j]) / 2
    
    # Perfect_datafile.loc[row, 'x1'] = AxisPoints[0][0]
    # Perfect_datafile.loc[row, 'y1'] = AxisPoints[0][1]
    # Perfect_datafile.loc[row, 'z1'] = AxisPoints[0][2]
    # Perfect_datafile.loc[row, 'x2'] = AxisPoints[1][0]
    # Perfect_datafile.loc[row, 'y2'] = AxisPoints[1][1]
    # Perfect_datafile.loc[row, 'z2'] = AxisPoints[1][2]

    Perfect_datafile.loc[row, 'x1'] = AxisPoints[0][0]*10
    Perfect_datafile.loc[row, 'y1'] = AxisPoints[0][1]*10
    Perfect_datafile.loc[row, 'z1'] = AxisPoints[0][2]*10
    Perfect_datafile.loc[row, 'x2'] = AxisPoints[1][0]*10
    Perfect_datafile.loc[row, 'y2'] = AxisPoints[1][1]*10
    Perfect_datafile.loc[row, 'z2'] = AxisPoints[1][2]*10

    print(f'-----> DATA SAVED <-----')

Perfect_datafile.to_csv(f'{ROOT_DIR}files_ArUco/data_ArUco.csv', index=False)
