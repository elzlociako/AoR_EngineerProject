import open3d as o3d
import numpy as np
import pandas as pd
import sys


def CreatePointCloud(RGB_PATH, DEPTH_PATH):
    color_raw = o3d.io.read_image(RGB_PATH) # Reads RGB image
    color_raw = np.asarray(color_raw)
    depth_npy = np.load(DEPTH_PATH) # Reads Depth data
    color_raw  = o3d.geometry.Image(np.uint8(color_raw))
    depth_raw  = o3d.geometry.Image(np.float32(depth_npy)) # Converts depth data into image format
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,10000) # 
    PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
                                                                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)) # Creates Point Cloud from rgbd image/
    # PointCloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
    return PointCloud

def CreateAxisCloud(DEPTH_PATH):
    depth_npy= np.load(DEPTH_PATH) # Reads Depth data
    depth_raw  = o3d.geometry.Image(np.float32(depth_npy)/1) # Converts depth data into image format
    PointCloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # PointCloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # Flip it, otherwise the pointcloud will be upside down
    return PointCloud

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    numpy_array=np.asarray(pcd.points)
    point_id=vis.get_picked_points()

    return [numpy_array[point_id[0]],numpy_array[point_id[1]]]

def draw_arrow(pcd, points_real, points_extimated):
    lines=[[0,1],[2,3]]
    points = np.concatenate((points_real, points_extimated), axis=0)
    colors = [[1,0,0],[0,1,0]] # Red is REAL and Green is ESTIMATED
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),

    )
    line_set.colors=o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd,line_set])
    
Rude_CSV = pd.read_csv('files/data.csv')

row = int(sys.argv[1])
# row = 56
PC = CreatePointCloud(Rude_CSV.loc[row,'rgb_img_I'], Rude_CSV.loc[row,'depth_img_I'])
PC1 = CreatePointCloud(Rude_CSV.loc[1,'rgb_img_II'], Rude_CSV.loc[1,'depth_img_II'])
# PCA = CreateAxisCloud(f'/home/el_zlociako/Documents/AoR_CNN/Segmentacja/files/axis/AX{str(row).zfill(5)}.npy')
PCA = CreateAxisCloud(f'/home/el_zlociako/Documents/AoR_CNN/Segmentacja/Images/test_seg.npy')

x1 = Rude_CSV.loc[row, 'x1']
y1 = Rude_CSV.loc[row, 'y1']
z1 = Rude_CSV.loc[row, 'z1'] 
x2 = Rude_CSV.loc[row, 'x2']
y2 = Rude_CSV.loc[row, 'y2']
z2 = Rude_CSV.loc[row, 'z2']

# For scaled use
scale_var = 10
x1 = Rude_CSV.loc[row, 'x1']/scale_var
y1 = Rude_CSV.loc[row, 'y1']/scale_var
z1 = Rude_CSV.loc[row, 'z1']/scale_var
x2 = Rude_CSV.loc[row, 'x2']/scale_var
y2 = Rude_CSV.loc[row, 'y2']/scale_var
z2 = Rude_CSV.loc[row, 'z2']/scale_var

REAL = [[x1,y1,z1],[x2,y2,z2]]
ESTIMATED = [[x1,y1,z1],[x2,y2,z2]]
# ESTIMATED = [[     0.07255,      0.33570,     -1.08489], [     0.06919,     -0.20737,     -1.23719]]
# draw_arrow(PC, REAL, ESTIMATED)
# vis = o3d.visualization.VisualizerWithEditing()
# vis.create_window()
# vis.add_geometry(PC)
# vis.run()
# vis.destroy_window()
o3d.visualization.draw_geometries([PC, PCA])