# Estimating axis of rotation using DNN
## 1) Introduction
### Goal of the project
This is a project that can allow me and my friend Adam to recognize the axis of rotation of articulated objects in pictures (in the future in videos). We use a camera which has a function to deliver images in RGB and depth. We will be able to provide better estimations using RGBD Camera, because it gives us additional data.

### What tools we use?
* **Hardware**:
    * Astra Orbec Camera
* **Software**:
    * ROS Noetic 
    * pyTorch
    * ArUco

## 2) Software installaion

### RGBD Camera
We use Orbec Astra Camera to take images. We need to configure it properly to work with ROS and with Python libraries.
Installation: https://github.com/orbbec/ros_astra_camera

### ArUco tags 
We use the **aruco_ros** library to get some positions of articulated objects. Using axis-angle representation we can specify not only the location of the rotational axis but also the angle of opened doors. We can install and configure this library by:

`
sudo apt-get install ros-noetic-aruco-ros
`

Then we need to configure launch file for our Astra Camera:

`
sudo nano /opt/ros/noetic/share/aruco_ros/launch/single.launch
`

single.launch file:
```
  GNU nano 4.8  /opt/ros/noetic/share/aruco_ros/launch/single.launch            
<launch>
   <arg name="markerId"        default="500"/>
   <arg name="markerSize"      default="0.1"/>    <!-- in m -->
   <arg name="eye"             default="left"/>
   <arg name="marker_frame"    default="aruco_marker_frame"/>
   <arg name="ref_frame"       default=""/>  <!-- leave empty and the pose will>
   <arg name="corner_refinement" default="LINES" /> <!-- NONE, HARRIS, LINES, S>
   <node pkg="aruco_ros" type="single" name="aruco_single">
   <remap from="/camera_info" to="/camera/color/camera_info" />
   <remap from="/image" to="/camera/color/image_raw" />
   <param name="image_is_rectified" value="True"/>
   <param name="marker_size"        value="$(arg markerSize)"/>
   <param name="marker_id"          value="$(arg markerId)"/>
   <param name="reference_frame"    value="$(arg ref_frame)"/>   <!-- frame in >
   <param name="camera_frame"       value="stereo_gazebo_$(arg eye)_camera_opti>
   <param name="marker_frame"       value="$(arg marker_frame)" />
   <param name="corner_refinement"  value="$(arg corner_refinement)" />
   </node>
</launch>

```
### Collecting Data
