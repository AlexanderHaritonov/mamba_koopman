# Coordinate transformations

Where the project changes coordinate frames. The world frame of a drive is East-North-Up in metres, with its origin at the drive's first GPS position.

## Ego poses

In compute_ego_poses.py:

- **GPS to world:** latitude, longitude and altitude become local metres with the Mercator projection of the devkit. The scale comes from the first frame's latitude, and the first position becomes the origin.
- **IMU orientation:** roll, pitch and yaw become the rotation Rz·Ry·Rx. Together with the position, this gives `T_world_imu` for each frame.
- **Velodyne to camera 2:** Velodyne → camera 0 → rectified camera 0 → rectified camera 2. `P_rect_02` is split into the intrinsics K and the offset of camera 2 from camera 0.
- **World poses:** `T_world_velo = T_world_imu · inv(T_velo_imu)` and `T_world_cam2 = T_world_velo · inv(T_cam2_velo)`.
- **World to image_02 pixels:** only described in the module docstring, not in code yet.

## Vehicles into the world

In transform_vehicle_poses_to_world.py:

- **Vehicle position:** each position in the Velodyne frame goes into the world as `R · p + t`, with `T_world_velo` of that frame.
- **Vehicle yaw:** the forward vector of the box is rotated into the world, the yaw is taken with `atan2`, and the yaws of a track are unwrapped.
- **Ego car:** a fixed point under the car (`EGO_POINT_VELO`) goes into the world. The yaw comes from the Velodyne +x axis.
- **Height check:** the road height is the Velodyne z minus 1.73 m.

## Target-car frame

In car_frame.py, used by cut_tracks_into_samples.py:

- **World to car frame:** a 2D shift and rotation to the car's pose at time t, so the car is at the origin and faces +x. The yaw is reduced by the anchor yaw.
- **Car frame to world:** the inverse, to bring predictions back into the world.
- **Use:** cut_tracks_into_samples.py re-centres the past and future of each window. A check there converts to the car frame and back, and compares with the world poses.

## Files without transforms

calibration.py only reads the R, T, `R_rect_00` and `P_rect_02` matrices. extract_vehicle_poses.py keeps the tracklet poses in the Velodyne frame as they are.
