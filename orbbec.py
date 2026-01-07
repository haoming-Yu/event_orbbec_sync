import os
import cv2
import numpy as np
from pprint import pprint
from typing import Union, Any, Optional
from pyorbbecsdk import OBFormat, VideoFrame

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image

def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


class OrbbecCamera:
    def __init__(self):
        from pyorbbecsdk import (
            Pipeline, Config, 
            OBSensorType, OBStreamType, 
            AlignFilter, PointCloudFilter,
            OBMultiDeviceSyncMode
        )
        
        self.pipeline = Pipeline()
        self.device = self.pipeline.get_device()
        self.config = Config()
        
        # Stream Setup
        d_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.config.enable_stream(d_profiles.get_default_video_stream_profile())
        
        self.has_color = False
        try:
            c_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self.config.enable_stream(c_profiles.get_default_video_stream_profile())
            self.has_color = True
        except:
            self.has_color = False

        sync_config = self.device.get_multi_device_sync_config()
        sync_config.mode = OBMultiDeviceSyncMode.SECONDARY
        sync_config.color_delay_us = 0
        sync_config.depth_delay_us = 0
        sync_config.trigger_to_image_delay_us = 0
        sync_config.trigger_out_enable = True
        sync_config.trigger_out_delay_us = 0
        sync_config.frames_per_trigger = 1
        self.device.set_multi_device_sync_config(sync_config)

        # self.pipeline.enable_frame_sync() # if this is open, 
        # there will be a delay of 14ms or so. 
        # Shut this down but use hardware sync, will shorten the duration to 2ms
        self.pipeline.start(self.config)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.pc_filter = PointCloudFilter()
        self.pc_filter.set_camera_param(self.pipeline.get_camera_param())

        self.first_frame_log_flag = True

    def get_frames(self):
        from pyorbbecsdk import OBFormat
        try:
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                return None, None, None, 0, 0
            
            aligned = self.align.process(frames)
            if aligned is None:
                return None, None, None, 0, 0
            
            aligned = aligned.as_frame_set()
                
            d_frame = aligned.get_depth_frame()
            c_frame = aligned.get_color_frame()

            o_c_ts = c_frame.get_timestamp_us() if c_frame else 0
            o_d_ts = d_frame.get_timestamp_us() if d_frame else 0

            if c_frame is not None and d_frame is not None and self.first_frame_log_flag:
                print(f"[First Color Frame] color frame timestamp from camera booting: {o_c_ts} us , system timestamp: {c_frame.get_system_timestamp_us()} us")
                print(f"[First Depth Frame] depth frame timestamp from camera booting: {o_d_ts} us , system timestamp: {d_frame.get_system_timestamp_us()} us")
                self.first_frame_log_flag = False
            if c_frame is not None:
                c_img = frame_to_bgr_image(c_frame)
            else:
                return None, None, None, 0, 0
            
            if d_frame is None or c_frame is None:
                return None, None, None, 0, 0
            
            point_format = OBFormat.RGB_POINT if self.has_color and c_frame is not None else OBFormat.POINT
            self.pc_filter.set_create_point_format(point_format)
            pc_frame = self.pc_filter.process(aligned)
            if pc_frame is None:
                pc_data = None
            else:
                raw_data = np.frombuffer(pc_frame.get_data(), dtype=np.float32)
                if point_format == OBFormat.RGB_POINT:
                    # RGB_POINT style is [x, y, z, r, g, b, ...]
                    # reshape to (-1, 6)
                    pc_data = raw_data.reshape(-1, 6)
                else:
                    # POINT style is [x, y, z, ...]
                    pc_data = raw_data.reshape(-1, 3)
            
            if d_frame is not None:
                d_img = np.frombuffer(d_frame.get_data(), dtype=np.uint16).reshape(
                    d_frame.get_height(), d_frame.get_width())
            else:
                return None, None, None, 0, 0
            
            return c_img, d_img, pc_data, o_c_ts, o_d_ts
        except Exception as e:
            print(f"Orbbec inner error: {e}")
            return None, None, None, 0, 0

    def stop(self):
        self.pipeline.stop()