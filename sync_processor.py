import cv2
import time
import h5py
import numpy as np
from queue import Empty
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class SensorSynchronizer:
    def __init__(self, slice_queue, o_queue_sync, stop_event, record_enabled, output_directory):
        self.slice_queue = slice_queue
        self.o_queue_sync = o_queue_sync
        self.stop_event = stop_event

        self.evs_buffer = deque(maxlen=100)
        self.orb_buffer = deque(maxlen=100)

        self.diff_samples = [] # just for monitoring now
        self.initial_delta_monitoring = None # just for monitoring now
        
        self.delta_orb_to_evs = None # totally trust the first frame capture of hardwares
        self.sync_threshold_us = 5000 # maximum error of matching between ORBBEC and Events -> 5ms
        # the interval between two consecutive frames is 33ms for now (30fps), thus 5ms is far more less than 33ms
        self.warning_threshold_us = 2000

        self.last_monitor_time = time.time()
        self.drift_alpha = 0.01 # fine-tuning delta

        self.record_enabled = record_enabled
        self.output_directory = output_directory
        self.record_idx = 0

        if self.record_enabled:
            worker_count = 8
            self.executor = ThreadPoolExecutor(max_workers=worker_count)
            print(f"[SensorSynchronizer] Record enabled. ThreadPoolExecutor started with {worker_count} workers.")

    def sync_process(self):
        """
        synchronization main process.
        
        :param self: member method
        """
        print("[SensorSynchronizer] Processor Loop Started. Waiting for sensor information to sync...")
        while not self.stop_event.is_set():
            has_data = False
            try:
                evs_data = self.slice_queue.get_nowait()
                self.evs_buffer.append(evs_data)
                has_data = True
            except Empty:
                pass

            try:
                orb_data = self.o_queue_sync.get_nowait()
                self.orb_buffer.append(orb_data)
                has_data = True
            except Empty:
                pass

            if len(self.evs_buffer) > 0 and len(self.orb_buffer) > 0:
                self.sync()

            # buffer length monitoring module, for debugging
            # check buffer status every 5 seconds
            if time.time() - self.last_monitor_time > 5:
                self.monitor_buffers()
                self.last_monitor_time = time.time()

            if not has_data:
                time.sleep(0.001)
        
        if self.record_enabled:
            print("[SensorSynchronizer] Waiting for remaining record tasks to complete...")
            self.executor.shutdown(wait=True)
        print("[SensorSynchronizer] Processor Loop Stopped. Sync Stopped.")

    def monitor_buffers(self):
        e_len = len(self.evs_buffer)
        o_len = len(self.orb_buffer)

        drift_amount = 0
        if self.initial_delta_monitoring is not None and self.delta_orb_to_evs is not None:
            drift_amount = self.delta_orb_to_evs - self.initial_delta_monitoring

        if self.diff_samples:
            avg_diff = sum(self.diff_samples) / len(self.diff_samples)
            max_diff = max(self.diff_samples)
            self.diff_samples.clear()
        else:
            avg_diff = 0
            max_diff = 0

        print(f"[Monitor] EVS:{e_len:2d} | ORB:{o_len:2d} | "
              f"Drift:{drift_amount:6d}us | "
              f"AvgDiff:{avg_diff:5.0f}us | MaxDiff:{max_diff:5.0f}us")
        
        if e_len >= 90 or o_len >= 90:
            print(f"!!! [CRITICAL] Buffer almost full! Check if sync logic is too slow.")

    def sync(self):   
        # basic check: ensure the buffer has data 
        if not self.evs_buffer or not self.orb_buffer:
            return   
         
        # get the oldest event slice
        evs_data = self.evs_buffer[0]
        evs_ts = evs_data['start_ts']

        # initialize delta
        if self.delta_orb_to_evs is None:
            orb_data = self.orb_buffer[0]
            self.delta_orb_to_evs = evs_ts - orb_data['rgb_ts']
            self.initial_delta_monitoring = self.delta_orb_to_evs
            print(f"[SensorSynchronizer] Delta between ORBBEC and Events: {self.delta_orb_to_evs} us")
        
        # put orb frames onto event timeline
        best_match_idx = None
        min_diff = float('inf')

        for i, orb in enumerate(self.orb_buffer):
            # change orb timestamp onto event timeline
            mapped_orb_ts = orb['rgb_ts'] + self.delta_orb_to_evs
            diff = abs(mapped_orb_ts - evs_ts)

            if diff < min_diff:
                min_diff = diff
                best_match_idx = i
        
        if best_match_idx != None and min_diff < self.sync_threshold_us:
            # successfully matched
            matched_evs = self.evs_buffer.popleft() # pop out the matched event slice
            for _ in range(best_match_idx): # pop out the outdated ORBBEC frames
                self.orb_buffer.popleft()
            matched_orb = self.orb_buffer.popleft()

            # finetuning delta.
            # Useful, two devices has different timestamp rate, even they are all us timestamp,
            # the error will be compensated using alpha
            # NOTE: drift is the timestamp correction quantity, thus it is going to be bigger, and it is correct!
            current_error = evs_ts - (matched_orb['rgb_ts'] + self.delta_orb_to_evs)
            self.delta_orb_to_evs += int(current_error * self.drift_alpha)

            self.diff_samples.append(min_diff)
            # DEBUG INFO
            # self.output_sync_info(matched_evs, matched_orb, min_diff)
            if self.record_enabled:
                self.record(matched_evs, matched_orb)
        else:
            # if the oldest evs is older than all mapped_orb in buffer, and out of the threshold
            # thus this evs frame can not be matched with proper ORBBEC frame
            # scatter this evs frame to avoid blocking
            mapped_oldest_orb_ts = self.orb_buffer[0]['rgb_ts'] + self.delta_orb_to_evs
            mapped_newest_orb_ts = self.orb_buffer[-1]['rgb_ts'] + self.delta_orb_to_evs
            if evs_ts < mapped_oldest_orb_ts - self.sync_threshold_us:
                self.evs_buffer.popleft()
            elif evs_ts > mapped_newest_orb_ts + self.sync_threshold_us:
                self.orb_buffer.popleft()
                # effective?? Determine whether this condition will be encountered.
                # Answer: Normally this condition will not be encountered. when ORB frames are all too old, scatter them.
                print(f"[SensorSynchronizer] ORB Frame Dropped. ")

    def output_sync_info(self, evs_data, orb_data, diff):
        mapped_orb_ts = orb_data['rgb_ts'] + self.delta_orb_to_evs

        print(f"--- [SYNC PAIR MATCHED] ---")
        print(f"EVS Start TS: {evs_data['start_ts']} us")
        print(f"ORB Mapped TS: {mapped_orb_ts} us (Original: {orb_data['rgb_ts']})")
        print(f"Precision Error: {diff} us")
        print(f"Event Count: {len(evs_data['event_volume'])}")
        
        # further processing of image + event data
        if diff > self.warning_threshold_us:
            print(f"[WARNING] Sync Error Too Big: {diff} us")
    
    # old version: one executor, no thread pool
    # def record(self, matched_evs, matched_orb):
    #     # the record_enabled flag is checked outside, so no need to check again here
    #     file_idx_str = f"{self.record_idx:06d}"

    #     event_dir = self.output_directory / "event"
    #     frame_dir = self.output_directory / "frame"

    #     # store the event data
    #     event_filename = event_dir / f"{file_idx_str}.h5"
    #     try:
    #         with h5py.File(event_filename, 'w') as h5f:
    #             events = matched_evs['event_volume']
    #             # h5f.create_dataset('events', data=events, compression="gzip", compression_opts=4)
    #             h5f.create_dataset('events', data=events, compression="lzf")
    #             # store meta data
    #             h5f.attrs['start_ts'] = matched_evs['start_ts']
    #             h5f.attrs['end_ts'] = matched_evs['end_ts']
    #     except Exception as e:
    #         print(f"[RECORD ERROR] Failed to write event data to {event_filename}, error: {e}")

    #     # store the frame data
    #     if 'rgb' in matched_orb:
    #         rgb_filename = frame_dir / f"{file_idx_str}_rgb.jpg"
    #         rgb_img = matched_orb['rgb']
    #         cv2.imwrite(str(rgb_filename), rgb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    #     if 'depth' in matched_orb:
    #         depth_filename = frame_dir / f"{file_idx_str}_depth.png"
    #         depth_img = matched_orb['depth']
    #         cv2.imwrite(str(depth_filename), depth_img)

    #     self.record_idx += 1

    def record(self, matched_evs, matched_orb):
        # deep copy in main thread
        rgb_img = matched_orb['rgb'].copy() if 'rgb' in matched_orb else None
        depth_img = matched_orb['depth'].copy() if 'depth' in matched_orb else None

        data_bundle = {
            'idx': self.record_idx,
            'rgb': rgb_img,
            'depth': depth_img,
            'event_volume': matched_evs['event_volume'],
            'start_ts': matched_evs['start_ts'],
            'end_ts': matched_evs['end_ts']
        }

        self.executor.submit(self._async_write_task, data_bundle)
        self.record_idx += 1

    def _async_write_task(self, bundle):
        # writer thread
        idx_str = f"{bundle['idx']:06d}"
        event_dir = self.output_directory / "event"
        frame_dir = self.output_directory / "frame"

        try:
            event_filename = event_dir / f"{idx_str}.h5"
            with h5py.File(event_filename, 'w') as h5f:
                h5f.create_dataset('events', data=bundle['event_volume'], compression="gzip", compression_opts=4)
                h5f.attrs['start_ts'] = bundle['start_ts']
                h5f.attrs['end_ts'] = bundle['end_ts']

            if bundle['rgb'] is not None:
                cv2.imwrite(str(frame_dir / f"{idx_str}_rgb.jpg"), bundle['rgb'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            if bundle['depth'] is not None:
                cv2.imwrite(str(frame_dir / f"{idx_str}_depth.png"), bundle['depth'])

        except Exception as e:
            print(f"[ASYNC RECORD ERROR] Task {idx_str} failed: {e}")