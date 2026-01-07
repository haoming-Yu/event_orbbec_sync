import sys
import time
import numpy as np

if "/usr/lib/python3/dist-packages/" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages/")

from metavision_core.event_io.raw_reader import initiate_device # type: ignore
from metavision_core.event_io import EventsIterator # type: ignore
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm # type: ignore
from metavision_sdk_ui import EventLoop # type: ignore
import metavision_hal # type: ignore
from pprint import pprint

class PropheseeCamera:
    def __init__(self, accumulation_time_us=33000):
        self.device = initiate_device("")
        self.device.get_i_ll_biases().set("bias_diff_on", 76)
        self.device.get_i_ll_biases().set("bias_diff_off", 20)
        self.device.get_i_erc_module().enable(True)
        self.device.get_i_erc_module().set_cd_event_rate(20000000) # 20M maximum event rate to avoid stutter
        i_trigger_in = self.device.get_i_trigger_in()
        i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN) 
        self.iterator = EventsIterator.from_device(device=self.device)
        self.height, self.width = self.iterator.get_size()
        self.accumulation_time_us = accumulation_time_us
        self.frame_gen = None

        # buffer for unsliced events
        self.event_buffer = []
        self.last_trigger_ts = None
        self.active = False

    def start_loop(self, p_queue, slice_queue, stop_event):
        self.frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=self.width, sensor_height=self.height,
                                                            accumulation_time_us=self.accumulation_time_us, fps=30)
        def on_cd_frame_cb(ts, cd_frame): # this timestamp is PeriodicFrameGenerationAlgorithm's timestamp, not precise
            if not p_queue.full():
                p_queue.put({'frame': cd_frame.copy(), 'ts':ts})
        self.frame_gen.set_output_callback(on_cd_frame_cb)
        print("[Prophesee] Waiting for the first trigger to start slicing...")

        for evs in self.iterator:
            EventLoop.poll_and_dispatch()
            self.frame_gen.process_events(evs) # for ui displaying
            
            triggers = self.iterator.reader.get_ext_trigger_events()
            if triggers.size > 0:
                rising_edges = triggers[triggers['p'] == 1]
                for trig in rising_edges:
                    current_trig_ts = trig['t']
                    if not self.active:
                        # first activation
                        self.active = True
                        self.last_trigger_ts = current_trig_ts

                        # check whether current evs has event after trigger
                        if evs.size > 0:
                            post_trigger_evs = evs[evs['t'] > current_trig_ts]
                            if post_trigger_evs.size > 0:
                                self.event_buffer.append(post_trigger_evs.copy())

                        print(f"[Prophesee] First trigger at {current_trig_ts}.")
                    else:
                        # later trigger, do the slicing as normal case
                        # put the current-batch's events(before this trigger, belongs to the last slice) into buffer
                        pre_trigger_mask = evs['t'] <= current_trig_ts
                        if np.any(pre_trigger_mask):
                            self.event_buffer.append(evs[pre_trigger_mask].copy())

                        if len(self.event_buffer) > 0:
                            merged_evs = np.concatenate(self.event_buffer)
                            # slice construction completes
                            if not slice_queue.full():
                                slice_queue.put({
                                    'event_volume': merged_evs,
                                    'start_ts': self.last_trigger_ts,
                                    'end_ts': current_trig_ts
                                })
                            else:
                                print("[WARNNING!!!] [Prophesee] Slice queue is full. Skipping this slice.")
                        
                        # refresh last_trigger_ts(slice start point), 
                        # and put the current-batch's events(after this trigger) into buffer for later slice creation.
                        self.last_trigger_ts = current_trig_ts
                        post_trigger_evs = evs[evs['t'] > current_trig_ts]
                        self.event_buffer = [post_trigger_evs.copy()] if post_trigger_evs.size > 0 else []
            else: 
                # current batch has no trigger, and the activation has been done, 
                # then store all the events into buffer
                if self.active and evs.size > 0:
                    self.event_buffer.append(evs.copy())
            self.iterator.reader.clear_ext_trigger_events()

            if stop_event.is_set():
                break
        
        print("Prophesee: Headless Processing Finished.")
