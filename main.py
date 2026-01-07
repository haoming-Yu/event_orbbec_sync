import os
import sys
import multiprocessing as mp
import numpy as np

def sync_processor_worker(slice_queue, o_queue_sync, stop_event):
    from sync_processor import SensorSynchronizer

    try:
        synchronizer = SensorSynchronizer(slice_queue, o_queue_sync, stop_event)
        synchronizer.sync_process()
    except Exception as e:
        print(f"Sync Process Error: {e}")
    finally:
        stop_event.set()

def prophesee_worker(p_queue, slice_queue, stop_event):
    from prophesee import PropheseeCamera
    
    try:
        p_cam = PropheseeCamera()
        print("Prophesee Process: Initialized.")
        p_cam.start_loop(p_queue, slice_queue, stop_event)
    except Exception as e:
        print(f"Prophesee Process Error: {e}")
    finally:
        stop_event.set()

def orbbec_worker(o_queue, o_queue_sync, stop_event):
    from orbbec import OrbbecCamera
    try:
        o_cam = OrbbecCamera()
        print("Orbbec Process: Initialized.")
        
        while not stop_event.is_set():
            rgb, depth, pc, o_c_ts, o_d_ts = o_cam.get_frames()
            
            if rgb is not None:
                if o_queue.full():
                    try:
                        p_queue.get_nowait()
                    except:
                        pass
                if o_queue_sync.full():
                    print("[WARNING!!!] [Orbbec] orbbec sync queue is full. Dropping frame.")
                
                o_queue.put({
                    'rgb': rgb.copy(),
                    'depth': depth.copy(),
                    'pc': pc[::15].copy() if pc is not None else None
                })
                o_queue_sync.put({
                    'rgb': rgb.copy(),
                    'depth': depth.copy(),
                    'rgb_ts': o_c_ts,
                    'depth_ts': o_d_ts
                })
    except Exception as e:
        print(f"Orbbec Process Error: {e}")
    finally:
        o_cam.stop()
        print("Orbbec Process: Stopped.")

def run_ui(p_queue, o_queue, o_queue_sync, slice_queue, stop_event):
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, 
        QVBoxLayout, QHBoxLayout, QGridLayout, 
        QPushButton
    )
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont # Correct module for QFont
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from ui_components import ImageDisplayWidget
    from pprint import pprint

    class PointCloudWidget(gl.GLViewWidget):
        def __init__(self):
            super().__init__()
            self.setMinimumSize(400, 300)
            self.opts['distance'] = 1370
            self.opts['elevation'] = -90
            self.opts['azimuth'] = 270
            self.opts['center'] = pg.Vector(0, 0, 1000)
            self.setBackgroundColor('#050505')

            self.lastPose = None

            grid = gl.GLGridItem()
            grid.scale(200, 200, 1) # This modifies 'grid' in place
            self.addItem(grid)      # Add the modified 'grid' object
            # ------------------------------------------

            self.scatter = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)), 
                color=(0, 1, 0.8, 0.5), 
                size=2,
                pxMode=True
            )
            self.addItem(self.scatter)
            self.setCameraPosition(distance=self.opts['distance'])

        def mousePressEvent(self, ev):
            self.lastPose = ev.pos()

        def mouseMoveEvent(self, ev):
            if self.lastPose is None:
                return
            
            diff = ev.pos() - self.lastPose
            self.lastPose = ev.pos()
            if ev.buttons() == Qt.LeftButton:
                # Left-button drag: Rotate (Azimuth, Elevation) 
                # Multiplying by 0.5 is to make rotation smoother and less sensitive
                self.opts['azimuth'] -= diff.x() * 0.5
                self.opts['elevation'] += diff.y() * 0.5
                # Limit the elevation range to prevent flipping
                self.opts['elevation'] = max(-90, min(90, self.opts['elevation']))
                
            elif ev.buttons() == Qt.MidButton or (ev.buttons() == Qt.LeftButton and ev.modifiers() == Qt.ControlModifier):
                # Middle-button (or Ctrl+Left-button) drag: Pan the view center 
                # Adjust panning sensitivity based on the current zoom distance
                dist = self.opts['distance'] * 0.001
                self.pan(diff.x() * dist, diff.y() * dist, 0, relative='view')
                
            elif ev.buttons() == Qt.RightButton:
                # use right button to Zoom
                self.opts['distance'] *= 0.999**diff.y()

            # debugging info
            # pprint(f"Azimuth: {self.opts['azimuth']}, Elevation: {self.opts['elevation']}")
            # update the ui
            self.update()

        def mouseReleaseEvent(self, ev):
            self.lastPose = None

        def wheelEvent(self, ev):
            delta = ev.angleDelta().y()
            self.opts['distance'] *= 0.999**delta
            # debugging info
            # pprint(f"Distance: {self.opts['distance']}")
            self.update()

        def update_pc(self, pc_data):
            if pc_data is not None and pc_data.size > 0:
                # check if z coordinate is positive
                # z coordinate is at index 2
                mask = pc_data[:, 2] > 0
                valid_points = pc_data[mask]
                
                if valid_points.shape[1] == 6:
                    # color mode
                    pos = valid_points[:, :3]
                    # get color (r, g, b)，nomalized to [0, 1]
                    colors = valid_points[:, 3:] / 255.0
                    colors = colors.clip(0, 1) * 0.8
                    # colors = valid_points[:, 3:]
                    # add alpha channel
                    rgba = np.ones((colors.shape[0], 4))
                    rgba[:, :3] = colors
                    rgba[:, 3] = 0.6
                    
                    self.scatter.setData(pos=pos, color=rgba)
                else:
                    # no color
                    self.scatter.setData(pos=valid_points, color=(0, 1, 0.8, 0.8))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Multiprocess Sensor Fusion")
            self.resize(1280, 800)

            self.p_prophesee = None
            self.p_orbbec = None
            
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self.start_button = QPushButton("START SENSORS")
            self.start_button.setFixedHeight(40)
            self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
            self.start_button.setStyleSheet("background-color: #2E7D32; color: white; border-radius: 10px;")
            self.start_button.clicked.connect(self.toggle_sensors)
            layout.addWidget(self.start_button)

            # style setting for beautiful display
            self.setStyleSheet("QMainWindow { background-color: #121212; }")
        
            # button style
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #2E7D32;
                    color: white;
                    border-radius: 15px;
                    border: 1px solid #388E3C;
                }
                QPushButton:hover {
                    background-color: #388E3C;
                }
                QPushButton:pressed {
                    background-color: #1B5E20;
                }
            """)

            grid = QGridLayout()

            grid.setSpacing(15)
            grid.setContentsMargins(10, 10, 10, 10)

            self.v_ev = ImageDisplayWidget("PROPHESEE EVENT")
            self.v_rgb = ImageDisplayWidget("ORBBEC RGB")
            self.v_depth = ImageDisplayWidget("ORBBEC DEPTH")
            self.v_pc = PointCloudWidget()
            
            grid.addWidget(self.v_ev, 0, 0)
            grid.addWidget(self.v_rgb, 0, 1)
            grid.addWidget(self.v_depth, 1, 0)
            grid.addWidget(self.v_pc, 1, 1)
            layout.addLayout(grid)

            # Polling Timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.poll_queue)
            self.timer.start(10) # ~33 FPS

        def toggle_sensors(self):
            if self.p_prophesee is None or not self.p_prophesee.is_alive():
                stop_event.clear()
                self.p_prophesee = mp.Process(target=prophesee_worker, args=(p_queue, slice_queue, stop_event))
                self.p_orbbec = mp.Process(target=orbbec_worker, args=(o_queue, o_queue_sync, stop_event))
                self.p_sync_process = mp.Process(target=sync_processor_worker, args=(slice_queue, o_queue_sync, stop_event))

                self.p_prophesee.daemon = True
                self.p_orbbec.daemon = True
                self.p_sync_process.daemon = True

                self.p_prophesee.start()
                self.p_orbbec.start()
                self.p_sync_process.start()
                
                self.start_button.setText("STOP SENSORS")
                self.start_button.setStyleSheet("background-color: #C62828; color: white; border-radius: 10px;")
                
            else:
                stop_event.set()

                p_queue.cancel_join_thread()
                o_queue.cancel_join_thread()
                o_queue_sync.cancel_join_thread()
                slice_queue.cancel_join_thread()

                self.p_prophesee.join(timeout=1)
                self.p_orbbec.join(timeout=1)
                self.p_sync_process.join(timeout=1)
                self.start_button.setText("START SENSORS")
                self.start_button.setStyleSheet("background-color: #2E7D32; color: white; border-radius: 10px;")

        def poll_queue(self):
            # for visualization
            while not p_queue.empty():
                try:
                    ev_queue_element = p_queue.get_nowait()
                    ev_frame = ev_queue_element['frame']
                    # frame timestamp, not precise, just for debugging
                    # ev_frame_ts = ev_queue_element['ts']
                    # print(f"[POLL QUEUE] Event Frame Timestamp: {ev_frame_ts}")
                    self.v_ev.update_frame(ev_frame)
                except: break

            while not o_queue.empty():
                try:
                    data = o_queue.get_nowait()
                    if 'rgb' in data: self.v_rgb.update_frame(data['rgb'])
                    if 'depth' in data: self.v_depth.update_frame(data['depth'])
                    if 'pc' in data: self.v_pc.update_pc(data['pc'])
                except: break

    # Apply environment scaling before app creation
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1.0" # Adjusted for better fit

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    
    p_queue = mp.Queue(maxsize=1)
    o_queue = mp.Queue(maxsize=1)
    o_queue_sync = mp.Queue(maxsize=5)
    slice_queue = mp.Queue(maxsize=5)
    stop_event = mp.Event()

    try:
        run_ui(p_queue, o_queue, o_queue_sync, slice_queue, stop_event)
    except KeyboardInterrupt:
        stop_event.set()