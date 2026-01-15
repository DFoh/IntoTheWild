#!/usr/bin/env python3
import os
import cv2
import json
import matplotlib.pyplot as plt

class ZoomPanSelect:
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.selected = None
        self.press = None
        self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)

    def zoom(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1.2 if event.button == 'up' else 1/1.2
        xcenter, ycenter = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        new_w = (xlim[1]-xlim[0]) * scale
        new_h = (ylim[1]-ylim[0]) * scale
        relx = (xlim[1] - xcenter) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ycenter) / (ylim[1] - ylim[0])
        self.ax.set_xlim(xcenter - new_w*(1-relx), xcenter + new_w*relx)
        self.ax.set_ylim(ycenter - new_h*(1-rely), ycenter + new_h*rely)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:  # right-click to pan
            self.press = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())
        elif event.button == 1:  # left-click to select
            self.selected = (int(event.xdata), int(event.ydata))
            plt.close(self.fig)

    def on_release(self, event):
        self.press = None

    def on_drag(self, event):
        if not self.press or event.inaxes != self.ax:
            return
        xpress, ypress, (x0,x1), (y0,y1) = self.press
        dx, dy = xpress - event.xdata, ypress - event.ydata
        self.ax.set_xlim(x0 + dx, x1 + dx)
        self.ax.set_ylim(y0 + dy, y1 + dy)
        self.fig.canvas.draw_idle()


def main():
    video_dir = os.path.join('data')
    results = {}

    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith('.avi'):
            continue
        cam_id = os.path.splitext(fname)[0]
        path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 18*85)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame from {fname}")
            results[cam_id] = None
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f"Camera {cam_id}: scroll to zoom, right-drag to pan, left-click to select")
        selector = ZoomPanSelect(ax)
        plt.show()

        results[cam_id] = selector.selected

    with open('selected_points.json', 'w') as jf:
        json.dump(results, jf, indent=2)
    print('Saved selections to selected_points.json')

if __name__ == '__main__':
    main()
