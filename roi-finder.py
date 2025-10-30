# save as roi_picker_min.py
import argparse, json, time, cv2, numpy as np

class Picker:
    def __init__(self, rtsp, max_width=1280):
        self.cap = cv2.VideoCapture(rtsp)
        if not self.cap.isOpened():
            raise SystemExit(1)
        ok, frame = self.cap.read()
        if not ok: raise SystemExit(1)
        self.h, self.w = frame.shape[:2]
        self.scale = min(1.0, max_width / float(self.w)) if max_width else 1.0
        self.frame = frame
        self.last_ts = time.time()
        self.polys = {"main": [], "secondary": []}
        self.active = "main"
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI", self.on_mouse)

    def img2scr(self, x, y): return int(round(x*self.scale)), int(round(y*self.scale))
    def scr2img(self, x, y): return int(round(x/self.scale)), int(round(y/self.scale))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.scr2img(x, y)
            self.polys[self.active].append((ix, iy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            pts = self.polys[self.active]
            if pts:
                ix, iy = self.scr2img(x, y)
                i = min(range(len(pts)), key=lambda k:(pts[k][0]-ix)**2+(pts[k][1]-iy)**2)
                pts.pop(i)

    def draw(self):
        disp = self.frame if self.scale==1.0 else cv2.resize(self.frame, (int(self.w*self.scale), int(self.h*self.scale)))
        canvas = disp.copy()
        colors = {"main": (0,255,255), "secondary": (255,255,0)}
        for name, pts in self.polys.items():
            if not pts: continue
            sp = np.array([self.img2scr(x,y) for x,y in pts], np.int32)
            for p in sp: cv2.circle(canvas, tuple(p), 4, colors[name], -1, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [sp], isClosed=len(sp)>=3, color=colors[name], thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"Active: {self.active}  [SPACE switch, Z undo, S save, Q quit]", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2, cv2.LINE_AA)
        return canvas

    def normalized(self):
        def norm(pt): return [round(pt[0]/self.w, 3), round(pt[1]/self.h, 3)]
        return {"main":[norm(p) for p in self.polys["main"]],
                "secondary":[norm(p) for p in self.polys["secondary"]]}

    def run(self):
        while True:
            if time.time()-self.last_ts > 1/15:
                ok, f = self.cap.read()
                if ok:
                    self.frame = f
                    self.last_ts = time.time()
            cv2.imshow("ROI", self.draw())
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                self.active = "secondary" if self.active=="main" else "main"
            elif k == ord('z'):
                if self.polys[self.active]: self.polys[self.active].pop()
            elif k == ord('s'):
                print(json.dumps(self.normalized(), separators=(',',':')))
                break
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--max-width", type=int, default=1280)
    args = ap.parse_args()
    Picker(args.rtsp, args.max_width).run()

if __name__ == "__main__":
    main()