#!/usr/bin/env python3
"""Detect a known-size dark circular disc and estimate its distance.

Designed for Raspberry Pi 4 / 4 GB at 640x480. It uses a grayscale mask,
contours, ellipse fitting, temporal median filtering, and a Laplacian focus
check. Camera calibration is strongly recommended: it removes lens distortion
before measuring the ellipse.

The reported distance is:
  z     = distance along the camera optical axis
  range = straight-line distance from the camera centre to the disc centre

For a planar circular disc, the ellipse minor/major ratio estimates the disc
plane tilt. The major axis is used for depth because it is much less sensitive
to tilt than the minor axis. This is a practical monocular approximation; for
high-accuracy metrology, use a stereo/depth camera or add an ArUco marker.

Examples:
  python3 disc_distance.py --diameter 0.273 --camera 0
  python3 disc_distance.py calibrate --distance 1.0 --diameter 0.273
  python3 disc_distance.py --camera-matrix camera.json --show-mask

Dependencies: opencv-python(-headless), numpy
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# One OpenCV worker is usually faster and uses less RAM than a thread pool on
# a 4 GB Pi for this small 640x480 workload.
cv2.setNumThreads(1)
cv2.setUseOptimized(True)


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray

    @property
    def f_mean(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_json(cls, filename: str) -> "CameraModel":
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        k = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        d = np.asarray(data.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
        return cls(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), d)

    def save(self, filename: str) -> None:
        data = {
            "camera_matrix": self.matrix().tolist(),
            "distortion": self.distortion.reshape(-1).tolist(),
        }
        Path(filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class Detection:
    center: tuple[float, float]
    major_px: float
    minor_px: float
    tilt_deg: float
    z_m: float
    range_m: float
    confidence: float
    ellipse: tuple
    contour: Optional[np.ndarray] = None


class OnlineSelfTuner:
    """Conservative persistent adaptation from high-confidence detections."""
    def __init__(self, args, filename):
        self.args, self.filename, self.observations = args, Path(filename), 0
        self._load()
    def _load(self):
        try:
            d=json.loads(self.filename.read_text(encoding="utf-8"))
            self.args.black=int(np.clip(d.get("black", self.args.black),35,95))
            self.args.contrast=int(np.clip(d.get("contrast", self.args.contrast),8,35))
            self.args.reference_sharpness=float(np.clip(d.get("reference_sharpness",self.args.reference_sharpness),30.0,300.0))
        except (OSError,ValueError,TypeError,json.JSONDecodeError):
            pass
    def save(self):
        self.filename.parent.mkdir(parents=True,exist_ok=True)
        d={"black":int(self.args.black),"contrast":int(self.args.contrast),"reference_sharpness":float(self.args.reference_sharpness),"observations":self.observations}
        tmp=self.filename.with_suffix(self.filename.suffix+".tmp")
        tmp.write_text(json.dumps(d,indent=2),encoding="utf-8"); tmp.replace(self.filename)
    def observe(self, frame, hit):
        if hit.confidence < max(70.0,self.args.min_command_confidence+10.0): return
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); x,y=map(int,hit.center)
        r=max(8,int(hit.major_px*0.35)); x0,x1=max(0,x-r),min(gray.shape[1],x+r); y0,y1=max(0,y-r),min(gray.shape[0],y+r)
        patch=gray[y0:y1,x0:x1]
        if patch.size==0: return
        sharp=float(cv2.Laplacian(patch,cv2.CV_64F).var()); level=float(np.percentile(patch,60))
        bg=cv2.GaussianBlur(gray,(31,31),0); contrast=float(max(0.0,int(bg[y,x])-int(gray[y,x]))); a=0.003
        self.args.reference_sharpness=float(np.clip((1-a)*self.args.reference_sharpness+a*max(30.0,sharp),30.0,300.0))
        self.args.black=int(round((1-a)*self.args.black+a*float(np.clip(level+20.0,35.0,95.0))))
        self.args.contrast=int(round((1-a)*self.args.contrast+a*float(np.clip(contrast*0.55,8.0,35.0))))
        self.observations+=1
        if self.observations%60==0: self.save()

class ViewpointServo:
    """Optional bounded GPIO servo for vertical target recentering."""
    def __init__(self,args):
        self.args=args; self.servo=None; self.angle=float(args.servo_center); self.last_update=0.0
        if args.servo_pin is None: return
        try:
            from gpiozero import AngularServo
            self.servo=AngularServo(args.servo_pin,min_angle=args.servo_min_angle,max_angle=args.servo_max_angle,min_pulse_width=args.servo_min_pulse,max_pulse_width=args.servo_max_pulse)
            self.servo.angle=self.angle
        except Exception as exc:
            raise RuntimeError(f"Could not initialize servo on GPIO {args.servo_pin}: {exc}") from exc
    def update(self,target_y,frame_h,confidence):
        if self.servo is None or confidence<self.args.min_command_confidence: return
        now=time.monotonic()
        if now-self.last_update<1.0/max(self.args.servo_update_hz,1.0): return
        error=float((target_y-frame_h/2.0)/max(frame_h/2.0,1.0))
        if abs(error)<self.args.servo_y_deadband: return
        self.angle=float(np.clip(self.angle+self.args.servo_y_direction*error*self.args.servo_y_gain,self.args.servo_min_angle,self.args.servo_max_angle))
        self.servo.angle=self.angle; self.last_update=now
    def close(self):
        if self.servo is not None: self.servo.detach()

class RobotJsonSender:
    """Send one self-contained navigation JSON object per processed frame."""

    def __init__(self, host: str, port: int, camera: CameraModel, args: argparse.Namespace):
        self.address = (host, port)
        self.camera = camera
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if host == "255.255.255.255" or host.endswith(".255"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sequence = 0

    def send(self, detections: list[Detection], frame_w: int, frame_h: int,
             photo_path: Optional[str] = None) -> None:
        self.sequence += 1
        target = detections[0] if detections and detections[0].confidence >= self.args.min_command_confidence else None
        if target is None:
            packet = {
                "type": "disc_navigation", "sequence": self.sequence,
                "timestamp": time.time(), "count": 0, "valid": False,
                "target": None,
                "command": {"forward": 0.0, "turn": 0.0, "stop": True},
                "detections": [], "photo": photo_path,
            }
        else:
            # Positive turn means target is to the robot's right. The robot can
            # map this normalized value to its own wheel-speed controller.
            x_error = (target.center[0] - self.camera.cx) / max(self.camera.cx, frame_w / 2.0)
            x_error = float(np.clip(x_error, -1.0, 1.0))
            stop = target.range_m <= self.args.stop_distance
            turn = float(np.clip(x_error * self.args.turn_gain, -1.0, 1.0))
            forward = 0.0 if stop else float(np.clip(
                (target.range_m - self.args.stop_distance) / self.args.approach_distance,
                0.0, 1.0))
            # Do not drive quickly while a large steering correction is needed.
            forward *= max(0.10, 1.0 - 0.80 * abs(turn))
            packet = {
                "type": "disc_navigation", "sequence": self.sequence,
                "timestamp": time.time(), "count": len(detections), "valid": True,
                "target": {"id": 0, "x_px": target.center[0], "y_px": target.center[1],
                           "z_m": target.z_m, "range_m": target.range_m,
                           "tilt_deg": target.tilt_deg, "confidence": target.confidence},
                "command": {"forward": forward, "turn": turn, "stop": stop},
                "detections": [
                    {"id": i, "x_px": d.center[0], "y_px": d.center[1], "z_m": d.z_m,
                     "range_m": d.range_m, "tilt_deg": d.tilt_deg, "confidence": d.confidence}
                    for i, d in enumerate(detections)
                ],
                "photo": photo_path,
            }
        self.sock.sendto((json.dumps(packet, separators=(",", ":")) + "\n").encode(), self.address)

    def close(self) -> None:
        self.sock.close()


class PhotoTrigger:
    """Capture one JPEG on the transition from fewer than N discs to N discs."""

    def __init__(self, directory: str, threshold: int, quality: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.quality = int(np.clip(quality, 50, 100))
        self.armed = True

    def maybe_capture(self, frame: np.ndarray, detections: list[Detection]) -> Optional[str]:
        count = len(detections)
        if count < self.threshold:
            self.armed = True
            return None
        if not self.armed:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        path = self.directory / f"disc_{count}_{stamp}.jpg"
        ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self.armed = False
        return str(path) if ok else None


class DiscDetector:
    def __init__(self, args: argparse.Namespace, camera: CameraModel):
        self.args = args
        self.camera = camera
        self.history: deque[tuple[float, float]] = deque(maxlen=args.smooth)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.lock_center: Optional[np.ndarray] = None
        self.lock_misses = 0
        self.last_view: Optional[np.ndarray] = None
        self._distorted = bool(np.any(np.abs(camera.distortion) > 1e-12))
        self._map1 = self._map2 = None

    def mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use a strict mask plus a local adaptive fallback. The fallback keeps
        # exposure changes from making a clearly visible dark disc disappear.
        _, dark = cv2.threshold(gray, self.args.black, 255, cv2.THRESH_BINARY_INV)
        background = cv2.GaussianBlur(gray, (31, 31), 0)
        contrast = cv2.subtract(background, gray)
        _, local = cv2.threshold(contrast, self.args.contrast, 255, cv2.THRESH_BINARY)
        strict = cv2.bitwise_and(dark, local)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
        adaptive_limit = int(np.clip(max(self.args.black, np.percentile(gray, 40)), 70, 135))
        _, dark_adaptive = cv2.threshold(gray, adaptive_limit, 255, cv2.THRESH_BINARY_INV)
        fallback = cv2.bitwise_and(adaptive, dark_adaptive)
        result = cv2.bitwise_or(strict, fallback)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel, iterations=1)
        return cv2.morphologyEx(result, cv2.MORPH_CLOSE, self.kernel, iterations=2)

    def find_all(self, frame: np.ndarray) -> list[Detection]:
        # Correcting distortion before measurement is important with wide-angle lenses.
        # Build maps once; cv2.undistort() would rebuild them on every frame.
        if self._distorted and self._map1 is None:
            h, w = frame.shape[:2]
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.camera.matrix(), self.camera.distortion, None,
                self.camera.matrix(), (w, h), cv2.CV_16SC2)
        undistorted = (cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
                       if self._distorted else frame)
        self.last_view = undistorted
        mask = self.mask(undistorted)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []

        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.args.min_area or len(contour) < 5:
                continue
            ellipse = cv2.fitEllipse(contour)
            (x, y), (a, b), angle = ellipse
            major, minor = max(a, b), min(a, b)
            axis_ratio = minor / major if major > 0 else 0.0
            if major <= 0 or minor <= 0 or axis_ratio < self.args.min_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (4.0 * math.pi * area / (perimeter * perimeter)
                           if perimeter > 0 else 0.0)
            if circularity < self.args.min_circularity:
                continue
            ellipse_area = math.pi * major * minor / 4.0
            area_error = abs(area - ellipse_area) / max(ellipse_area, 1.0)
            if area_error > self.args.max_area_error:
                continue

            # Reject badly defocused detections. Thresholding can still produce a
            # blob when the image is soft, so measure edge sharpness locally.
            x0, y0 = max(0, int(x - major / 2)), max(0, int(y - minor / 2))
            x1, y1 = min(gray.shape[1], int(x + major / 2)), min(gray.shape[0], int(y + minor / 2))
            patch = gray[y0:y1, x0:x1]
            sharpness = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size else 0.0
            if sharpness < self.args.min_sharpness:
                continue

            # Full major-axis diameter gives z ~= f*D / diameter_px.
            tilt = math.degrees(math.acos(np.clip(minor / major, 0.0, 1.0)))
            z = self.camera.f_mean * self.args.diameter / major
            xn = (x - self.camera.cx) / self.camera.fx
            yn = (y - self.camera.cy) / self.camera.fy
            ray_range = z * math.sqrt(1.0 + xn * xn + yn * yn)
            # Confidence rewards circularity, centrality, and sharpness.
            circularity_score = max(0.0, 1.0 - area_error / self.args.max_area_error)
            sharp_score = min(1.0, sharpness / max(self.args.reference_sharpness, 1.0))
            confidence = 100.0 * (0.60 * circularity_score + 0.25 * sharp_score + 0.15 * (minor / major))
            candidate = Detection((x, y), major, minor, tilt, z, ray_range, confidence, ellipse)
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        if not candidates:
            self.lock_misses += 1
            if self.lock_misses > self.args.lock_timeout:
                self.lock_center = None
            return []
        self.lock_misses = 0
        if self.lock_center is not None:
            scale = float(max(undistorted.shape[:2]))
            def lock_score(item: Detection) -> float:
                distance = np.linalg.norm(np.asarray(item.center) - self.lock_center) / scale
                return item.confidence - self.args.lock_distance_penalty * distance
            best_index = max(range(len(candidates)), key=lambda i: lock_score(candidates[i]))
            best = candidates.pop(best_index)
            candidates.insert(0, best)
        else:
            best = candidates[0]
        self.lock_center = np.asarray(best.center, dtype=np.float64)
        self.history.append((best.z_m, best.range_m))
        med_z, med_range = np.median(np.asarray(self.history), axis=0)
        best.z_m, best.range_m = float(med_z), float(med_range)
        return candidates[:self.args.max_detections]

    def find(self, frame: np.ndarray) -> Optional[Detection]:
        """Compatibility helper: return the highest-confidence disc only."""
        detections = self.find_all(frame)
        return detections[0] if detections else None


class RpiMjpegSource:
    """Read JPEG frames from rpicam-vid without a heavyweight GUI stack."""

    def __init__(self, index: int, width: int, height: int, autofocus: bool,
                 fps: int = 15, quality: int = 65, buffer_count: int = 3):
        command = ["rpicam-vid", "--camera", str(index), "--timeout", "0", "--nopreview",
                   "--codec", "mjpeg", "--quality", str(quality), "--width", str(width),
                   "--height", str(height), "--framerate", str(fps),
                   "--buffer-count", str(buffer_count), "-o", "-"]
        if autofocus:
            command += ["--autofocus-mode", "continuous"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0)
        self.buffer = bytearray()

    def isOpened(self) -> bool:
        return self.process.poll() is None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.process.stdout is None:
            return False, None
        while True:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpeg = bytes(self.buffer[start:end + 2])
                del self.buffer[:end + 2]
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return image is not None, image
            if len(self.buffer) > 2_000_000:
                del self.buffer[:-65536]

    def release(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


def open_camera(index: int, width: int, height: int, autofocus: bool = False,
                source: str = "auto", fps: int = 15, quality: int = 65,
                buffer_count: int = 3):
    if source in ("auto", "rpicam") and shutil.which("rpicam-vid"):
        camera = RpiMjpegSource(index, width, height, autofocus, fps, quality, buffer_count)
        if camera.isOpened():
            return camera
        camera.release()
        if source == "rpicam":
            raise RuntimeError("rpicam-vid could not start the Pi camera")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Autofocus hunting changes edge sharpness and destabilizes measurements.
    # This is best-effort; unsupported UVC controls are harmless.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    return cap


def draw(frame: np.ndarray, detections: list[Detection], fps: float) -> np.ndarray:
    if detections:
        for i, hit in enumerate(detections):
            colour = (0, 255, 0) if i == 0 else (255, 180, 0)
            cv2.ellipse(frame, hit.ellipse, colour, 2)
            x, y = map(int, hit.center)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"#{i} {hit.range_m * 1000:.0f}mm {hit.tilt_deg:.0f}deg",
                        (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        hit = detections[0]
        cv2.putText(frame, f"{len(detections)} discs | target {hit.range_m * 1000:.0f}mm",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
    else:
        cv2.putText(frame, "no reliable disc", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"{fps:.1f} fps   ESC quits", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)
    return frame


def run(args: argparse.Namespace, camera: CameraModel) -> None:
    cap = open_camera(args.camera, args.width, args.height, args.autofocus, args.source,
                      args.fps, args.camera_quality, args.buffer_count)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check /dev/video0 and permissions.")
    detector = DiscDetector(args, camera)
    sender = RobotJsonSender(args.udp_host, args.udp_port, camera, args)
    photo_trigger = PhotoTrigger(args.capture_dir, args.photo_threshold, args.jpeg_quality)
    tuner = OnlineSelfTuner(args, args.adaptive_profile) if args.self_tune else None
    viewpoint = ViewpointServo(args)
    last = time.monotonic()
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            frame_number += 1
            if frame_number % args.process_every:
                continue
            detections = detector.find_all(frame)
            if detections:
                if tuner is not None: tuner.observe(frame, detections[0])
                viewpoint.update(detections[0].center[1], frame.shape[0], detections[0].confidence)
            photo_path = photo_trigger.maybe_capture(frame, detections)
            sender.send(detections, frame.shape[1], frame.shape[0], photo_path)
            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            shown = draw(detector.last_view if detector.last_view is not None else frame, detections, fps)
            cv2.imshow("disc distance", shown)
            if args.show_mask:
                cv2.imshow("disc mask", detector.mask(frame))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        sender.close()
        viewpoint.close()
        if tuner is not None: tuner.save()
        cv2.destroyAllWindows()


def calibrate(args: argparse.Namespace) -> None:
    if args.distance is None:
        raise SystemExit("calibrate requires --distance, e.g. calibrate --distance 1.0")
    cap = open_camera(args.camera, args.width, args.height, args.autofocus, args.source,
                      args.fps, args.camera_quality, args.buffer_count)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    # Use a neutral model first; calibration is only measuring pixels-per-metre.
    approx = CameraModel(640, 640, args.width / 2, args.height / 2, np.zeros(5))
    detector = DiscDetector(args, approx)
    samples = []
    print("Hold the disc face-on, fill a useful part of the frame, and press ESC when done.")
    try:
        for _ in range(args.samples):
            ok, frame = cap.read()
            if not ok:
                continue
            hit = detector.find(frame)
            if hit:
                samples.append(hit.major_px)
            cv2.imshow("calibration (ESC quits)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if len(samples) < 5:
        raise RuntimeError("Too few detections; adjust --black/--contrast or lighting")
    focal = float(np.median(samples) * args.distance / args.diameter)
    model = CameraModel(focal, focal, args.width / 2, args.height / 2, np.zeros(5))
    model.save(args.camera_matrix)
    print(f"Saved {args.camera_matrix}; estimated focal length = {focal:.1f} px from {len(samples)} samples")


def focus_helper(args: argparse.Namespace) -> None:
    """Show a live focus score; adjust the lens until the score peaks."""
    cap = open_camera(args.camera, args.width, args.height, args.autofocus, args.source,
                      args.fps, args.camera_quality, args.buffer_count)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera for focus helper")
    best = 0.0
    print("Adjust the lens manually until the Focus score peaks. Press ESC to exit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed during focus helper")
            h, w = frame.shape[:2]
            # Use the central 60%: it avoids measuring empty borders and matches
            # the area where a target is most likely to be placed.
            x0, x1 = int(w * 0.20), int(w * 0.80)
            y0, y1 = int(h * 0.20), int(h * 0.80)
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            best = max(best, score)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 255), 2)
            cv2.putText(frame, f"Focus: {score:.1f}  best: {best:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)
            crop = cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("focus helper - live", frame)
            cv2.imshow("focus helper - center crop", crop)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Best focus score observed: {best:.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=["run", "calibrate", "focus"], default="run")
    p.add_argument("--diameter", type=float, default=0.273, help="disc diameter in metres")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera-matrix", default="camera.json")
    p.add_argument("--focal-length", type=float, default=320.0,
                   help="fallback focal length in pixels when camera matrix is absent")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15, help="camera frame rate; 15 is adequate for robot approach")
    p.add_argument("--camera-quality", type=int, default=65, help="rpicam MJPEG quality; lower uses less CPU/bandwidth")
    p.add_argument("--buffer-count", type=int, default=3, help="rpicam in-flight buffers; lower uses less RAM")
    p.add_argument("--process-every", type=int, default=2, help="analyze every Nth frame; 2 lowers CPU while keeping <150 ms heartbeat")
    p.add_argument("--distance", type=float, help="known face-on distance for calibration")
    p.add_argument("--samples", type=int, default=90)
    p.add_argument("--black", type=int, default=70, help="maximum grayscale value for dark disc")
    p.add_argument("--contrast", type=int, default=15, help="minimum local contrast")
    p.add_argument("--min-area", type=float, default=400)
    p.add_argument("--min-ratio", type=float, default=0.35, help="minimum ellipse axis ratio; rejects highly edge-on/line-like shapes")
    p.add_argument("--min-circularity", type=float, default=0.45, help="minimum contour circularity; rejects line-like blobs")
    p.add_argument("--max-area-error", type=float, default=0.35)
    p.add_argument("--min-sharpness", type=float, default=8.0)
    p.add_argument("--reference-sharpness", type=float, default=80.0)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--show-mask", action="store_true")
    p.add_argument("--autofocus", action="store_true", help="enable autofocus; off by default for stable measurements")
    p.add_argument("--source", choices=["auto", "rpicam", "v4l2"], default="auto", help="camera backend; auto prefers rpicam on Raspberry Pi")
    p.add_argument("--max-detections", type=int, default=32, help="maximum discs reported per frame")
    p.add_argument("--udp-host", default="127.0.0.1", help="robot JSON UDP destination")
    p.add_argument("--udp-port", type=int, default=5005, help="robot JSON UDP port")
    p.add_argument("--stop-distance", type=float, default=0.45, help="stop command distance in metres")
    p.add_argument("--approach-distance", type=float, default=2.0, help="distance span mapped to forward=0..1")
    p.add_argument("--turn-gain", type=float, default=1.0, help="multiplier for normalized left/right turn")
    p.add_argument("--min-command-confidence", type=float, default=45.0, help="minimum confidence allowed to move")
    p.add_argument("--lock-timeout", type=int, default=8, help="frames without a target before releasing target lock")
    p.add_argument("--lock-distance-penalty", type=float, default=35.0, help="target-lock position penalty")
    p.add_argument("--photo-threshold", type=int, default=3, help="capture when at least this many discs are visible")
    p.add_argument("--capture-dir", default="/home/tearx/disc_photos", help="directory for triggered JPEG photos")
    p.add_argument("--jpeg-quality", type=int, default=90, help="photo JPEG quality")
    p.add_argument("--self-tune", dest="self_tune", action="store_true", default=True)
    p.add_argument("--no-self-tune", dest="self_tune", action="store_false")
    p.add_argument("--adaptive-profile", default="adaptive_profile.json")
    p.add_argument("--servo-pin", type=int, default=None, help="BCM GPIO pin for optional vertical viewpoint servo")
    p.add_argument("--servo-center", type=float, default=0.0)
    p.add_argument("--servo-min-angle", type=float, default=-60.0)
    p.add_argument("--servo-max-angle", type=float, default=60.0)
    p.add_argument("--servo-min-pulse", type=float, default=0.0005)
    p.add_argument("--servo-max-pulse", type=float, default=0.0025)
    p.add_argument("--servo-y-gain", type=float, default=18.0)
    p.add_argument("--servo-y-deadband", type=float, default=0.06)
    p.add_argument("--servo-update-hz", type=float, default=8.0)
    p.add_argument("--servo-y-direction", type=int, choices=(-1,1), default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.process_every = max(1, args.process_every)
    if args.mode == "calibrate":
        calibrate(args)
        return
    if args.mode == "focus":
        focus_helper(args)
        return
    if Path(args.camera_matrix).exists():
        camera = CameraModel.from_json(args.camera_matrix)
    else:
        # Fallback is usable but less accurate. Run calibrate for real values.
        camera = CameraModel(args.focal_length, args.focal_length,
                             args.width / 2, args.height / 2, np.zeros(5))
        print(f"Warning: {args.camera_matrix} not found; using approximate intrinsics")
    run(args, camera)


if __name__ == "__main__":
    main()
