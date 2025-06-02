import cv2, shutil, subprocess, sys


in_file  = r"data/Running trial Markerless 4_Miqus_3_26071.avi"
out_file = "clip.avi"

start_f = 17_850          # first frame to keep (0-based)
end_f   = 18_700          # last frame to keep, inclusive


# locate ffmpeg.exe ----------------------------------------------------------
ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
if not shutil.which(ffmpeg):                          # still not found?
    sys.exit("ffmpeg.exe not found – install FFmpeg or set ffmpeg variable.")

# frame → seconds ------------------------------------------------------------
cap = cv2.VideoCapture(in_file)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

start_sec   = start_f / fps
duration_sec = (end_f - start_f + 1) / fps

# loss-less cut --------------------------------------------------------------
subprocess.run([
    ffmpeg,
    "-hide_banner", "-loglevel", "error",
    "-ss",  f"{start_sec:.6f}",
    "-i",   in_file,
    "-t",   f"{duration_sec:.6f}",
    "-c",   "copy",
    out_file
], check=True)