import cv2
import os


def animate(case: str = 'test', framerate: int = 12):
  plots = os.listdir(case)
  plots.sort(key=lambda x: os.path.getmtime(os.path.join(case,x)))
  frame = cv2.imread(os.path.join(case, plots[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(case + '/' + case + '.mp4',
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), framerate,
                          (width, height))
  for i, p in enumerate(plots):
    p_path = os.path.join(case, p)
    video.write(cv2.imread(p_path))
    if i > 0 and i < len(plots) - 1:
      os.remove(p_path)

  cv2.destroyAllWindows()
  video.release()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", default='test', type=str)
  parser.add_argument("--fr", default=12, type=int)
  args = parser.parse_args()
  case = args.case
  framerate = args.fr
  animate(case, framerate)