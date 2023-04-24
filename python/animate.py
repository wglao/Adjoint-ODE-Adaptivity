import cv2
import os


def animate(case: str):
  plots = os.listdir(case)
  frame = cv2.imread(os.path.join(case, plots[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(case + '/' + case + '.mp4',
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 12,
                          (width, height))
  for i, p in enumerate(plots):
    p_path = os.path.join(case, p)
    video.write(cv2.imread(p_path))
    if i > 0 and i < len(plots) - 1:
      os.remove(p_path)

  cv2.destroyAllWindows()
  video.release()

if __name__ == "__main__":
  case = 'ResNet_no_matrix'
  animate(case)