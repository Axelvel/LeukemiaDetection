import os
import cv2
import random

def augment_data(path):
    # For each image in the folder apply either a rotation or a flip

    if "augmented_0.bmp" not in os.listdir(path):
        for i, file in enumerate(os.listdir(path)):
            image = cv2.imread(path + file)
            augment = random.randint(0, 5)
            match augment:
                case 0:
                    image = cv2.flip(image, 0)
                case 1:
                    image = cv2.flip(image, -1)
                case 2:
                    image = cv2.flip(image, 0)
                    image = cv2.flip(image, -1)
                case 3:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                case 4:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                case _:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(f"{path}augmented_{i}.bmp", image)
