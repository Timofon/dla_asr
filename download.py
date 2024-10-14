import os
import shutil

import gdown

URL_BEST_MODEL = "https://drive.google.com/file/d/1P_Hi8ZbASMmDJtE8o55OO6p9-XdE4B3f/view?usp=sharing"

def download():
    gdown.download(URL_BEST_MODEL)

    os.makedirs("src/model_weights", exist_ok=True)
    shutil.move("model_best.pth", "src/model_weights/model_best.pth")


if __name__ == "__main__":
    download()
