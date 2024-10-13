import os
import shutil

import gdown

URL_BEST_MODEL = "https://drive.google.com/file/d/1l0hXnnkZHiUM8TDuvCwlSIwdtURxY9ug/view?usp=sharing"

def download():
    gdown.download(URL_BEST_MODEL)

    os.makedirs("src/model_weights", exist_ok=True)
    shutil.move("model_best.pth", "src/model_weights/model_best.pth")


if __name__ == "__main__":
    download()
