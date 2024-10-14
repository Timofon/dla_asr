import argparse
import pathlib
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def calc_metrics(pred_dir: str, gt_dir: str):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    cer = 0
    wer = 0
    file_counter = 0

    for pred_file in pred_dir.iterdir():
        filename = pred_file.name
        target_text = None
        with open(gt_dir / filename, 'r') as f_gt:
            target_text = f_gt.read()
        target_text = CTCTextEncoder.normalize_text(target_text)

        predicted_text = None
        with open(pred_file, 'r') as f_pred:
            predicted_text = f_pred.read()

    cer += calc_cer(target_text, predicted_text)
    wer += calc_wer(target_text, predicted_text)
    file_counter += 1

    cer_avg = cer / file_counter
    wer_avg = wer / file_counter
    print(f"CER: {cer_avg:.4f}")
    print(f"WER: {wer_avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir")
    parser.add_argument("--gt_dir")
    args = parser.parse_args()
    calc_metrics(pred_dir=args.pred_dir, gt_dir=args.gt_dir)
