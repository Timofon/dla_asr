from editdistance import eval


def calc_cer(target_text: str, predicted_text: str) -> float:
    if not len(target_text):
        return 1
    return eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    if not len(target_text):
        return 1
    return eval(target_text.split(), predicted_text.split()) / len(target_text.split())
