import torch
from torch.nn.utils.rnn import pad_sequence

# def collate_fn(dataset_items):
#     """
#     Collate and pad fields in dataset items
#     """
#     batch = {}
#     batch["text"] = [item["text"] for item in dataset_items]
#     batch["audio_path"] = [item["audio_path"] for item in dataset_items]
#     batch["spectrogram"] = pad_sequence(
#         [item["spectrogram"].squeeze(0).permute(1, 0) for item in dataset_items],
#         batch_first=True,
#     )  # [batch_size, freq, max_time]

#     batch["spectrogram_length"] = torch.tensor(
#         [item["spectrogram"].shape[2] for item in dataset_items]
#     )

#     batch["text_encoded"] = pad_sequence(
#         [item["text_encoded"].squeeze(0) for item in dataset_items], batch_first=True
#     )

#     batch["text_encoded_length"] = torch.tensor(
#         [item["text_encoded"].shape[1] for item in dataset_items]
#     )

#     batch["audio"] = pad_sequence(
#         [item["audio"].squeeze(0) for item in dataset_items], batch_first=True
#     )
#     return batch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version of the tensors.
    """
    dataset_padded = dict()

    for key in dataset_items[0].keys():
        if key in ["text", "audio_path"]:
            dataset_padded[key] = [item[key] for item in dataset_items]
        elif key == "spectrogram":
            freq_length = dataset_items[0]["spectrogram"].shape[1]
            max_time_length = max(
                item["spectrogram"].shape[2] for item in dataset_items
            )
            dataset_padded["spectrogram"] = torch.zeros(
                (len(dataset_items), freq_length, max_time_length)
            )
            for i, item in enumerate(dataset_items):
                current_length = item["spectrogram"].shape[2]
                dataset_padded["spectrogram"][i, :, :current_length] = item[
                    "spectrogram"
                ][0]
            dataset_padded["spectrogram_length"] = torch.tensor(
                [item["spectrogram"].shape[2] for item in dataset_items],
                dtype=torch.int32,
            )
        elif key == "audio":
            dataset_padded[key] = [item[key] for item in dataset_items]
        else:
            sequences = [item[key][0] for item in dataset_items]
            dataset_padded[key] = pad_sequence(sequences, batch_first=True)
            if key == "text_encoded":
                dataset_padded["text_encoded_length"] = torch.tensor(
                    [len(seq) for seq in sequences], dtype=torch.int32
                )

    return dataset_padded


# def collate_fn(dataset_items: list[dict]):
#     """
#     Collate and pad fields in the dataset items.
#     Converts individual items into a batch.

#     Args:
#         dataset_items (list[dict]): list of objects from
#             dataset.__getitem__.
#     Returns:
#         result_batch (dict[Tensor]): dict, containing batch-version
#             of the tensors.
#     """
#     dataset_padded = dict()

#     for key in dataset_items[0].keys():
#         if key == "text" or key == "audio_path":
#             dataset_padded[key] = [
#                 dataset_items[i][key] for i in range(len(dataset_items))
#             ]
#         elif key == "spectrogram":
#             dataset_padded["spectrogram"] = torch.zeros(
#                 (len(dataset_items)),
#                 dataset_items[0]["spectrogram"].shape[1],
#                 max(item["spectrogram"].shape[2] for item in dataset_items),
#             )
#             for i in range(len(dataset_items)):
#                 current_width = dataset_items[i]["spectrogram"].shape[2]
#                 dataset_padded["spectrogram"][i, ..., :current_width] = dataset_items[
#                     i
#                 ]["spectrogram"]
#             dataset_padded["spectrogram"] = dataset_padded["spectrogram"]
#             dataset_padded["spectrogram_length"] = torch.tensor(
#                 [dataset_items[i][key].shape[2] for i in range(len(dataset_items))]
#             )
#         else:
#             dataset_padded[key] = pad_sequence(
#                 [dataset_items[i][key].squeeze() for i in range(len(dataset_items))],
#                 batch_first=True,
#             )
#             if key == "text_encoded":
#                 dataset_padded["text_encoded_length"] = torch.tensor(
#                     [dataset_items[i][key].shape[1] for i in range(len(dataset_items))]
#                 )

#     return dataset_padded
