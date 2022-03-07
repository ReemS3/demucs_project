from importlib.resources import contents
from moviepy.editor import *
import musdb
import tensorflow as tf
import tqdm as tqdm
import os

ORIGINAL_DATASET_ROOT = "./data/musdb18"
ORIGINAL_TRAIN_DATASET_ROOT = "./data/musdb18/train/"
ORIGINAL_TEST_DATASET_ROOT = "./data/musdb18/test/"

TRAIN_LABELS, TEST_LABELS = [], []


def get_labels(source_dir, training=False):
    if training:
        mu = musdb.DB(source_dir, subsets="train")
        for track in mu:
            TRAIN_LABELS.append(track.targets)
    else:
        mu = musdb.DB(source_dir, subsets="test")
        for track in mu:
            TEST_LABELS.append(track.targets)


def clip_and_convert_toWAV(source_dir, target_dir):
    for test_track in tqdm.tqdm(os.listdir(source_dir)):
        track_path = source_dir + test_track
        audio = VideoFileClip(track_path)

        if audio.duration > 30:
            new_audio = audio.set_duration(30)
        else:
            new_audio = audio
        new_audio_name = target_dir + test_track[:-3] + "wav"

        new_audio.audio.write_audiofile(new_audio_name)


def make_new_folder(folder_name):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print("Folder already exists.")


def decode(folder_name):
    decoded_data = []
    for track in os.listdir(folder_name):
        raw_audio = tf.io.read_file(folder_name + track)
        decoded_track, _ = tf.audio.decode_wav(contents=raw_audio, desired_channels=2)
        decoded_data.append(decoded_track)
    return decoded_data


def main():
    parent_path = "./data/musdb18_clipped"
    train_folder_path = parent_path + "/train/"
    test_folder_path = parent_path + "/test/"

    make_new_folder(parent_path)
    make_new_folder(train_folder_path)
    make_new_folder(test_folder_path)

    clip_and_convert_toWAV(ORIGINAL_TRAIN_DATASET_ROOT, train_folder_path)
    clip_and_convert_toWAV(ORIGINAL_TEST_DATASET_ROOT, test_folder_path)

    get_labels(ORIGINAL_DATASET_ROOT, training=True)
    get_labels(ORIGINAL_DATASET_ROOT, training=False)

    tf.compat.v1.enable_eager_execution()
    train_ds = decode(train_folder_path)
    test_ds = decode(test_folder_path)

    tf.data.experimental.save(train_ds, "./")


if __name__ == "__main__":
    main()
