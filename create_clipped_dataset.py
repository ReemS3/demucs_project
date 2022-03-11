from moviepy.editor import *
import tensorflow as tf
import tqdm
import os

ORIGINAL_DATASET_ROOT = "./data/musdb18"
ORIGINAL_TRAIN_DATASET_ROOT = "./data/musdb18/train/"
ORIGINAL_TEST_DATASET_ROOT = "./data/musdb18/test/"


def decode_sources(source_dir):
    """Decodes sources the sources are in following order: bass, drums, other, vocals."""
    tracks_labels = []
    for track in tqdm.tqdm(os.listdir(source_dir)):
        sources = []
        if "DS_Store" not in track:
            for source in tqdm.tqdm(os.listdir(source_dir+track)):
                if "piano" not in source and "DS_Store" not in source:
                    raw_audio = tf.io.read_file(source_dir + track+'/'+source)
                    decoded_source, _ = tf.audio.decode_wav(
                        contents=raw_audio, desired_samples=10000, desired_channels=2)
                    sources.append(decoded_source)
            tracks_labels.append(sources)
    return tracks_labels


def clip_and_convert_toWAV(source_dir, target_dir):
    """Clip the audio track and create a new track containing only the first 30 seconds."""
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
    """Create a new folder if it does not exist"""
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print("Folder already exists.")


def decode(folder_name):
    """Decode the WAV files in the given folder."""
    decoded_data = []
    for track in tqdm.tqdm(os.listdir(folder_name)):
        if "DS_Store" not in track:
            raw_audio = tf.io.read_file(folder_name + track)
            decoded_track, _ = tf.audio.decode_wav(
                contents=raw_audio, desired_samples=10000, desired_channels=2)
            decoded_data.append(decoded_track)
    return decoded_data


def main():
    parent_path = "./data/musdb18_clipped"
    train_folder_path = parent_path + "/train/"
    test_folder_path = parent_path + "/test/"

    parent_labels_path = "./data/musdb18_clipped_separated"
    train_labels_folder_path = parent_labels_path + "/train/"
    test_labels_folder_path = parent_labels_path + "/test/"

    make_new_folder(parent_path)
    make_new_folder(train_folder_path)
    make_new_folder(test_folder_path)

    clip_and_convert_toWAV(ORIGINAL_TRAIN_DATASET_ROOT, train_folder_path)
    clip_and_convert_toWAV(ORIGINAL_TEST_DATASET_ROOT, test_folder_path)

    train_labels = decode_sources(train_labels_folder_path)
    test_labels = decode_sources(test_labels_folder_path)

    train_ds = decode(train_folder_path)
    test_ds = decode(test_folder_path)

    train_tfds = (tf.data.Dataset.from_tensor_slices((
        tf.cast(train_ds, tf.float32),
        tf.cast(train_labels, tf.float32))))

    test_tfds = (tf.data.Dataset.from_tensor_slices((
        tf.cast(test_ds, tf.float32),
        tf.cast(test_labels, tf.float32))))

    tf.data.experimental.save(train_tfds, "./dataset_train")
    tf.data.experimental.save(test_tfds, "./dataset_test")


if __name__ == "__main__":
    main()
