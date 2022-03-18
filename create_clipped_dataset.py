from pydub import AudioSegment
import tensorflow as tf
import tqdm
import os

ORIGINAL_DATASET_ROOT = "./data/musdb18hq"
ORIGINAL_TRAIN_DATASET_ROOT = "./data/musdb18hq/train/"
ORIGINAL_TEST_DATASET_ROOT = "./data/musdb18hq/test/"


def decode(source_dir):
    """Decodes sources the sources are in following order: bass, drums, other, vocals."""
    mixture, sources_ls = [], []
    for track in tqdm.tqdm(os.listdir(source_dir)):
        sources_track = []
        for source in os.listdir(source_dir+track):
            raw_audio = tf.io.read_file(source_dir + track+'/'+source)
            decoded_source, _ = tf.audio.decode_wav(
                contents=raw_audio, desired_samples=1323000, desired_channels=2)
            if "mixture" in source:
                mixture.append(decoded_source)
            else:
                sources_track.append(decoded_source)
        sources_ls.append(sources_track)
    return mixture, sources_ls


def clip_and_convert_toWAV(source_dir, target_dir):
    """Clip the audio track and create a new track containing only the first 30 seconds."""
    new_length = 30000
    for test_track in tqdm.tqdm(os.listdir(source_dir)):
        track_path = source_dir + test_track
        os.mkdir(target_dir + "/" + test_track)
        print("Create a folder for this track: ", test_track)
        for source in os.listdir(track_path):
            source_path = track_path + "/" + source
            audio = AudioSegment.from_wav(source_path)
            if len(audio) > new_length:
                new_audio = audio[:30000]
            else:
                new_audio = audio
            new_audio_name = target_dir + test_track + "/" + source
            new_audio.export(new_audio_name, format="wav")


def make_new_folder(folder_name):
    """Create a new folder if it does not exist"""
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print("Folder already exists.")


def main():
    parent_path = "./data/musdb18_clipped"
    train_folder_path = parent_path + "/train/"
    test_folder_path = parent_path + "/test/"

    make_new_folder(parent_path)
    make_new_folder(train_folder_path)
    make_new_folder(test_folder_path)

    # if you already have clipped the tracks, comment out the two following lines
    clip_and_convert_toWAV(ORIGINAL_TRAIN_DATASET_ROOT, train_folder_path)
    clip_and_convert_toWAV(ORIGINAL_TEST_DATASET_ROOT, test_folder_path)

    train_mixture, train_sources = decode(
        train_folder_path)

    test_mixture, test_sources = decode(
        test_folder_path)

    train_tfds = (tf.data.Dataset.from_tensor_slices((
        tf.cast(train_mixture, tf.float32),
        tf.cast(train_sources, tf.float32))))

    test_tfds = (tf.data.Dataset.from_tensor_slices((
        tf.cast(test_mixture, tf.float32),
        tf.cast(test_sources, tf.float32))))

    tf.data.experimental.save(train_tfds, "./dataset_train")
    tf.data.experimental.save(test_tfds, "./dataset_test")


if __name__ == "__main__":
    main()
