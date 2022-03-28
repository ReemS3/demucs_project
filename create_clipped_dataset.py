import audioop
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
        for subaudio in os.listdir(source_dir+track):
            sources_track = []
            for source in os.listdir(source_dir+track+'/'+subaudio):
                raw_audio = tf.io.read_file(source_dir+track+'/'+subaudio+'/'+source)
                decoded_source, _ = tf.audio.decode_wav(
                    contents=raw_audio, desired_samples=441000, desired_channels=2)

                if "mixture" in source:
                    mixture.append(decoded_source)
                else:
                    sources_track.append(decoded_source)
            sources_ls.append(sources_track)
    return mixture, sources_ls

def reduce_length(audio_ls, name, source):
    if len(audio_ls) == 1:
        path = name + "1"
        make_new_folder(path)
        new_audio = audio_ls[0]
        new_audio.export(path+'/'+source, format="wav")
    else:
        for idx, audio in enumerate(audio_ls):
            path = name + str(idx)
            make_new_folder(path)
            audio.export(path+'/'+source, format="wav")


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
                tmp = []
                for i in range(3):
                    tmp.append(audio[i*10000:(i+1)*10000])
                new_audio = tmp
            else:
                new_audio = []
                new_audio.append(audio)

            new_audio_name = target_dir + test_track + "/" 
            reduce_length(new_audio, new_audio_name, source)

def make_new_folder(folder_name):
    """Create a new folder if it does not exist"""
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print()

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
