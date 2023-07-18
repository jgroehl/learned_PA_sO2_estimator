from google_drive_downloader import GoogleDriveDownloader
import inspect
import os

FILE_LOOKUP = {
    "BASE_train.npz":  "1cds1N9o_qJorc9zTkWuNLA3szTMjMsoh",
    "BASE_LSTM_10.h5": "1gny69sKmE_Ojlk0OOStS1_RXg8VjBOm2",
    "BASE_LSTM_5.h5":  "1Zb47HNV3gj_JjEo1rvzB-uWDvlHf6wg_",
    "BASE_LSTM_41.h5": "1HXH95ptrnZla6FXdk37t7I6fzlA3Recg",
}


def download_file(file_id):

    base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    target_path = f"{base_script_path}/../../data/{file_id}"
    if not os.path.exists(target_path):
        GoogleDriveDownloader.download_file_from_google_drive(file_id=FILE_LOOKUP[file_id],
                                                              dest_path=target_path,
                                                              overwrite=False)
    if not os.path.exists(target_path):
        raise FileNotFoundError("File does not exist after attempting to download...")


if __name__ == "__main__":
    download_file("BASE_train.npz")
    download_file("BASE_LSTM_10.h5")
    download_file("BASE_LSTM_5.h5")
    download_file("BASE_LSTM_41.h5")
