from google_drive_downloader import GoogleDriveDownloader
import inspect
import os

FILE_LOOKUP = {
    # These are the training dataset links
    "BASE_train.npz":               "1cds1N9o_qJorc9zTkWuNLA3szTMjMsoh",
    "ACOUS_train.npz":              "1HFdubUueUeDN4cP2Ea9FG8Ab1TgTbe5h",
    "BG_0-100_train.npz":           "137Fzbe930XBA_UbR1ciKduyhQ5pRMl0Q",
    "BG_60-80_train.npz":           "1-pcK82wLNoj2auVyMtwW6aAI-pa0aBAG",
    "BG_H2O_train.npz":             "1Ig7ox29hw79hqPYYwsdSdUJvbZJsy6BC",
    "HET_0-100_train.npz":          "154ZedQeieTcLRsg0hJpYWNibpQUZr8o5",
    "HET_60-80_train.npz":          "1i-zKtQ7fmTB4-HOvTkrK1PnJaK01_9KD",
    "ILLUM_5mm_train.npz":          "1u6voToyq2Ax_3224jJgRa9BJMnbHg1mL",
    "ILLUM_POINT_train.npz":        "1JtW1zKldTP0taj30WlKeIqn7eI4dNJik",
    "INVIS_ACOUS_train.npz":        "1sJhNneM9JAp1NwDdQwEhsUGLVQL7a_4n",
    "INVIS_SKIN_ACOUS_train.npz":   "129nYSBg6RXQMZ8J1LxzJxO32AqML0a0q",
    "INVIS_SKIN_train.npz":         "10dbsoCLOYDXDpVuczz93g5EUzctXZ7Fj",
    "INVIS_train.npz":              "1FhcLxG9EHUmV7bgKmgiWzCn9_m43ddNr",
    "MSOT_ACOUS_SKIN_train.npz":    "1LUVy6eOfXYQ28XbwLyCPfptK_1ickTJW",
    "MSOT_ACOUS_train.npz":         "1IgZFLOSZA4QqKFh8mXMa201nCS-_0IWj",
    "MSOT_SKIN_train.npz":          "1-9BT5Sp7mX8EUdTU09Er5LNTSzzA_rQ1",
    "MSOT_train.npz":               "1qlBEvPsdZ6LHyTlLi8d0KCH-Y83nYA2X",
    "RES_0.6_train.npz":            "1FXoxZu6o0NXeCF_hDiBVZN8xzhq5d6nB",
    "RES_0.15_SMALL_train.npz":     "1na0Y8Zc_eiVg87A-3KdeTNft4jQ4J8rz",
    "RES_0.15_train.npz":           "1nnHiu02nuSykX9UL_ZEDd6Id6H572Fbb",
    "RES_1.2_train.npz":            "1yuyjIpaZi-jxUeXtWZv92ZCOpWqfSs1M",
    "SKIN_train.npz":               "1ns67HEjcvFbo1z43orOm2wVtX0bNfOzZ",
    "SMALL_train.npz":              "1XMfMIS8wQoMV33YNcsbhrnJ5Tdczpvv5",
    "WATER_2cm_train.npz":          "1Q_qwcrYOoaKzONIqrVe4EvWzof6yuCiI",
    "WATER_4cm_train.npz":          "1B_GRpiLrSA0S-BstQQSNWaKNdO0q74EH",

    # These are the trained network drive links
    "ACOUS_LSTM_5.h5":              "1sFcAdNCpW1Osaz6mZhzaMXmG9Z1qMoDN",
    "ACOUS_LSTM_10.h5":             "1qoy3Y5wgUNhd9pxjPpoFdybz4xiKuCUP",
    "ACOUS_LSTM_41.h5":             "1dvVRa2I0-kgBKd9gPpevNCwDxV0J9ys1",
    "BASE_LSTM_5.h5":               "1Zb47HNV3gj_JjEo1rvzB-uWDvlHf6wg_",
    "BASE_LSTM_10.h5":              "1gny69sKmE_Ojlk0OOStS1_RXg8VjBOm2",
    "BASE_LSTM_41.h5":              "1HXH95ptrnZla6FXdk37t7I6fzlA3Recg",
    "BG_0-100_LSTM_5.h5":           "1nfhi6G8KZNjn5DypVoGk6eBb-yI3R6yz",
    "BG_0-100_LSTM_10.h5":          "1_alJ6qsfSI_KYuj0iKlLXvdfXqJJ5uFd",
    "BG_0-100_LSTM_41.h5":          "1FYUXNt6xnkWGOmRjP4MI9OfVVHm5JOWR",
    "BG_60-80_LSTM_5.h5":           "11rPeBZYr3W0FC4dXonritriJm_MnstVr",
    "BG_60-80_LSTM_10.h5":          "1ZqIHj6y4JS80oA6CfhyBCZCdmiHVt4kx",
    "BG_60-80_LSTM_41.h5":          "1igidCXvWVGYI763M0BTwflUF-TNzTdL5",
    "BG_H2O_LSTM_5.h5":             "1yvayfIzPzkQ1-__frbr_YXylpICB4BIA",
    "BG_H2O_LSTM_10.h5":            "1CM2CGzXTSfF-L0XBR2Lv4g1QXxpIxSC5",
    "BG_H2O_LSTM_41.h5":            "1dTWjN22yXSjFUD8EFNwKVDhx4vrsuTIx",
    "HET_0-100_LSTM_5.h5":          "10OZITcyl38BIk_Iuzl4Jm-7wt9nQztSD",
    "HET_0-100_LSTM_10.h5":         "1ZPQ3eRbGwGLqGW5o4hN5I37R8oS8Memp",
    "HET_0-100_LSTM_41.h5":         "1UI5vMdjZOu1HSUmuSxlK40HAwy1xgCiw",
    "HET_60-80_LSTM_5.h5":          "1cmOOf2JILJc80OFmDzdolfUaABMPSWZS",
    "HET_60-80_LSTM_10.h5":         "1BYOQFiA4bCO9wtZWMPwngeC72K0vevRJ",
    "HET_60-80_LSTM_41.h5":         "1ySLGsMD4sFJgm1WzuWL86T19JHFF6Zv8",
    "ILLUM_5mm_LSTM_5.h5":          "1gOQhKRR0NlpS4CVmG4sDtlhrDFI7GuQm",
    "ILLUM_5mm_LSTM_10.h5":         "1CfezpU-ybDX8Ufm33zQFGY9Fjgy0Q1dR",
    "ILLUM_5mm_LSTM_41.h5":         "1URx4nqHZcmXUUlSScXUP02rd2NEMHqCF",
    "ILLUM_POINT_LSTM_5.h5":        "1zvWfSNfN99y7MuOxAfpmPQGFJ5pt-0lc",
    "ILLUM_POINT_LSTM_10.h5":       "1lmPjhUDNYniwBSz26dTfZHtOw502o7hf",
    "ILLUM_POINT_LSTM_41.h5":       "1YRM_6N7B_qhTBTP_ybpP4tH-wxSkxGVE",
    "INVIS_ACOUS_LSTM_5.h5":        "1vwO3mVRrpnI8wnzNMXQ_BDAFtoF38yLd",
    "INVIS_ACOUS_LSTM_10.h5":       "1Eh-OF1RJOUUCiZC_5XDBtllDvkxC38IK",
    "INVIS_ACOUS_LSTM_41.h5":       "1s7wGQYdYp4kqF2PEP682wcC5sCbjWP01",
    "INVIS_SKIN_ACOUS_LSTM_5.h5":   "1nw6fuRujJrX8GUEF3bh9JLOj9k6ErVwc",
    "INVIS_SKIN_ACOUS_LSTM_10.h5":  "1R230HAANxfHVtnZRGH1IYvGPIIPcW6CQ",
    "INVIS_SKIN_ACOUS_LSTM_41.h5":  "1E3vQKiS1mqo2FZ7HWnm2KvrGFdk0fLli",
    "INVIS_SKIN_LSTM_5.h5":         "1uIteDExkBHJ3cqBM4qY95LON9bKEVQpQ",
    "INVIS_SKIN_LSTM_10.h5":        "1fopNrsgi17uEAfEIuc30o8YVgMghGjdV",
    "INVIS_SKIN_LSTM_41.h5":        "1pO7N0gFNeuGChVZE0cfDge2jEG0dZOqB",
    "INVIS_LSTM_5.h5":              "1o5Np-dgNgJk74xLVukDhfx9_PCp54OsA",
    "INVIS_LSTM_10.h5":             "17qwo8Nnl7xQpBGtb_hhuPm1L5FvIkfLZ",
    "INVIS_LSTM_41.h5":             "1nz_5AnsyKAiWJY_AN11dN4hvp_JwF2ff",
    "MSOT_ACOUS_SKIN_LSTM_5.h5":    "1Azby2wiJbBTUbzYPgz0xzAOO3G7izpGa",
    "MSOT_ACOUS_SKIN_LSTM_10.h5":   "1TF-ZMcLR5kmbYykdTA3136GeY6e7poGD",
    "MSOT_ACOUS_SKIN_LSTM_41.h5":   "1NpWKp-ud3ouPNgk30lFpms-Z2X01c8UU",
    "MSOT_ACOUS_LSTM_5.h5":         "1BhFkhVAYE8LM_fUilJ47a-DAmjpiFHYS",
    "MSOT_ACOUS_LSTM_10.h5":        "1SyqnWPYEQoGTWFRm7UsMW8kQiS4xouMA",
    "MSOT_ACOUS_LSTM_41.h5":        "1Fz6hbSlmqm0cxUYEY_2N2oRp2B0sPwRu",
    "MSOT_SKIN_LSTM_5.h5":          "1FQrfiJyejS0XymuFZfWoBXyruD4ug6FC",
    "MSOT_SKIN_LSTM_10.h5":         "1CNdtWlDkh67JVGOZSESIeRbKNhm0zb-f",
    "MSOT_SKIN_LSTM_41.h5":         "1lrR8OP7Y_M75sEH6KRf9xEhWKoawZPwk",
    "MSOT_LSTM_5.h5":               "1EyK0ac91M3eVPfba7w-3UjeC9HSq6sY7",
    "MSOT_LSTM_10.h5":              "1K2hdTwV9AjtkwnX2MLuaJvJuBlspjnJM",
    "MSOT_LSTM_41.h5":              "1SyqDea4gJ4QtVjIzxKyTEGrzKh6KjsWp",
    "RES_0.6_LSTM_5.h5":            "1tFdht2UrAVDmtKLjDCPA3ocrXZGI39xY",
    "RES_0.6_LSTM_10.h5":           "1VXdLM04zO8ZYGkfYyQMgDscwbxRh2lKW",
    "RES_0.6_LSTM_41.h5":           "1cD9A1XN3Q_nP6kx9EJqjaTyhWH9_YVG0",
    "RES_0.15_SMALL_LSTM_5.h5":     "1KUkJCNqK4lX9D3GDyBqKhNSUlQY_QkEM",
    "RES_0.15_SMALL_LSTM_10.h5":    "1tbbgX47luJ4Po71MsOkDF2TUn2y1QI88",
    "RES_0.15_SMALL_LSTM_41.h5":    "1rCeO-ittvXCmeCNnveu3E25xPPa5ecPT",
    "RES_0.15_LSTM_5.h5":           "17tmHbsXZzLcsn3elQLf4TeFJJ4CfkhTw",
    "RES_0.15_LSTM_10.h5":          "11PEZ8E_6UiNhFgi73fV-bp2dCCYXMsC5",
    "RES_0.15_LSTM_41.h5":          "1NqNyogDEowOYayrkL9Ln0M4GduxH6hoC",
    "RES_1.2_LSTM_5.h5":            "1h_FaE7Hey-2m7tHfyS9nWUau74G8BzKV",
    "RES_1.2_LSTM_10.h5":           "1Bme-QikLtPliXxAR-Ewi9c9E_L8ZJ7T6",
    "RES_1.2_LSTM_41.h5":           "1vvLPItdZ_QLl_KL1_2aZLRADNTfAWCcz",
    "SKIN_LSTM_5.h5":               "1p7EFl_O2U4Ar4AnMrKaYC9PeVRbY4ykG",
    "SKIN_LSTM_10.h5":              "1wNjJLn0Krcu1w16N0-jvQ0PweRgCvIR2",
    "SKIN_LSTM_41.h5":              "1DCywUgPXLgu-8Las4TGRj1uN7y0enR5X",
    "SMALL_LSTM_5.h5":              "1MYCUKeKTsZBVLGf_dpoNW84N2k0lKfz4",
    "SMALL_LSTM_10.h5":             "1oN8iVQEiQZCea8Up_UD949zn1dWQx7g2",
    "SMALL_LSTM_41.h5":             "1CzSpZ2KrxdP3G8OAMYPn4uVKfFOKVORD",
    "WATER_2cm_LSTM_5.h5":          "15HlXo9rw0hHanbBoHD-scpKQhrn3-ANc",
    "WATER_2cm_LSTM_10.h5":         "1uBcA0JpM-dCg7y-v9P6YKNI30dLBma6T",
    "WATER_2cm_LSTM_41.h5":         "1jXKh9SlGTsHBuXR_xCEvCz33kylXBg6d",
    "WATER_4cm_LSTM_5.h5":          "11WbNFU1_a_nxJYvRQAqmSDxGKI9I3Tpu",
    "WATER_4cm_LSTM_10.h5":         "1s1-UGvKr9slKCq_4SJnCGxxmw0L3rM5y",
    "WATER_4cm_LSTM_41.h5":         "1U8MIfYHJpP_U0p-NC2KrFSCaa72JaBkk",
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
