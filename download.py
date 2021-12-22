import os
import numpy as np
import pandas as pd
import requests
import csv

multi_data = pd.read_csv("./MuMu_dataset/MuMu_dataset_multi-label.csv", sep=",", index_col=0)
amazon_meta = pd.read_json("./MuMu_dataset/amazon_metadata_MuMu.json")

img_path = "./MuMu_dataset/album_images"
csv_path = "./MuMu_dataset/available_album_ids.csv"

headers = ['amazon_id', 'img_path']

total_data_num = len(amazon_meta)
no_match_num = 0
req_error_num = 0
no_image_num = 0

print("Start download #%d of album images" %(total_data_num))
if not os.path.exists(img_path):
    os.makedirs(img_path)

with open(csv_path, 'w', encoding='UTF8') as c:
    writer = csv.writer(c)
    writer.writerow(headers)

    for idx in range(total_data_num):
        # iteration checking
        if (idx+1) % 3000 == 0:
            break
            print("Downloaing...", int(idx/total_data_num*100), "%")
            
        album = amazon_meta.iloc[idx]
        amazon_id = album['amazon_id']
        img_url = album['imUrl']

        r = requests.get(img_url)
        if r.status_code != 200:
            req_error_num = 0
            continue

        # Check valid match
        try:
            _ = multi_data.loc[amazon_id]
        except:
            no_match_num += 1
            print("No Matched Tracks Error")
            continue

        # check no-img-lg
        _, f_name = os.path.split(img_url)
        if "no-img-lg" in f_name or ".gif" in f_name:
            no_image_num += 1
            continue

        # save image
        f_path = os.path.join(img_path, amazon_id + ".jpg")
        with open(f_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
        f.close()

        # Write csv
        row = [amazon_id, f_path]
        writer.writerow(row)  

    c.close()  
        
print("Completed.")
print("From total #%d of data,\nNo matching error: %d\nRequest error: %d\nNo image error: %d" %(total_data_num, no_match_num, req_error_num, no_image_num))
