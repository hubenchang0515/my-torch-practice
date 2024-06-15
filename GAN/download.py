#! /usr/bin/env python3

import os
from bs4 import BeautifulSoup
from urllib.request import urlopen

AZUR_LANE_WIKI_URL:str = r"https://wiki.biligame.com/blhx/%E8%88%B0%E8%88%B9%E5%9B%BE%E9%89%B4"
DOWNLOAD_DIR:str = "./data/azur_lane_avatar/images"

def download(filename:str, url:str):
    with open(filename, "wb") as fp, urlopen(url) as response:
        fp.write(response.read())

if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with urlopen(AZUR_LANE_WIKI_URL) as response:
        soup = BeautifulSoup(response, "html.parser")
        for img in soup.find_all("img"):
            if img.has_attr("src") and img.has_attr("alt") and img["alt"].endswith("头像.jpg"):
                print(f"downloading {img['alt']} ...")
                download(f"{DOWNLOAD_DIR}/{img['alt']}", img["src"])