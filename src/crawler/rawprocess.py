import argparse
import os
import shutil
from pathlib import Path

import natsort
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from dcardpost import DcardPost


def rename_crawl_failure_file():
    files = os.listdir('dataset/raw/')
    for fn in files:
        file_path = os.getcwd() + '/dataset/raw/' + fn
        with open(file_path, encoding="utf-8") as f:
            txt = f.read()
            if "需要確認您的連線是安全的" in txt:
                print("get this one is wrong!!!!: ", file_path)
                new_fn = fn.split("_")[0]+"_.html"
                new_file_path = os.getcwd() + '/dataset/raw/' + new_fn
                os.rename(file_path, new_file_path)


def separate_drity_raws():
    """
    Check the files in the raw directory to determine which dataset each file belongs to, and copy it to a separate
    directory. Note that this function does not handle cases where a file has the same title and appears in multiple
    datasets.

    Returns:
        None
    """
    private_df = pd.read_csv(
        "dataset/intern_homework_private_test_dataset.csv")
    public_df = pd.read_csv("dataset/intern_homework_public_test_dataset.csv")
    train_df = pd.read_csv("dataset/intern_homework_train_dataset.csv")
    DFs = [private_df, public_df, train_df]
    files = os.listdir('dataset/raw/')
    for fn in files:
        print("fn = ", fn)
        file_path = os.getcwd() + '/dataset/raw/' + fn
        title = ""
        with open(file_path, encoding="utf-8") as f:
            html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            try:
                title = soup.select("h1")[0].text
            except:
                title = html
        print("title = ", title)
        target_index = ""
        for i in range(len(DFs)):
            row, _ = np.where(DFs[i] == title)
            print(row.size)
            if row.size > 0:
                target_index = i
                break

        print("target_index = ", target_index)
        if target_index == "":
            continue
        target_paths_dict = {0: "/dataset/private_df/",
                             1: "/dataset/public_df/", 2: "/dataset/train_df/"}
        print(target_paths_dict[target_index])
        shutil.copyfile(file_path, os.getcwd() +
                        target_paths_dict[target_index]+fn)
        pass
    pass


def main(args):
    dataset_path = Path(args.path)
    raw_dirs = dataset_path.glob('*raw')
    for raw_dir in raw_dirs:
        empty_post = DcardPost()
        df = pd.DataFrame(empty_post.__dict__, index=[0])
        htmls = raw_dir.glob('*.html')
        htmls = natsort.natsorted(htmls)
        for html in htmls:
            with open(html) as f:
                contents = f.read()
                soup = BeautifulSoup(contents, 'lxml')
                post = DcardPost(soup)
                p_row = pd.DataFrame(post.__dict__, index=[0])
                df = pd.concat([df, p_row], ignore_index=True)

        df = df.drop([0])
        print("saving crawler_{}_dataset.csv in {}".format(
            raw_dir.stem, raw_dir.parent))
        df.to_csv("{}/crawler_{}_dataset.csv".format(raw_dir.parent.resolve(),
                  raw_dir.stem), index=False)
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description='process the raw data from dcard crawler and save to csv')
    args.add_argument('-p', '--path', default="../../dataset/", type=str,
                      help="path to dataset (default: ../../dataset/)")
    args = args.parse_args()
    print("Opening raw directories in ", args.path)

    main(args)
