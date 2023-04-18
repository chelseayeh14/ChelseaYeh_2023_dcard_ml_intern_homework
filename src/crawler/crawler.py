import argparse
import json
import os
import ssl
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytz
import requests
import undetected_chromedriver as webdriver
from autoclick import click_cloudflare
from bs4 import BeautifulSoup
from googledcard import search_with_google
from selenium.webdriver import DesiredCapabilities

ssl._create_default_https_context = ssl._create_unverified_context

HOST_URL = "https://www.dcard.tw"
SEARCH_URL = "https://www.dcard.tw/search/posts?field=title&query="
POST_API_URL = "https://www.dcard.tw/service/api/v2/posts/"
SEARCH_API_URL = "https://www.dcard.tw/service/api/v2/search/posts?limit=100&sort=relevance&field=title&highlight=true&query="
TAIPEI_TZ = pytz.timezone('Asia/Taipei')


def query_proxy_list():
    proxy_url = 'https://free-proxy-list.net/'
    response = requests.get(proxy_url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.select("table")[0]
    df = pd.read_html(str(table))[0]
    condition = df['Https'] == 'yes'
    df = df[condition]
    return df


def get_one_proxy(proxy_list):
    ip_df = proxy_list.iloc[0]
    proxy = "https://{}:{}".format(ip_df['IP Address'], ip_df['Port'])
    proxy_list = proxy_list.drop(proxy_list.index[0])
    print("Using proxy at ", proxy)
    return proxy_list, proxy


def get_post_id_by_search_api(post_title, post_created_at, driver, opt, dc):
    global SEARCH_API_URL, TAIPEI_TZ

    utc_time = datetime.strptime(post_created_at, '%Y-%m-%d %H:%M:%S %Z')
    taipei_time = utc_time.replace(tzinfo=pytz.utc).astimezone(TAIPEI_TZ)
    post_created_at = taipei_time.strftime(
        f'{taipei_time.year}年{taipei_time.month}月{taipei_time.day}日')

    target_url = SEARCH_API_URL + quote(post_title)

    if driver == "":
        driver = webdriver.Chrome(chrome_options=opt, desired_capabilities=dc)

    driver = get_with_autoclick(driver, target_url)
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')
    res_json = soup.select('pre')[0].text
    post_list = json.loads(res_json)

    p_id = ""
    for post in post_list:
        title = post['title'].replace("</em>", "").replace("<em>", "")
        utc_time = datetime.strptime(
            post['createdAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
        taipei_time = utc_time.astimezone(TAIPEI_TZ)
        taipei_time_str = taipei_time.strftime(
            f'{taipei_time.year}年{taipei_time.month}月{taipei_time.day}日')
        if post_created_at == taipei_time_str:
            print("find same time: {}".format(taipei_time_str))
            if post_title == title:
                print("find same title: {}".format(title))
                p_id = post['id']
                print(type(p_id))
                print("p_id = {}".format(p_id))
                break

    if p_id == "":
        print("fail to find ", post_title)
    return p_id


def get_posts_and_save_html(ds_csv_path, driver, opt, dc):
    global HOST_URL, SEARCH_URL, TAIPEI_TZ

    index, _ = check_progress(ds_csv_path)
    print("continue from {} rows".format(index))
    df = pd.read_csv(ds_csv_path)
    progress_df = df[index:]

    for post_title, post_created_at in zip(progress_df.title, progress_df.created_at):

        utc_time = datetime.strptime(post_created_at, '%Y-%m-%d %H:%M:%S %Z')
        taipei_time = utc_time.replace(tzinfo=pytz.utc).astimezone(
            TAIPEI_TZ)
        post_created_at = taipei_time.strftime(
            f'{taipei_time.year}年{taipei_time.month}月{taipei_time.day}日')
        target_url = SEARCH_URL + quote(post_title)

        if driver == "":
            driver = webdriver.Chrome(
                chrome_options=opt, desired_capabilities=dc)

        result, data_key, href = check_google_query(
            driver, post_title, post_created_at)
        if result:
            driver = get_with_autoclick(driver, HOST_URL + href)
            html = driver.page_source
            post_id = data_key.split('-')[-1]
            save_html(ds_csv_path, html, post_title, post_id)
            continue

        driver = get_with_autoclick(driver, target_url, 3)
        roll = 8
        for i in range(roll):
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            soup_posts = soup.select("time")
            result, data_key, href = check_dcard_query(
                soup_posts, post_title, post_created_at)
            if result:
                driver = get_with_autoclick(driver, HOST_URL + href)
                html = driver.page_source
                post_id = data_key.split('-')[-1]
                save_html(ds_csv_path, html, post_title, post_id)
                break
            elif i == roll-1:
                save_html(ds_csv_path, "", post_title, "")
                break
            else:
                driver.execute_script(
                    "window.scrollTo({}, {});".format(1080 * i, 1080 * (i + 1)))


def check_dcard_query(soup_posts, post_title, post_created_at):
    data_key = ""
    href = ""

    for post in soup_posts:
        post = post.find_parent('div', {'data-key': True})
        p_title = post.find('a').text
        p_time = post.find('time').text
        print("{} = {}, \t{} = {}".format(
            post_title, p_title, post_created_at, p_time))
        if post_created_at == p_time or p_time in post_created_at:
            print("find same time: {}".format(p_time))
            if post_title == p_title:
                print("find same title: {}".format(p_title))
                data_key = post['data-key']
                print("data_key = {}".format(data_key))
                href = post.find('a')['href']
                print("href = {}".format(href))
                return True, data_key, href

    if data_key == "":
        print("fail to find ", post_title)
        return False, data_key, href


def check_google_query(driver, post_title, post_created_at):
    '''
        return 
            result, Bool
            data_key, post-241719570
            href, /f/mood/p/241719570
    '''
    r, url = search_with_google(post_title)
    if r:
        driver = get_with_autoclick(driver, url)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        try:
            p_time = soup.find('time').text
        except:
            return False, "", ""
        p_time = p_time[:p_time.index("日")+1]
        if post_created_at == p_time or p_time in post_created_at:
            print("find same time by google query: {} == {}".format(
                p_time, post_created_at))
            return True, "post-"+url.split('/')[-1], url[url.index('f')-1:]
        else:
            print("find diff time by google query: {} == {}".format(
                p_time, post_created_at))
            return False, "", ""
    else:
        print("fail to find ", post_title)
        return False, "", ""


def get_post_json_by_api(post_id):
    global POST_API_URL
    if post_id == "":
        return ""

    post_api_url = POST_API_URL + post_id
    print(post_api_url)
    html = ""
    while True:
        chrome_options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(chrome_options=chrome_options)
        # driver.set_page_load_timeout(5)
        driver.get(post_api_url)
        html = driver.page_source
        if "Too many requests, try again later" in html:
            print("Too many requests, try again later")
            time.sleep(10)
            driver.close()
            continue
        elif "需要確認您的連線是安全的" in html:
            print("encounter 需要確認您的連線是安全的")
            time.sleep(10)
            click_cloudflare()
            time.sleep(10)
            html = driver.page_source
            break
        else:
            # driver.close()
            print("Success pass cloudflare")
            break

    soup = BeautifulSoup(html, 'html.parser')
    driver.close()
    res_json = soup.select('pre')[0].text
    return json.loads(res_json)


def get_with_autoclick(driver, target_url, sleep_one=2):
    while True:
        driver.get(target_url)
        time.sleep(sleep_one)
        html = driver.page_source
        if "Too many requests, try again later" in html:
            print("encounter Too many requests, try again later")
            # proxy_list, proxy_url = get_one_proxy(proxy_list)
            time.sleep(60 * 5)
            continue
        elif "需要確認您的連線是安全的" in html:
            print("encounter 需要確認您的連線是安全的")
            time.sleep(5)
            click_cloudflare()
            time.sleep(5)
            continue
        else:
            print("Success pass cloudflare")
            return driver


def save_json(j, post_title, post_id):
    # if j == "":
    #     path_name = DATASET + "raw/" + post_id + "_" + post_title + ".json"
    #     try:
    #         with open(path_name, 'w', encoding='UTF-8') as f:
    #             f.write(json.dumps({}))
    #     except Exception as e:
    #         print(e)
    #         print("saving {} as json failure because not find the post".format(post_title))
    #     return

    # path_name = DATASET + "raw/" + post_id + "_" + post_title + ".json"
    # try:
    #     with open(path_name, 'w', encoding='UTF-8') as f:
    #         f.write(json.dumps(j))
    # except Exception as e:
    #     print(e)
    #     print("saving {} as json failure".format(post_title))
    pass


def save_html(ds_csv_path, h, post_title, post_id):
    index, raw_dir = check_progress(ds_csv_path)
    num = str(index)
    fname = num + "_" + post_id + ".html"
    broken_fname = num + "_.html"
    print("saving ", fname)
    with open(raw_dir.joinpath(fname), 'w', encoding='UTF-8') as f:
        if h == "":
            f.write(post_title)
        else:
            f.write(h)
    if post_id != "":
        if raw_dir.joinpath(broken_fname).exists():
            print("replace {} with {}".format(broken_fname, fname))
            raw_dir.joinpath(broken_fname).unlink()


def check_progress(ds_csv_path):
    dir_name = '_'.join(ds_csv_path.stem.split('_')[2:-1] + ["raw"])
    raw_dir = ds_csv_path.parent.joinpath(dir_name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files_path = list(raw_dir.glob("*.html"))
    try:
        row = list(map(lambda f: int(f.stem.split('_')[0]), files_path))
        return max(row) + 1, raw_dir
    except:
        return 0, raw_dir


def get_no_success_crawl_row():
    rows = []
    data_files = os.listdir('dataset/raw/')
    for file in data_files:
        l = file.split("_")
        if ".html" == l[1]:
            rows = rows + [int(l[0])]

    return rows


def check_missing(ran):
    files = os.listdir('dataset/raw/')
    had_rows = list(map(lambda x: int(x.split("_")[0]), files))
    had_rows.sort()
    missing = set(ran) - set(had_rows)
    return sorted(missing)


def main(args):
    dataset_path = Path(args.path)
    for ds_csv_path in list(dataset_path.glob('intern*.csv')):
        opt = webdriver.ChromeOptions()
        dc = DesiredCapabilities.CHROME
        dc["pageLoadStrategy"] = "none"
        driver = webdriver.Chrome(chrome_options=opt,
                                  desired_capabilities=dc)
        get_posts_and_save_html(ds_csv_path, driver, opt, dc)
        driver.close()
        driver.quit()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Dcard Crawler')
    args.add_argument('-p', '--path', default="../../dataset/", type=str,
                      help="path to dataset (default: ../../dataset/)")
    args = args.parse_args()
    print("Start Dcard crawler ...")
    print("opening csv in ", Path(args.path).resolve())

    main(args)
