import re

from googlesearch import search


def search_with_google(title):
    url = ""
    for url in search(title + " dcard", num=10, lang="zh-tw", stop=10):
        # print("google {} get this {}".format(title, url))
        if check_url_in_dcard(url):
            print(">>>>find {} from dcard".format(url))
            return check_url_in_dcard(url), url
    return False, url


def check_url_in_dcard(url):
    pattern = r'https://www\.dcard\.tw/f/[\w_]+/p/\d+'
    return re.match(pattern, url)


if __name__ == "__main__":
    search_with_google("排骨")
