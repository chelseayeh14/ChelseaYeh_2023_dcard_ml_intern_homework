import json
from datetime import datetime, timezone


class DcardPost:

    def __init__(self, soup=None):
        self.post_id = ""
        self.title = ""
        self.comment_count_now = 0
        self.like_count_now = 0
        self.created_at = ""
        self.au_name = ""
        self.au_gender = ""
        self.content_text = ""
        self.forum_name = ""
        self.img_count = 0
        if soup != None:
            self.parse_soup(soup)

    # 方法(Method)

    def convert_time_format(self, raw_time):
        dt = datetime.fromisoformat(raw_time)
        utc_dt = dt.astimezone(timezone.utc)
        formatted_dt = utc_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        self.created_at = formatted_dt

    def parse_soup(self, soup):
        script = soup.find('script', {'type': 'application/ld+json'})
        if script == None:
            self.title = soup.text
            return
        article_tag = soup.find('article')
        self.parse_script(script)
        self.parse_article_tag(article_tag)

    def parse_script(self, script):
        j = json.loads(script.text)
        if type(j) == type([]):  # exclude not normal post, ex: "@type": "Product"
            j = j[0]
        if j["url"] != "https://www.dcard.tw":
            self.post_id = j['url'].split('/')[-1]
            self.forum_name = j['url'].split('/')[-3]
            self.title = j['headline']
            self.now_comment_count = j['commentCount']
            self.now_like_count = j['interactionStatistic']['userInteractionCount']
            self.content_text = j['description']
            try:
                self.au_name = j['author']['name']
            except:
                self.au_name = ""
            try:
                self.au_gender = j['author']['gender']
            except:
                self.au_gender = ""

            self.convert_time_format(j['datePublished'])

    def parse_article_tag(self, article_tag):
        try:
            self.content_text = article_tag.text
            self.img_count = len(
                article_tag.find_all("div", {'role': "button"}))
        except:
            pass
