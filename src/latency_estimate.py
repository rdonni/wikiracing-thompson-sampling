from urllib.request import urlopen
from tqdm.notebook import tqdm
import pywikibot
from time import time


def estimate_url_latency(url):
    website = urlopen(url)
    open_time = time()
    output = website.read()
    close_time = time()
    website.close()
    return (close_time - open_time) * 1000


def get_url_from_article_title(title):
    site = pywikibot.Site('en', 'wikipedia')
    page = pywikibot.Page(site, title)
    url = page.full_url()
    return url


def estimate_articles_latency_once(node_list):

    latency_estimates = {}
    for title in tqdm(node_list):
        article_url = get_url_from_article_title(title)
        latency = estimate_url_latency(article_url)

        latency_estimates[title] = latency
    return latency_estimates