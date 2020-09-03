from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from time import sleep
import requests
import re
import spacy
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import googletrans
from googletrans import Translator
import wikipedia
from collections import Counter
from pprint import pprint
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

URL = 'https://main.knesset.gov.il/mk/Pages/current.aspx?pg=mklist'
BASE_URL = 'https://main.knesset.gov.il/'
nlp = spacy.load("en_core_web_lg")
excluded_websites = ['facebook', 'youtube']


def check_excluded(url, excluded_sites=excluded_websites):
    include = True
    for site in excluded_sites:
        include *= site not in url
    exclude = not include
    return exclude


def clean(s: str):
    """
    Cleans a string from unwanted characters
    :param s: string to be cleaned
    :return: the clean string
    """
    s.strip('\n').strip('\t')
    return s


def get_names_from_webpage(url):
    """
    Gets all named entities which are a person from a given url's text
    :param url: web page
    :return: list of possible names
    """
    page = requests.get(url).text
    clean_page = BeautifulSoup(page, 'lxml').get_text().replace('\n', ' ').replace('\t', '').replace('\r', '')
    doc = nlp(clean_page)
    names = [clean(ent.text) for ent in doc.ents if ent.label_ == 'PERSON']
    return names


def get_names_from_all_webpages(urls):
    """
    Gets all named entities which are a person from a list of urls
    :param urls: list of urls
    :return: list of possible names
    """
    names = []
    for url in urls:
        if not check_excluded(url):
            print(f'getting names from {url}')
            names += get_names_from_webpage(url)
        else:
            print(f'skipping excluded {url}')
    return names


def google_mk_urls(query, n_results=10):
    """
    Searches a query on google, return the urls
    valid options for n_results: 10, 20, 30, 40, 50, and 100
    """
    page = requests.get(f"https://www.google.com/search?q={query}&num={n_results}")
    soup = BeautifulSoup(page.content, "html5lib")
    links = soup.findAll("a")

    urls = []
    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and "webcache" not in link_href:
            url = link.get('href').split("?q=")[1].split("&sa=U")[0]
            print(url)
            urls.append(url)
    return urls


def get_mklist():
    """
    Scraping MK related data from the the Knesset website
    :return: list of dicts, each dict contains MK data
    """
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(URL)
    elements = driver.find_elements_by_class_name('MKLobbyMKNameDiv')

    mk_list = []
    for elem in elements:
        mk_name = elem.text
        soup = BeautifulSoup(elem.get_attribute('innerHTML'))
        mk_url = soup.find('a', href=True)['href']
        mk_details = dict()
        mk_details['name'] = mk_name
        mk_details['url'] = BASE_URL + mk_url
        mk_list.append(mk_details)
        print(mk_details)

    new_mk_list = []
    for mk in mk_list:
        driver.get(mk['url'])
        soup = BeautifulSoup(driver.find_elements_by_class_name('MKPersonalContent')[0].get_attribute('innerHTML'))
        mk_details['name'] = mk['name']
        mk_details['url'] = mk['url']

        try:
            mk_details['date_of_birth'] = soup.find('span',
                                                    id='ctl00_ctl57_g_4e85a275_d641_41ab_8bc5_98c140927f3d_BirthDateSpn').text
        except Exception as e:
            print(f'could not get date_of_birth for {mk["name"]}')

        try:
            mk_details['place_of_birth'] = soup.find('span',
                                                     id='ctl00_ctl57_g_4e85a275_d641_41ab_8bc5_98c140927f3d_countrySpn').text
        except Exception as e:
            print(f'could not get place_of_birth for {mk["name"]}')

        try:
            mk_details['city_of_residence'] = soup.find('span',
                                                        id='ctl00_ctl57_g_4e85a275_d641_41ab_8bc5_98c140927f3d_citySpn').text
        except Exception as e:
            print(f'could not get city_of_residence for {mk["name"]}')

        print(mk_details)
        new_mk_list.append(mk_details.copy())

    sleep(3)
    driver.close()
    return new_mk_list


def get_google_transliteration(s: str):
    """
    returns a translation of a
    :param s:
    :return:
    """
    translator = Translator()
    return translator.translate(s)


# if __name__ == '__main__':
# df = pd.read_csv('./mk_list.csv')
# google_mk_urls('Miri Regev')
# names = get_names_from_webpage('http://www.israel.org/MFA/AboutIsrael/State/Personalities/Pages/Miri-Regev-MK.aspx')
# print(names)

urls = google_mk_urls('Miri Miriam Regev -"Netanyahu"', 50)
names = get_names_from_all_webpages(urls)
c = Counter(names)
pprint(c.most_common(20))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(names)
print(vectorizer.get_feature_names())
print(X.shape)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
zipped = pd.DataFrame(zip(names, kmeans.labels_))
