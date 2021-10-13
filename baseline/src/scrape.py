"""
This file contains the main script responsible for fetching text data from
Project Gutenberg's top Fantasy Fiction novels.
"""
import argparse
import bs4
import requests
import re
import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

base_url = 'https://www.gutenberg.org'

headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
}


def get_soup_content(url: str):
    request = requests.get(url, headers)
    content = request.content.decode('utf-8')
    soup = bs4.BeautifulSoup(content, features="html.parser")
    return soup


def scrape_text(read_online_url: str):
    soup = get_soup_content(read_online_url)
    blocks = map(lambda fragment: fragment.get_text(
        strip=True).lower(), soup.find_all('p', class_=''))
    for block in blocks:
        lines = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+', block)
        for line in lines:
            with open('novels.txt', 'a+') as file:
                file.write(f"{line}\n")


def scrape_book_link(book_url: str):
    soup = get_soup_content(book_url)
    bibrec = soup.find('table', class_='bibrec')
    language = bibrec.find(
        'tr', {'property': 'dcterms:language'}).find('td').text

    if language != 'English':
        return

    read_online_url = soup.find('tr', class_='even').find(
        'a', class_='link', href=True)['href']

    if not read_online_url:
        return

    text_name = soup.find('h1', {'itemprop': 'name'}).text
    scrape_text(text_name, base_url + read_online_url)


def scrape_page(idx: int, book_titles: set):
    idx_url = base_url + f"/ebooks/subject/138?start_index={idx}"
    soup = get_soup_content(idx_url)

    book_links = soup.find_all('li', class_='booklink')

    for book_link in tqdm.tqdm(book_links):
        title = book_link.find('span', class_='title')
        if title in book_titles:
            continue
        book_titles.add(title)
        book_url_ending = book_link.find('a', href=True)['href']
        scrape_book_link(base_url + book_url_ending)


def driver_scrape(max_pages, n_threads):
    page_indices = [i for i in range(1, (max_pages*25)+1, 25)]
    open('novels.txt', 'a+').close()
    book_titles = set()
    partial_scrape_page = partial(scrape_page, book_titles=book_titles)

    with ThreadPool(n_threads) as pool:
        pool.map(partial_scrape_page, page_indices)


def main(args):
    driver_scrape(args.pages, args.threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scrape from Project Gutenberg')
    parser.add_argument('--threads', type=int,
                        help='Number of threads to scrape with')
    parser.add_argument('--pages', type=int,
                        help='Number of pages of the top fantasy novels to scrape (each page has 25 novels), max 325')
    args = parser.parse_args()
    main(args)
