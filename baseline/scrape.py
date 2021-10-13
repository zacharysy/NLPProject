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
    """
    Util for getting soup content from url.
    """
    request = requests.get(url, headers)
    content = request.content.decode('utf-8')
    soup = bs4.BeautifulSoup(content, features="html.parser")
    return soup


def scrape_text(read_online_url: str):
    """
    Scrape the book using the "Read Online" option
    and write all of the text blocks that make up the novel
    to a file line-by-line.
    """
    soup = get_soup_content(read_online_url)
    blocks = map(lambda fragment: fragment.get_text(
        strip=True).lower(), soup.find_all('p', class_=''))
    for block in blocks:
        # Split lines by ., ?, !, and any of the previous followed by a quote.
        lines = re.split(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+', block)
        for line in lines:
            with open('novels.txt', 'a+') as file:
                file.write(f"{line}\n")


def scrape_book_link(book_url: str):
    """
    Check validity of the given book page.
    Reject if the novel isn't in English or it's not readable online.
    """
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

    scrape_text(base_url + read_online_url)


def scrape_page(idx: int, book_titles: set):
    """
    Scrape whole page of books. Each page has 25 books.
    """
    idx_url = base_url + f"/ebooks/subject/138?start_index={idx}"
    soup = get_soup_content(idx_url)

    book_links = soup.find_all('li', class_='booklink')

    for book_link in tqdm.tqdm(book_links):
        title = book_link.find('span', class_='title').text
        # Avoid scraping copies of the same book.
        if title in book_titles:
            continue
        book_titles.add(title)
        book_url_ending = book_link.find('a', href=True)['href']
        scrape_book_link(base_url + book_url_ending)


def driver_scrape(pages, n_threads):
    """
    Main driver function for scraping pages in parallel.
    """
    page_indices = [i for i in range(1, (pages*25)+1, 25)]
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
