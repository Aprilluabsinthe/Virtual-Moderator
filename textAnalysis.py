from googlesearch import search
import urllib.request
from bs4 import BeautifulSoup


def getLinks(query, totalLinks=10):
    results = search(query=query, num=totalLinks, stop=totalLinks, lang='en')
    links = []
    for result in results:
        links.append(result)
    return links


def contentFromLink(url):
    """ basic Bs4 code taken from https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text"""
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, features='lxml')
    # kill all script and style elements
    for script in soup({"script", "style",
                        'noscript',
                        'header',
                        'meta',
                        'input',}):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '...'.join(chunk for chunk in chunks if len(chunk))

    return text


content = contentFromLink(getLinks('nvidia', 5)[3])

print(content)

