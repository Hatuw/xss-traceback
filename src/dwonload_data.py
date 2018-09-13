import os
from urllib import request
from bs4 import BeautifulSoup

# data and files configure
DATA_DIR = '../data'
mirror_urls_file = 'mirror_urls.txt'
logs_file = os.path.join(DATA_DIR, 'crawler.log')

# urls configure
BASE_URL = 'http://xssed.com/archive'
# pages url: http://xssed.com/archive/page=2/
# mirror url: http://xssed.com/mirror/81219/

page_no = 1
mirror_urls_file = os.path.join(DATA_DIR, mirror_urls_file)

# about 1530 pages
# 30 * 1529 + 14 = 45884 records
while True:
    page_url = '{}/page={}'.format(BASE_URL, page_no)
    req = request.Request(page_url)
    try:
        res = request.urlopen(req)
    except Exception as e:
        print(e)
        # error logs
        with open(logs_file, 'a') as f_err:
            f_err.write("ERROR TO GET PAGE {}\n".format(page_no))
        page_no += 1
        continue

    page = res.read().decode('utf-8')
    # parse page
    soup = BeautifulSoup(page, 'html.parser')
    result_tags = soup.find_all('a', string='mirror')

    if result_tags:
            # write urls
        with open(mirror_urls_file, 'a') as f_out:
            for item in result_tags:
                try:
                    f_out.write(BASE_URL + item['href'] + '\n')
                except Exception as e:
                    print(e)
                    # error logs
                    with open(logs_file, 'a') as f_err:
                        f_err.write("ERROR TO GET PAGE {}\n".format(page_no))
                    page_no += 1
                    continue
        print("SUCCEED TO GET PAGE {}".format(page_no))
        page_no += 1
    else:
        break
