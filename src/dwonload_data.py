import os
import csv
from urllib import request
from bs4 import BeautifulSoup

# data and files configure
DATA_DIR = '../data'
mirror_urls_file = 'mirror_urls.txt'
mirror_file = 'mirror.csv'
logs_file = os.path.join(DATA_DIR, 'crawler.log')

# urls configure
BASE_URL = 'http://xssed.com'
# pages url: http://xssed.com/page=2/
# mirror url: http://xssed.com/mirror/81219/


def get_urls(outfile):
    page_no = 1
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
            with open(outfile, 'a') as f_out:
                for item in result_tags:
                    try:
                        f_out.write(BASE_URL + item['href'] + '\n')
                    except Exception as e:
                        print(e)
                        # error logs
                        with open(logs_file, 'a') as f_err:
                            f_err.write("ERROR TO GET PAGE {}\n".
                                        format(page_no))
                        page_no += 1
                        continue
            print("SUCCEED TO GET PAGE {}".format(page_no))
            page_no += 1
        else:
            break


def parse_mirror(tags):
    # Date submitted
    # Date published
    # Author
    header = []
    body = []
    for index, item in enumerate(tags):
        if index == 2:
            continue
        text = item.get_text()
        tags_kv = "".join(text.split()).split(':', 1)  # remove &nbsp;
        header.append(tags_kv[0])
        body.append(tags_kv[1])
    return header, body


def get_mirrors(infile, outfile):
    with open(infile, 'r') as f_in:
        mirror_urls = [item.strip() for item in f_in.readlines()]

    for url_idx, url in enumerate(mirror_urls):
        # if url != 'http://xssed.com/mirror/5120/':
        #     continue
        req = request.Request(url)

        try:
            res = request.urlopen(req)
        except Exception as e:
            print(e)
            # error log
            with open(logs_file, 'a') as f_err:
                f_err.write("ERROR TO GET MIRROR FROM {}\n".
                            format(url))
            continue

        page = res.read().decode('utf-8')
        # parse page
        soup = BeautifulSoup(page, 'html.parser')
        result_tags = soup.select('#contentpaneOpen')[0].find_all('th')
        # only neeed tages 2~10
        header, body = parse_mirror(result_tags[1:10])
        with open(outfile, 'a', newline='') as f_out:
            csv_write = csv.writer(f_out, dialect='excel')
            if url_idx == 0:
                # write header
                csv_write.writerow(header)
            csv_write.writerow(body)

        print("{}.SUCCEED TO GET MIRROR FROM {}".format(url_idx, url))


def main():
    global mirror_urls_file, mirror_file
    mirror_urls_file = os.path.join(DATA_DIR, mirror_urls_file)

    if not os.path.exists(mirror_urls_file):
        get_urls(mirror_urls_file)
    else:
        # mirror_urls_file has already existed.
        # get mirrors
        mirror_file = os.path.join(DATA_DIR, mirror_file)
        get_mirrors(mirror_urls_file, mirror_file)

if __name__ == '__main__':
    main()
