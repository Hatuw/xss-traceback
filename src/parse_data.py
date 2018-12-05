# coding: utf-8
import os
import re
import pandas as pd

# set global vars
DATA_DIR = '../data'
mirror_urls_file = 'mirror_demo.csv'


def load_data():
    """
        load mirrors file
        @return:
            pruned_data: <pandas.DataFrame>
    """
    file_path = os.path.join(DATA_DIR, mirror_urls_file)
    assert os.path.exists(file_path), "file not exist"
    data = pd.read_csv(file_path)

    # get data columns
    columns = data.columns
    if 'Author' in columns and 'URL' in columns:
        # extract `Author` and `URL` from data
        pruned_data = data.loc[:, ['Author', 'URL']]
    else:
        exit(1)

    return pruned_data


def split_urls(urls):
    """
        split urls
        param:
            urls: string[]
        @return:
            splited_urls: string[]
    """
    result_urls = []

    # remove domain
    fix_urls = [re.split('/', url, 3)[-1] for url in urls]

    for idx, url in enumerate(fix_urls):
        # split %xx
        tmp_url = re.split(r'(%\w{2})', url)
        tmp_url = [item for item in tmp_url if item]
        fix_urls[idx] = tmp_url

        # split symbol
        tmp_fixed_url = []
        for item in fix_urls[idx]:
            if not re.match(r'%\w{2}', item):
                for re_result in re.split(r'(\W)', item):
                    if re_result:
                        tmp_fixed_url.append(re_result)
            else:
                tmp_fixed_url.append(item)

        result_urls.append(tmp_fixed_url)
    else:
        return result_urls


# parse data
def main():
    # load data
    pruned_data = load_data()
    assert 'URL' in pruned_data.columns, 'data format error'
    pruned_urls = split_urls(pruned_data['URL'])

    for i in range(50):
        if len(pruned_urls[i]) < 30:
            print(pruned_urls[i])


if __name__ == '__main__':
    main()
