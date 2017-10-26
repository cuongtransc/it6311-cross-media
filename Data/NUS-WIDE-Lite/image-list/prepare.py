#!/usr/bin/env python3

import os


def get_cates(data_urls_filename):
    cates = set()

    with open(data_urls_filename) as f:
        for line in f:
            cate = line.strip().split(',')[0]
            cates.add(cate)

    return cates


def create_cate_directories(cates):
    root_dir = 'images'

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for cate in cates:
        data_dir = '{}/{}'.format(root_dir, cate)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)


def main(data_urls_filename):
    cates = get_cates(data_urls_filename)

    create_cate_directories(cates)


if __name__ == '__main__':
    data_urls_filename = 'train_urls.txt'
    main(data_urls_filename)
