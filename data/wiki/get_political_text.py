#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path
from blingfire import text_to_sentences
from os.path import isfile, join
from tqdm import tqdm

def main():
    wiki_dump_file_out = "./political_text/political_text_full_len.txt"
    wiki_source_title_file = "./political_text/political_text_source.txt"
    wikitext_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk("./text") for f in filenames ] # get all files in the text folder
    with open(wiki_dump_file_out, 'w', encoding='utf-8') as out_f:
        with open(wiki_source_title_file, 'w', encoding='utf-8') as out_f2:
            for file in tqdm(wikitext_files):
                with open(file, 'r') as in_f:
                    for line in in_f:
                        tmp = json.loads(line)
                        if ('政府工作报告' in tmp['title']) or ('人民政府' in tmp['title']) or ():
                            #print(tmp['title'])
                            text = tmp['text']
                            sentences = text.replace('\n','[SEP]')
                            #sentences = text_to_sentences(text)
                            out_f2.write(tmp['title'] + '\t' + tmp['url'] + '\n')
                            out_f2.flush()
                            out_f.write(sentences + '\n')
                            out_f.flush()
                in_f.close()
        out_f2.close()
    out_f.close()

if __name__ == '__main__':
    main()

            