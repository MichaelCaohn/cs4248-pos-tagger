# python3.5 eval.py <output_file_absolute_path> <reference_file_absolute_path>
# make no changes in this file

import os
import sys
from collections import defaultdict
import json


if __name__ == "__main__":
    out_file = sys.argv[1]
    reader = open(out_file)
    out_lines = reader.readlines()
    reader.close()

    ref_file = sys.argv[2]
    reader = open(ref_file)
    ref_lines = reader.readlines()
    reader.close()

    if len(out_lines) != len(ref_lines):
        print('Error: No. of lines in output file and reference file do not match.')
        exit(0)

    total_tags = 0
    matched_tags = 0
    mistagged = defaultdict(lambda:defaultdict(lambda:0))
    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tags = cur_out_line.split(' ')
        cur_ref_line = ref_lines[i].strip()
        cur_ref_tags = cur_ref_line.split(' ')
        total_tags += len(cur_ref_tags)

        for j in range(0, len(cur_ref_tags)):
            if cur_out_tags[j] == cur_ref_tags[j]:
                matched_tags += 1
            else:
                true_tag = cur_ref_tags[j].rpartition('/')[2]
                false_tag = cur_out_tags[j].rpartition('/')[2]
                mistagged[true_tag][false_tag] += 1
                # if true_tag == 'NN' and false_tag == 'JJ':
                # print(cur_out_tags[j], cur_ref_tags[j])
                # if true_tag == 'VBD' and cur_ref_tags[j].rpartition('/')[0][-1] != 'd':
                #     print(cur_ref_tags[j])

    print("Accuracy=", float(matched_tags) / total_tags)
    # print(json.dumps(mistagged, indent=2))
