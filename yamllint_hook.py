#!/usr/bin/env python
import sys
import subprocess
import os
import tempfile
import shutil


def convert_to_utf8(file_path):
    with open(file_path, 'rb',  encoding="utf-8" ) as f:
        raw_data = f.read()
    try:
        decoded_data = raw_data.decode('utf-8')
    except UnicodeDecodeError:
        decoded_data = raw_data.decode('cp1252')
    return decoded_data.encode('utf-8')


def main():
    files = sys.argv[1:]
    temp_dir = tempfile.mkdtemp()

    temp_files = []
    try:
        for file in files:
            temp_file = os.path.join(temp_dir, os.path.basename(file))
            with open(temp_file, 'wb',  encoding="utf-8") as f:
                f.write(convert_to_utf8(file))
            temp_files.append(temp_file)

        result = subprocess.run(['yamllint'] + temp_files)
        return result.returncode
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    sys.exit(main())
