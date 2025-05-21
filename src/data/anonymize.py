import hashlib
import argparse

def anonymize(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
    anonymized_lines = []
    for l in lines:
        split_prefix = l[:l.find("[")].split(" ")
        split_suffix = "".join(l[l.find("["):].split('"')[:2]) #time and get requesti
        anonymized_lines.append(hashlib.md5(split_prefix[1].encode()).hexdigest() 
                                + " " + hashlib.md5(split_prefix[2].encode()).hexdigest() 
                                + " " + split_suffix + "\n")
    with open("anonymized_"+fn,"w") as f:
        f.writelines(anonymized_lines)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default='sample.txt')

    args = parser.parse_args()
    anonymize(args.file)
