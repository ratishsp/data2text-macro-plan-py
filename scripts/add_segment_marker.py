import argparse
import nltk


def process(input_file_name, output_file_name):
    input_file = open(input_file_name, mode='r', encoding='utf-8')
    output_file = open(output_file_name, mode='w', encoding='utf-8')
    for line in input_file:
        sentences = nltk.sent_tokenize(line)
        output_line = " <segment> ".join(sentences)
        output_file.write(output_line)
        output_file.write("\n")
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add <segment> markers between each sentence')
    parser.add_argument('-input_file', type=str,
                        help='path of input file', default=None)
    parser.add_argument('-output_file', type=str,
                        help='path of output file', default=None)
    args = parser.parse_args()

    process(args.input_file, args.output_file)
