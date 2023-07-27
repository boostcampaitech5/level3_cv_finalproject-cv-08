import argparse
import os


def main(args):
    for path in args.data_paths:
        for filename in os.listdir(path):
            link_target = os.path.abspath(os.path.join(args.target_path, filename))
            link_original = os.path.abspath(os.path.join(path, filename))
            os.system(f'ln -s "{link_original}" "{link_target}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_paths", type=str, nargs="+")
    parser.add_argument("--target_path", type=str)

    args = parser.parse_args()

    main(args)
