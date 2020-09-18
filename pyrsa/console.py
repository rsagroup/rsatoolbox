"""Command line interface function
"""
import argparse
import pkg_resources


def main():
    """This function gets called when the user executes `rsa3`.

    It defines and interprets the console arguments, then calls
    the relevant python code.
    """
    version = pkg_resources.get_distribution('rsa3').version

    ## define the main parser
    parser = argparse.ArgumentParser(prog='rsa3')
    parser.add_argument('--version', action='version', version=version)
    _ = parser.add_subparsers(dest='command', title='subcommands')

    args = parser.parse_args()
    print(args.command)
