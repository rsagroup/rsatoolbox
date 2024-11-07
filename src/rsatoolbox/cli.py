"""Command Line Interface
"""
from argparse import ArgumentParser
from importlib.metadata import version

parser = ArgumentParser(
    prog='rsatoolbox',
    description='Representational Similarity Analysis'
)
parser.add_argument(
    'file',
    default='.',
    help='Data directory or file'
)  
parser.add_argument(
    '--version',
    action='version',
    version=f'rsatoolbox {version("rsatoolbox")}'
)


def main():
    parser.parse_args()
