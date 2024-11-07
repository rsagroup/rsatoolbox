"""Command Line Interface
"""
from argparse import ArgumentParser

parser = ArgumentParser(
    prog='rsatoolbox',
    description='Representational Similarity Analysis'
)


def main():
    parser.parse_args()
