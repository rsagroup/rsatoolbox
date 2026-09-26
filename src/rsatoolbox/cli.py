"""Command Line Interface

Usage output:
```
Representational Similarity Analysis

positional arguments:
  file                  One or more files or directories with data

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --model MODEL         File with RDM models
  --estimator {auto,euclidean,correlation,mahalanobis,crossnobis,poisson,poisson_cv}
                        Which dissimilarity measure to generate RDM with
  --comparator {auto,cosine,spearman,corr,kendall,tau-a,rho-a,corr_cov,cosine_cov,neg_riem_dist,bures,bures_metric}
                        Which method to use when evaluating models or comparing RDMs
  --searchlight SEARCHLIGHT
                        Run a searchlight over the whole space with the given radius. Unit depends on data modality.
  --window [WINDOW ...]
                        Boundaries of a temporal window in milliseconds, e.g. "150 200"
  --roi [ROI ...]       One or more "mask" or "picks" files to specify voxel or channel selection
```

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
    help='One or more files or directories with data',
    nargs='*',
)
parser.add_argument(
    '--version',
    action='version',
    version=f'rsatoolbox {version("rsatoolbox")}'
)
parser.add_argument(
    '--model',
    help='File with RDM models',
)
parser.add_argument(
    '--estimator',
    help='Which dissimilarity measure to generate RDM with',
    choices=[
        'auto',
        'euclidean',
        'correlation',
        'mahalanobis',
        'crossnobis',
        'poisson',
        'poisson_cv',
    ],
    default='auto',
)
parser.add_argument(
    '--comparator',
    help='Which method to use when evaluating models or comparing RDMs',
    choices=[
        'auto',
        'cosine',
        'spearman',
        'corr',
        'kendall',
        'tau-a',
        'rho-a',
        'corr_cov',
        'cosine_cov',
        'neg_riem_dist',
        'bures',
        'bures_metric',
    ],
    default='auto',
)
parser.add_argument(
    '--searchlight',
    help='Run a searchlight over the whole space with the given radius. Unit depends on data modality.',
    type=int,
    default=0,
)
parser.add_argument(
    '--window',
    help='Boundaries of a temporal window in milliseconds, e.g. "150 200"',
    nargs='*',
)
parser.add_argument(
    '--roi',
    help='One or more "mask" or "picks" files to specify voxel or channel selection',
    nargs='*',
)


def main():
    parser.parse_args()
