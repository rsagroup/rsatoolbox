"""Just-in-time importing of optional dependencies

These can be installed with rsatoolbox using the square brackets syntax;
```
pip install rsatoolbox[imaging]
```
"""


def import_nibabel(mock=None):
    """Try to access the nibabel module or raise an exception.

    Nibabel is an open source python library for reading and writing
    MRI files. Used by the BIDS functionality.

    Args:
        mock (Mock, optional): When testing, a Mock object 
        can be injected here. Defaults to None.

    Raises:
        OptionalImportMissingException: Raised when the dependency 
        is not installed

    Returns:
        nibabel: nibabel main module
    """
    if mock:
        return mock
    try:
        import nibabel
    except ImportError:
        raise OptionalImportMissingException('nibabel')
    return nibabel


def import_nitools(mock=None):
    """Try to access the neuroimagingtools module or raise an exception.

    Neuroimagingtools is an open source python library for efficient 
    access to Nifti, Gifti and Cifti files. Used by the SpmGlm class.

    Args:
        mock (Mock, optional): When testing, a Mock object 
        can be injected here. Defaults to None.

    Raises:
        OptionalImportMissingException: Raised when the dependency 
        is not installed

    Returns:
        nitools: neuroimagingtools main module
    """
    if mock:
        return mock
    try:
        import nitools
    except ImportError:
        raise OptionalImportMissingException('nitools')
    return nitools


class OptionalImportMissingException(Exception):

    def __init__(self, name: str):
        super().__init__(f'[rsatoolbox] Missing optional dependency: {name}')
