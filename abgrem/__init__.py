"""
Paratope scoring and mutation prioritisation for anti-GREM antibodies.

The package is organised around the two stages of the method:

``config``     study definition and every scoring constant
``msa``        repertoire alignment statistics (frequency, conservation)
``structure``  docked-complex geometry (contacts, distance score)
``scoring``    Stage 1 position selection and Stage 2 PSBDM ranking
``figures``    manuscript figures, each written alongside its source CSV
"""

__version__ = "1.0.0"
