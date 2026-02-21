"""Quick demo using music21's built-in corpus."""
from music21 import corpus

s = corpus.parse('bach/bwv65.2.xml')
s.show('midi')
