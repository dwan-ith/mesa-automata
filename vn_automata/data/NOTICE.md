# Bundled reference artefact: `JvN29.rule`

`JvN29.rule` is the von Neumann 29-state cellular automaton rule table as
distributed with [Golly](http://golly.sourceforge.net/), the standard cellular
automata simulator. It is mirrored here from:

- https://github.com/EZLiang/vonNeumannTuringMachine/blob/master/JvN29.rule

It is used **only** as an external validation reference: `vn_automata/golly.py`
parses it and re-implements the transition function independently, and
`tests/test_golden_golly.py` checks that the production engine reproduces it
over the entire 29^5 state-transition space.

The original model is described in:

- John von Neumann, *Theory of Self-Reproducing Automata*, edited and completed
  by Arthur W. Burks, University of Illinois Press, 1966.

If you redistribute this repository, keep this notice with the data file so the
provenance of the reference table remains clear.
