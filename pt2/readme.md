
# Pt.2 Improvements, Optimizations & Scaling...

Well when you need to build something to scale rapidly, processing data becomes a topic... So how do we do this?

## Changes in Pt.2

* The modification of the evaluate_fuzziness function uses a LevenShtein (see https://en.wikipedia.org/wiki/Levenshtein_distance) approach as a scoring mechansim
* As the number of dendrites increase, so does the processing time - So time for a clustered index ;)
* Move of the code to separate dendrite and axon from the main neuralnet code function
* Added a few more tests
* Tweaked the formatting a little
* Dendrite has an autogenerating identifier now (using uuid7) as opposed to an incrementing id