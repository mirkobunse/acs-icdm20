import numpy as np
import pytest
from sandbox import data, methods # import from this project
import sandbox.methods.acs

def test__select_from_pool__basic():
	"""A simple test case which failed during an early development stage"""
	y = np.array([2, 1, 2, 1, 1, 0])
	scores = np.array([.1, .2, .2])

	selection = methods.acs.select_from_pool(y, scores, N=2)
	assert len(np.unique(selection)) == len(selection) # all indices are unique
	selection_counts = data.label_counts(y[selection], C=3)
	np.testing.assert_array_equal(selection_counts, [0, 1, 1]) # the selection is correct

@pytest.mark.parametrize('i', range(100))
def test__select_from_pool__random(i):
	"""Systematic test with multiple random situations"""
	C = np.random.choice(9)+1 # the number of classes
	N = np.random.choice(9)+1 # the number of samples to select
	y = np.random.choice(C, size=max((np.random.choice(9)+1)*C, N))
	label_counts = data.label_counts(y, C)
	scores = np.random.uniform(size=C)
	normalized_scores = scores / scores.sum()
	print('N = {}\nN_c = {}\ns_c = {}'.format(
		N, label_counts, normalized_scores
	)) # usually only printed if an assertion fails below

	selection = methods.acs.select_from_pool(y, scores, N=N)
	selection_counts = data.label_counts(y[selection], C)
	print('N_s = {}'.format(selection_counts))
	assert len(np.unique(selection)) == len(selection) # all indices are unique
	assert len(selection) == N # the correct number of samples is selected
	if N < len(y):
		np.testing.assert_allclose(
			selection_counts,
			np.minimum(normalized_scores * len(selection), label_counts),
			rtol=0,
			atol=np.max(np.abs(normalized_scores * len(selection) - label_counts))
		) # the selection is basically correct
		np.testing.assert_array_equal(
			selection_counts[(normalized_scores * len(selection)) >= label_counts],
			label_counts[(normalized_scores * len(selection)) >= label_counts]
		) # if all samples of a class must be selected, they are
	else:
		pass # all samples are selected, no additional check is required

	# test again, now only selecting a single example
	selection = methods.acs.select_from_pool(y, scores, N=1)
	scores[label_counts == 0] = -np.inf # ignore those when looking for max scores
	best_labels = np.argwhere(scores == np.max(scores)).flatten() # all max scores
	assert y[selection[0]] in best_labels

	# ..and again, now in spite of multiple maximum scores
	available_labels = set(range(C)) \
	  .difference(best_labels) \
	  .difference(np.argwhere(label_counts == 0).flatten())
	if len(available_labels) > 0:
		scores[np.random.choice(list(available_labels))] = np.max(scores)
		best_labels = np.argwhere(scores == np.max(scores)).flatten() # all max scores
		print('Scores with more than one maximum: {}'.format(scores))
		selection = methods.acs.select_from_pool(y, scores, N=1) # select again
		assert len(best_labels) > 1 # assert that the test setup is correct
		assert y[selection[0]] in best_labels
	else:
		pass # this test is not feasible if C=1 or no additional examples exist
