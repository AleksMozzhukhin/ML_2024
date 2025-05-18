import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    if not sample:
        return {"gini": 0.0, "entropy": 0.0, "error": 0.0}

    values, counts = np.unique(sample, return_counts=True)
    sample_probability = counts / len(sample)

    measures = dict()
    measures['gini'] = 1 - sum(sample_probability ** 2)
    measures['entropy'] = -sum(sample_probability * np.log(sample_probability))
    measures['error'] = 1 - max(sample_probability)
    return measures
