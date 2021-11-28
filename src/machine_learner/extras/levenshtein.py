import numpy as np


def levenshtein_nearest(token: str, list_tokens: iter) -> str:
    """
    Given a string and a list of strings, this function returns the nearest token to the string with respect to the
    Levenshtein distance.

    :param token: a string
    :param list_tokens: a list of strings
    :return: the nearest string in list_tokens to token, according to the Levenshtein distance
    """

    nearest_token, distance = "", 10e2

    for candidate in list_tokens:

        if (new_distance := levenshtein_distance(token, candidate)) < distance:
            nearest_token, distance = candidate, new_distance

    return nearest_token


def levenshtein_distance(token1: str, token2: str) -> int:
    """
    This function returns the Levenshtein distance between two strings.

    :param token1: a string
    :param token2: another string, to be compared with the first one
    :return: an integer, the Levenshtein distance between the two input strings
    """

    # Initialize the distances matrix
    distances = np.zeros((len(token1) + 1, len(token2) + 1)).astype("int8")

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    # Compute distance between every prefix
    for t1 in range(1, len(token1) + 1):

        for t2 in range(1, len(token2) + 1):

            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]

            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                distance = min(a, b, c) + 1

                distances[t1][t2] = distance

    return distances[len(token1)][len(token2)]


levenshtein_nearest("mon truc", ["ma truc", "machin", "bal", "c'est faux"])
