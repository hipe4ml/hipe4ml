""" Test suite to be run wit h pytest """


def inc(xVar):
    """ Dummy function decl, thanks for asking
    """
    return xVar + 1


def testAnswer():
    """ This tests tests the testing function
    """
    assert inc(3) == 4
