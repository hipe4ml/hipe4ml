""" Test suite to be run wit h pytest """


def inc(x_var):
    """ Dummy function decl, thanks for asking
    """
    return x_var + 1


def test_answer():
    """ This tests tests the testing function
    """
    assert inc(3) == 4
