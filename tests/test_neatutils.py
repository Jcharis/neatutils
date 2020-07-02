from neatutils import __version__
from neatutils import get_abbrev,get_fullname


def test_version():
    assert __version__ == '0.0.1'


def test_get_abbrev():
	result = get_abbrev('Logistic Regression')
	assert result == 'lr'

def test_get_abbrev_type():
	result = get_abbrev('Linear Regression','regression')
	assert result == 'lr'


def test_get_fullname():
	result = get_fullname('lr')
	assert result == 'Logistic Regression'
