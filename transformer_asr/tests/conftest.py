import pytest

from tests.test_positional_encoding import UnetTest
from tests.test_subsampling import SubsamplingTest

test_cases = (UnetTest,SubsamplingTest, )

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith('.test_session')]
        suite.addTests(filtered_tests)
    print("suite")
    print(suite)
    return suite

def pytest_collection_modifyitems(session, config, items):
    print(items)
    items[:] = [item for item in items if item.name != 'test_session']
    print(items)