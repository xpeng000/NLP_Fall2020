"""Grading system based on unittest test cases."""

# Copyright 2020 Constantine Lignos and Ceasar Bautista
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import threading
import unittest
from functools import partial


def points(n: int):
    """Decorator used to a _points attribute to an object."""
    return partial(_add_points, n=n)


def _add_points(obj, n):
    old_points = getattr(obj, "_points", None)
    assert not old_points, "Object already has a points attribute"
    obj._points = n
    return obj


class Problem:
    """A Problem that can be graded.

    test_case should be an instance of unittest.TestCase

    test_weights should be a list of test_name-weight pairs.

    timeout should be the time to wait before killing a test, specified in
    seconds. By default, timeout is None and the test will wait until
    completion."""

    def __init__(self, test_case, timeout=None):
        self.test_weights = {
            test: test._points
            for test in test_case.__dict__.values()
            if getattr(test, "_points", None)
        }
        self.timeout = timeout
        self._test_case = test_case
        self._results = {}

    def run_tests(self):
        """Run tests and populate results."""
        """Print the results of the tests and the grade for the problem."""
        print(f"Grading {self._test_case.__name__}...")
        for test, weight in self.test_weights.items():
            print(f"Running {test.__name__}...")
            if test.__doc__:
                print(test.__doc__)
            result = self.run(test)
            if result.wasSuccessful():
                print(f"Points: {weight}/{weight}")
            else:
                print("-" * 70)
                # Get the error/failure
                try:
                    print(result.errors[0][1])
                except IndexError:
                    pass
                try:
                    print(result.failures[0][1])
                except IndexError:
                    pass
                print(f"Points: 0/{weight}")
                print("-" * 70)
            print()
        print("=" * 70)

    @property
    def max_grade(self):
        """The maximum grade possible for the problem."""
        return sum(self.test_weights.values())

    def run(self, test_method):
        """Return the result for the given test."""
        # TestCase supports getting a runner for an individual method by name
        test = self._test_case(test_method.__name__)
        result = unittest.TestResult()
        test_runner = threading.Thread(target=test.run, args=(result,))
        test_runner.daemon = True
        test_runner.start()

        test_runner.join(self.timeout)

        # if the test is still running, report a failure
        if test_runner.is_alive():
            # create a fake exception so we can use that for the failure
            try:
                raise TimeoutError(
                    f"Test {repr(test)} took longer than {self.timeout} seconds"
                )
            except TimeoutError:
                info = sys.exc_info()
            result.addFailure(test, info)

        self._results[test] = result
        return result

    @property
    def grade(self):
        """Grade earned for the problem."""
        return sum(
            weight
            for test, weight in self.test_weights.items()
            if self.run(test).wasSuccessful()
        )


class Grader(object):
    """A grader object."""

    def __init__(self, test_classes):
        self.problems = [Problem(test_class) for test_class in test_classes]

    def print_results(self):
        """Grade each problem and print out the final grade."""
        print("=" * 70)
        total, max_points = 0, 0
        for problem in self.problems:
            problem.run_tests()
            total += problem.grade
            max_points += problem.max_grade
        print(f"Total Grade: {total}/{max_points}")
