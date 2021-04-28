import re
import unittest
from perceptron import *
import subprocess
import timeout_decorator
import time

class PerceptronTestCase(unittest.TestCase):



    def test_perceptron(self):
        self.assertIsNotNone( Neuron(10) )


