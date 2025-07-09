"""Test improvement"""
import logging

class TestImprovement:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply(self, context):
        return {'success': True}
