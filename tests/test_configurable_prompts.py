
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before they are imported
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['unsloth'] = MagicMock()
sys.modules['datasets'] = MagicMock()

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.core.model import FineTuner
import finetuner.core.model
finetuner.core.model.ML_DEPENDENCIES_AVAILABLE = True

class TestConfigurablePrompts(unittest.TestCase):
    def test_default_prompts(self):
        ft = FineTuner()
        self.assertEqual(ft.instruction_prompt, "What is the company name?")
        self.assertEqual(ft.system_prompt, "You are a helpful assistant that identifies company names.")

    def test_custom_prompts(self):
        custom_instr = "Identify the business entity:"
        custom_sys = "You are a specialized NER agent."
        
        ft = FineTuner(instruction_prompt=custom_instr, system_prompt=custom_sys)
        self.assertEqual(ft.instruction_prompt, custom_instr)
        self.assertEqual(ft.system_prompt, custom_sys)

    @patch('datasets.Dataset.from_list')
    def test_dataset_preparation_standard(self, mock_from_list):
        ft = FineTuner(instruction_prompt="Target:")
        ft.using_unsloth = False
        
        data = [{"Company Name": "Test Corp"}]
        ft.prepare_dataset(data)
        
        args, _ = mock_from_list.call_args
        self.assertEqual(args[0][0]['text'], "Target: Test Corp")

    @patch('datasets.Dataset.from_list')
    def test_dataset_preparation_unsloth(self, mock_from_list):
        sys_prompt = "Custom System"
        instr_prompt = "Custom Input"
        ft = FineTuner(instruction_prompt=instr_prompt, system_prompt=sys_prompt)
        ft.using_unsloth = True
        
        data = [{"Company Name": "Test Corp"}]
        ft.prepare_dataset(data)
        
        args, _ = mock_from_list.call_args
        self.assertEqual(args[0][0]['instruction'], sys_prompt)
        self.assertEqual(args[0][0]['input'], instr_prompt)
        self.assertEqual(args[0][0]['output'], "Test Corp")

if __name__ == '__main__':
    unittest.main()
