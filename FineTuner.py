import argparse
import json
import os
import torch

# Try to import Unsloth, fall back to transformers if not available
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset

class FineTuner:
    def __init__(self,
                 model_name="microsoft/DialoGPT-small",
                 max_seq_length=2048,
                 dtype=torch.float32,
                 load_in_4bit=True,
                 device="cpu"):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.model = None
        self.tokenizer = None
        self.using_unsloth = UNSLOTH_AVAILABLE

    def load_model(self):
        if self.using_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit
            )
            self.model.to(self.device)
        else:
            # Fallback to standard transformers
            print("Unsloth not available, using standard transformers library")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, data):
        """
        Expects a list of dicts: [{"prompt": "...", "completion": "..."}, ...]
        """
        if self.using_unsloth:
            return Dataset.from_list(data)
        else:
            # Format data for standard transformers training
            formatted_data = []
            for item in data:
                # Combine prompt and completion with separator
                text = f"{item['prompt']} {item['completion']}"
                formatted_data.append({"text": text})
            return Dataset.from_list(formatted_data)

    def fine_tune(self,
                  dataset,
                  batch_size=2,
                  lr=2e-4,
                  epochs=3,
                  lora_r=8,
                  lora_alpha=16,
                  lora_dropout=0.05,
                  use_gradient_checkpointing=True,
                  output_dir="fine_tuned_model"):
        
        if self.using_unsloth:
            self.model = FastLanguageModel.finetune(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=dataset,
                max_seq_length=self.max_seq_length,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_gradient_checkpointing=use_gradient_checkpointing,
                output_dir=output_dir
            )
        else:
            # Standard transformers fine-tuning
            print("Using standard transformers fine-tuning (no LoRA)")
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                save_steps=500,
                save_total_limit=2,
                logging_steps=100,
                gradient_checkpointing=use_gradient_checkpointing,
                remove_unused_columns=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

    def predict(self, prompt, max_new_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_dataset_from_file(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(description='Fine-tune language models using Unsloth or transformers')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='Mode: train for fine-tuning, predict for inference')
    
    # Model configuration
    parser.add_argument('--model-name', default='microsoft/DialoGPT-small',
                       help='Model name to load (default: microsoft/DialoGPT-small)')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048)')
    parser.add_argument('--device', default='cpu',
                       help='Device to use: cpu or cuda (default: cpu)')
    parser.add_argument('--load-in-4bit', action='store_true', default=True,
                       help='Load model in 4-bit quantization (default: True)')
    
    # Training configuration
    parser.add_argument('--dataset', help='Path to JSON dataset file for training')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha (default: 16)')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout (default: 0.05)')
    parser.add_argument('--output-dir', default='fine_tuned_model',
                       help='Output directory for fine-tuned model (default: fine_tuned_model)')
    
    # Prediction configuration
    parser.add_argument('--prompt', help='Prompt text for prediction')
    parser.add_argument('--max-new-tokens', type=int, default=10,
                       help='Maximum new tokens to generate (default: 10)')
    
    args = parser.parse_args()
    
    # Check if Unsloth is available
    if not UNSLOTH_AVAILABLE:
        print("⚠️  Unsloth not available (requires NVIDIA/Intel GPU)")
        print("📚 Falling back to standard transformers library")
        print("💡 For better performance, consider using a GPU-enabled system")
        print()
    
    # Initialize FineTuner
    fine_tuner = FineTuner(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )
    
    if args.mode == 'train':
        if not args.dataset:
            print("Error: --dataset is required for training mode")
            return
        
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file {args.dataset} not found")
            return
        
        print(f"Loading model: {args.model_name}")
        fine_tuner.load_model()
        
        print(f"Loading dataset from: {args.dataset}")
        data = load_dataset_from_file(args.dataset)
        dataset = fine_tuner.prepare_dataset(data)
        
        print("Starting fine-tuning...")
        fine_tuner.fine_tune(
            dataset=dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            output_dir=args.output_dir
        )
        print(f"Fine-tuning completed! Model saved to: {args.output_dir}")
        
    elif args.mode == 'predict':
        if not args.prompt:
            print("Error: --prompt is required for prediction mode")
            return
        
        print(f"Loading model: {args.model_name}")
        fine_tuner.load_model()
        
        print(f"Generating response for prompt: {args.prompt}")
        response = fine_tuner.predict(args.prompt, args.max_new_tokens)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()