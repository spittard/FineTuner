import argparse
import json
import os
from typing import List, Dict, Any

# Import CreateDataSet from separate file
from CreateDataSet import CreateDataSet

# Global flag for ML dependencies
ML_DEPENDENCIES_AVAILABLE = False
COMPANY_MATCHER_AVAILABLE = False

def check_ml_dependencies():
    """Check if ML dependencies are available"""
    global ML_DEPENDENCIES_AVAILABLE
    
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        
        # Try to import Unsloth
        try:
            from unsloth import FastLanguageModel
            UNSLOTH_AVAILABLE = True
        except (ImportError, NotImplementedError):
            UNSLOTH_AVAILABLE = False
        
        ML_DEPENDENCIES_AVAILABLE = True
        return UNSLOTH_AVAILABLE
    except ImportError:
        ML_DEPENDENCIES_AVAILABLE = False
        return False

def check_company_matcher():
    """Check if CompanyMatcher dependencies are available"""
    global COMPANY_MATCHER_AVAILABLE
    
    try:
        from CompanyMatcher import CompanyMatcher
        COMPANY_MATCHER_AVAILABLE = True
        return True
    except ImportError:
        COMPANY_MATCHER_AVAILABLE = False
        return False

class FineTuner:
    def __init__(self,
                 model_name="microsoft/DialoGPT-small",
                 max_seq_length=2048,
                 dtype=None,
                 load_in_4bit=True,
                 device="cpu"):
        if not ML_DEPENDENCIES_AVAILABLE:
            raise ImportError("ML dependencies not available. Please install torch, transformers, and datasets.")
        
        import torch
        if dtype is None:
            dtype = torch.float32
            
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Check Unsloth availability
        try:
            from unsloth import FastLanguageModel
            self.using_unsloth = True
        except (ImportError, NotImplementedError):
            self.using_unsloth = False

    def load_model(self):
        if self.using_unsloth:
            import torch
            from transformers import AutoTokenizer
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit
            )
            self.model.to(self.device)
        else:
            # Fallback to standard transformers
            print("WARNING: Unsloth not available (requires NVIDIA/Intel GPU)")
            print("INFO: Falling back to standard transformers library")
            print("TIP: For better performance, consider using a GPU-enabled system")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
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
        Expects a list of dicts: [{"Company Name": "Company Name"}, ...]
        """
        if self.using_unsloth:
            import torch
            from transformers import AutoTokenizer
            from datasets import Dataset
            
            # For Unsloth, format as instruction-following data
            formatted_data = []
            for item in data:
                company_name = item.get("Company Name", "")
                formatted_data.append({
                    "prompt": "What is the company name?",
                    "completion": company_name
                })
            return Dataset.from_list(formatted_data)
        else:
            # Format data for standard transformers language model training
            formatted_data = []
            for item in data:
                company_name = item.get("Company Name", "")
                # Create a simple text format for training, limit length
                if len(company_name) > 100:  # Limit very long names
                    company_name = company_name[:100]
                text = f"Company Name: {company_name}"
                formatted_data.append({"text": text})
            
            # Create dataset
            dataset = Dataset.from_list(formatted_data)
            
            # Simple tokenization without labels - let the data collator handle everything
            def tokenize_function(examples):
                import torch
                from transformers import AutoTokenizer
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name) # Use self.model_name here
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=self.max_seq_length,
                    return_tensors=None
                )
            
            # Tokenize and remove original text column
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            return tokenized_dataset

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
            import torch
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from unsloth import FastLanguageModel
            
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
            
            # Dataset is already tokenized from prepare_dataset
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
                dataloader_pin_memory=False,  # Disable for Windows compatibility
            )
            
            # Use proper data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Not masked language modeling
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,  # Use the pre-tokenized dataset
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

    def predict(self, prompt, max_new_tokens=10):
        import torch
        from transformers import AutoTokenizer
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def match_companies(self, query: str, company_names: List[str], top_k: int = 10, matcher_model: str = 'all-MiniLM-L6-v2') -> List[Dict[str, Any]]:
        """
        Match a query against a list of company names using CompanyMatcher
        
        Args:
            query: Company name to search for
            company_names: List of company names to search in
            top_k: Number of top matches to return
            matcher_model: Sentence transformer model to use
            
        Returns:
            List of dictionaries with match information
        """
        if not check_company_matcher():
            print("ERROR: CompanyMatcher not available. Please install sentence-transformers: pip install sentence-transformers")
            return []
        
        try:
            from CompanyMatcher import CompanyMatcher
            
            # Initialize CompanyMatcher
            matcher = CompanyMatcher(model_name=matcher_model)
            
            # Perform matching
            matches = matcher.find_matches(query, company_names, top_k=top_k)
            
            return matches
            
        except Exception as e:
            print(f"Error in company matching: {e}")
            return []

def load_dataset_from_file(file_path):
    """Load dataset from JSON file with Company Name format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate data format
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of dictionaries")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dictionary")
        if "Company Name" not in item:
            raise ValueError(f"Item {i} missing 'Company Name' field")
        if not isinstance(item["Company Name"], str):
            raise ValueError(f"Item {i} 'Company Name' must be a string")
    
    print(f"OK: Loaded {len(data)} company name entries")
    return data

def main():
    parser = argparse.ArgumentParser(description='Fine-tune language models using Unsloth or transformers, create datasets from databases, or perform company name matching')
    parser.add_argument('--mode', choices=['train', 'predict', 'create-dataset', 'match', 'cache-info', 'clear-cache'], required=True,
                       help='Mode: train for fine-tuning, predict for inference, create-dataset for database extraction, match for company matching, cache-info for cache status, clear-cache to clear cache')
    
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
    
    # Dataset creation configuration
    parser.add_argument('--db-type', choices=['sqlite', 'mysql', 'postgresql', 'sqlserver'], default='sqlite',
                       help='Database type for dataset creation (default: sqlite)')
    parser.add_argument('--db-host', default='localhost', help='Database host (default: localhost)')
    parser.add_argument('--db-server', help='SQL Server instance name (for SQL Server connections)')
    parser.add_argument('--db-user', help='Database username')
    parser.add_argument('--db-password', help='Database password')
    parser.add_argument('--db-port', type=int, help='Database port')
    parser.add_argument('--db-driver', help='ODBC driver for SQL Server (e.g., "ODBC Driver 17 for SQL Server")')
    parser.add_argument('--trusted-connection', action='store_true', help='Use Windows Authentication for SQL Server (Trusted_Connection=yes)')
    parser.add_argument('--db-name', help='Database name')
    parser.add_argument('--table-name', help='Table name to extract from')
    parser.add_argument('--column-name', help='Column name containing company names')
    parser.add_argument('--output-file', help='Output JSON file path')
    parser.add_argument('--max-rows', type=int, help='Maximum number of rows to extract from database (default: all rows)')
    
    # Company matching configuration
    parser.add_argument('--query', help='Company name to search for in match mode')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top matches to return (default: 10)')
    parser.add_argument('--matcher-model', default='all-MiniLM-L6-v2', help='Sentence transformer model for company matching (default: all-MiniLM-L6-v2)')
    
    args = parser.parse_args()
    
    # Show example usage for match mode
    if args.mode == 'match' and not args.query:
        print("Company Matching Mode Examples:")
        print("  python FineTuner.py --mode match --dataset training_data.json --query 'Microsoft'")
        print("  python FineTuner.py --mode match --dataset training_data.json --query 'Apple Inc' --top-k 5")
        print("  python FineTuner.py --mode match --dataset training_data.json --query 'Google' --matcher-model 'all-MiniLM-L6-v2'")
        print()
        return
    
    if args.mode == 'create-dataset':
        # Dataset creation mode
        if not all([args.db_name, args.table_name, args.column_name, args.output_file]):
            print("Error: --db-name, --table-name, --column-name, and --output-file are required for create-dataset mode")
            return
        
        # Set default ports based on database type
        if args.db_type == 'mysql' and not args.db_port:
            args.db_port = 3306
        elif args.db_type == 'postgresql' and not args.db_port:
            args.db_port = 5432
        elif args.db_type == 'sqlserver' and not args.db_port:
            args.db_port = 1433
        
        # Create dataset from database
        with CreateDataSet(args.db_type) as dataset_creator:
            # Connect to database
            if args.db_type == 'sqlite':
                success = dataset_creator.connect(args.db_name)
            elif args.db_type == 'sqlserver':
                # Use server parameter if provided, otherwise fall back to host
                server = args.db_server if args.db_server else args.db_host
                driver = args.db_driver if args.db_driver else "ODBC Driver 17 for SQL Server"
                
                if args.trusted_connection:
                    # Windows Authentication - no username/password needed
                    success = dataset_creator.connect(
                        args.db_name,
                        server=server,
                        port=args.db_port,
                        driver=driver,
                        trusted_connection=True
                    )
                else:
                    # SQL Authentication - username and password required
                    if not all([args.db_user, args.db_password]):
                        print(f"Error: --db-user and --db-password are required for SQL Authentication")
                        print("TIP: Use --trusted-connection for Windows Authentication")
                        return
                    
                    success = dataset_creator.connect(
                        args.db_name,
                        server=server,
                        user=args.db_user,
                        password=args.db_password,
                        port=args.db_port,
                        driver=driver,
                        trusted_connection=False
                    )
            else:
                if not all([args.db_user, args.db_password]):
                    print(f"Error: --db-user and --db-password are required for {args.db_type}")
                    return
                success = dataset_creator.connect(
                    args.db_name,
                    host=args.db_host,
                    user=args.db_user,
                    password=args.db_password,
                    port=args.db_port
                )
            
            if not success:
                print("Failed to connect to database")
                return
            
            # Extract and save data
            success = dataset_creator.create_dataset(
                args.table_name,
                args.column_name,
                args.output_file,
                max_rows=args.max_rows
            )
            
            if success:
                print(f"OK: Dataset created successfully: {args.output_file}")
                print(f"TIP: You can now use this file for training: python FineTuner.py --mode train --dataset {args.output_file}")
            else:
                print("ERROR: Failed to create dataset")
        
        return
    
    # For other modes, check ML dependencies
    if args.mode in ['train', 'predict']:
        unsloth_available = check_ml_dependencies()
        
        if not ML_DEPENDENCIES_AVAILABLE:
            print("⚠️  ML dependencies not available (requires torch, transformers, and datasets)")
            print("📚 Cannot proceed with training or prediction")
            print("💡 For full ML features, please install: pip install torch transformers datasets")
            return
        
        if not unsloth_available:
            print("⚠️  Unsloth not available (requires NVIDIA/Intel GPU)")
            print("📚 Falling back to standard transformers library")
            print("💡 For better performance, consider using a GPU-enabled system")
            print()
    
    # Check CompanyMatcher availability for match mode
    if args.mode == 'match':
        if not check_company_matcher():
            print("⚠️  CompanyMatcher not available (requires sentence-transformers)")
            print("📚 Cannot proceed with company matching")
            print("💡 For company matching features, please install: pip install sentence-transformers")
            return
    
    # Initialize FineTuner for training/prediction modes
    if args.mode in ['train', 'predict']:
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
    
    elif args.mode == 'match':
        if not args.query:
            print("Error: --query is required for match mode")
            return
        
        if not args.dataset:
            print("Error: --dataset is required for match mode")
            return
        
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file {args.dataset} not found")
            return
        
        print(f"Loading company data from: {args.dataset}")
        data = load_dataset_from_file(args.dataset)
        
        # Extract company names from the dataset
        company_names = []
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
        if not company_names:
            print("Error: No company names found in dataset")
            return
        
        print(f"Building company matching index with {len(company_names)} companies...")
        
        # Initialize CompanyMatcher
        matcher = CompanyMatcher(model_name=args.matcher_model)
        matcher.build_index(company_names)
        
        print(f"Searching for companies matching: {args.query}")
        matches = matcher.match(args.query, top_k=args.top_k)
        
        print(f"\nTop {len(matches)} matches for '{args.query}':")
        print("-" * 60)
        
        for i, match in enumerate(matches, 1):
            score_percent = match['score'] * 100
            match_type = match.get('match_type', 'unknown')
            print(f"{i:2d}. {match['name']:<40} {score_percent:5.1f}% [{match_type}]")
        
        print("-" * 60)
        
        # Show explanation for top match if available
        if matches:
            top_match = matches[0]
            explanation = matcher.explain_match(args.query, top_match['name'])
            print(f"\nExplanation for top match '{top_match['name']}':")
            print(f"  Match type: {explanation.get('match_type', 'unknown')}")
            print(f"  Query tokens: {', '.join(explanation['query_tokens'])}")
            print(f"  Match tokens: {', '.join(explanation['match_tokens'])}")
            print(f"  Overlap: {', '.join(explanation['overlap'])}")
            print(f"  Overlap score: {explanation['overlap_score']:.3f}")
            
            # Show additional details for word overlap matches
            if explanation.get('match_type') == 'word_overlap' and 'overlap_words' in explanation:
                print(f"  Overlap words: {', '.join(explanation['overlap_words'])}")
    
    elif args.mode == 'cache-info':
        print("Company Matching Cache Information:")
        print("=" * 50)
        
        # Initialize CompanyMatcher to check cache
        matcher = CompanyMatcher(model_name=args.matcher_model)
        cache_info = matcher.get_cache_info()
        print(cache_info)
        
        if args.dataset and os.path.exists(args.dataset):
            print(f"\nDataset: {args.dataset}")
            data = load_dataset_from_file(args.dataset)
            company_names = []
            for item in data:
                if isinstance(item, dict) and "Company Name" in item:
                    company_names.append(item["Company Name"])
            
            if company_names:
                cache_key = matcher.get_cache_key(company_names)
                print(f"Cache key for this dataset: {cache_key}")
                
                # Check if this specific dataset is cached
                if matcher.load_from_cache(cache_key):
                    print("✓ This dataset is cached and ready to use")
                else:
                    print("⚠ This dataset is not cached - will build index on first use")
        else:
            print("\nNo dataset specified - use --dataset to check specific dataset cache status")
    
    elif args.mode == 'clear-cache':
        print("Company Matching Cache Management:")
        print("=" * 40)
        
        # Initialize CompanyMatcher
        matcher = CompanyMatcher(model_name=args.matcher_model)
        
        if args.dataset and os.path.exists(args.dataset):
            # Clear specific dataset cache
            data = load_dataset_from_file(args.dataset)
            company_names = []
            for item in data:
                if isinstance(item, dict) and "Company Name" in item:
                    company_names.append(item["Company Name"])
            
            if company_names:
                cache_key = matcher.get_cache_key(company_names)
                print(f"Clearing cache for dataset: {args.dataset}")
                print(f"Cache key: {cache_key}")
                matcher.clear_cache(cache_key)
            else:
                print("No company names found in dataset")
        else:
            # Clear all cache
            print("Clearing all company matching cache...")
            matcher.clear_cache()
        
        print("Cache cleared successfully!")

if __name__ == "__main__":
    main()