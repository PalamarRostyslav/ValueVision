"""
ValueVision: Main script for processing Amazon product data.

This script orchestrates the complete data processing pipeline including:
- Loading data from multiple Amazon product categories
- Sampling and preprocessing the data
- Creating train/test splits
- Generating visualizations
- Saving the final datasets
- Testing and evaluating models
- Fine-tuning model parameters

Usage:
    python main.py                         # Full pipeline (load and process all data)
    python main.py --use-existing          # Use existing pickle files if available
    python main.py --analysis-only        # Only run analysis on existing data
    python main.py --force-reload         # Force reload even with --use-existing
    python main.py --test                 # Run model testing on existing data
    python main.py --fine-tune random-seed   # Run random seed fine-tuning
    python main.py --fine-tune feature-based # Run feature-based fine-tuning
    python main.py --fine-tune random-forest # Run Random Forest with Word2Vec fine-tuning
    python main.py --fine-tune openai        # Run OpenAI GPT fine-tuning
"""

import logging
import sys
import os
import argparse
import pickle
from src.testing.strategies.random_seed import FineTuningRandomSeed
from src.testing.strategies.feature_based import FineTuningWithFeatures
from src.testing.strategies.random_forest import FineTuningRandomForest
from src.testing.strategies.openai_strategy import FineTuningOpenAI
from src.testing.tester import ModelTester
from src.data.pipeline import DataPipeline
from src.ai_models.model_manager import ModelManager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ValueVision Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python main.py                              # Full pipeline from scratch
            python main.py --use-existing               # Use existing data if available
            python main.py --analysis-only             # Quick analysis of existing data
            python main.py --test                      # Test models on existing data
            python main.py --fine-tune openai          # Fine-tune with OpenAI (reuses existing models)
            python main.py --fine-tune openai --force-new-model  # Force new model training
            python main.py --fine-tune openai --model-id ft:gpt-4o-mini:example  # Use specific model
            python main.py --list-models               # List all saved models
            python main.py --cleanup-models 30         # Clean up models older than 30 days
                """
        )
    
    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Use existing pickle files if available instead of reprocessing data'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Only run analysis and visualization on existing data (no processing)'
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload data even when using existing files'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specify which datasets to process (e.g., --datasets Electronics Appliances)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run model testing and evaluation on existing data'
    )
    
    parser.add_argument(
        '--fine-tune',
        choices=['random-seed', 'feature-based', 'random-forest', 'openai'],
        help='Run fine-tuning optimization with specified strategy'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        help='Use specific fine-tuned model ID (for OpenAI fine-tuning)'
    )
    
    parser.add_argument(
        '--force-new-model',
        action='store_true',
        help='Force training new model even if existing model found'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all saved fine-tuned models'
    )
    
    parser.add_argument(
        '--cleanup-models',
        type=int,
        metavar='DAYS',
        help='Clean up models older than specified days (keeps latest 3 per provider)'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        default=250,
        help='Number of items to use for testing (default: 250)'
    )
    
    return parser.parse_args()


def run_testing_mode(args):
    """Run testing and evaluation mode."""
    
    # Load test data
    try:
        with open('output/test.pkl', 'rb') as f:
            test_data = pickle.load(f)
        logger.info(f"Loaded {len(test_data)} test items")
    except FileNotFoundError:
        logger.error("Test data not found. Run the pipeline first to generate test data.")
        sys.exit(1)
    
    def simple_predictor(item):
        """Simple example predictor based on title length and features."""
        base_price = len(item.title) * 2.5
        if hasattr(item, 'features') and item.features:
            feature_boost = len(item.features) * 10
            base_price += feature_boost
        return max(base_price, 10.0)
    
    # Run testing
    tester = ModelTester(simple_predictor, data=test_data, size=args.test_size)
    tester.run()


def run_model_management(args):
    """Run model management commands."""
    model_manager = ModelManager()
    
    if args.list_models:
        print("\n" + "="*60)
        print("FINE-TUNED MODELS REGISTRY")
        print("="*60)
        model_manager.print_models_summary()
        return
    
    if args.cleanup_models:
        days = args.cleanup_models
        print(f"\nCleaning up models older than {days} days...")
        deleted_count = model_manager.cleanup_old_models(days=days)
        if deleted_count > 0:
            print(f"✓ Cleaned up {deleted_count} old models")
        else:
            print("No old models to clean up")
        return


def run_fine_tuning_mode(args):
    """Run fine-tuning optimization mode."""
    
    # Load data
    try:
        with open('output/train.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('output/test.pkl', 'rb') as f:
            test_data = pickle.load(f)
        logger.info(f"Loaded {len(train_data)} training items and {len(test_data)} test items")
    except FileNotFoundError:
        logger.error("Training/test data not found. Run the pipeline first to generate data.")
        sys.exit(1)
    
    # Run fine-tuning with proper training/validation split
    if args.fine_tune == 'random-seed':
        optimizer = FineTuningRandomSeed(iterations=15)
    elif args.fine_tune == 'feature-based':
        optimizer = FineTuningWithFeatures()
    elif args.fine_tune == 'random-forest':
        optimizer = FineTuningRandomForest()
    elif args.fine_tune == 'openai':
        # Create OpenAI strategy with new options
        use_existing = not args.force_new_model
        optimizer = FineTuningOpenAI(
            use_existing=use_existing,
            model_id=args.model_id
        )
    else:
        logger.error("Unknown fine-tuning strategy")
        return
    
    optimized_predictor = optimizer.optimize_predictor(
        train_data[:700],
        test_data[:200]
    )
    
    # Test the optimized predictor on the full test set
    print(f"\n{'-'*50}")
    print("Testing optimized predictor...")
    tester = ModelTester(optimized_predictor, data=test_data, size=args.test_size)
    tester.run()


def main():
    """Main function to run the complete data processing pipeline."""
    try:
        args = parse_arguments()
        
        # Handle model management commands first
        if args.list_models or args.cleanup_models:
            run_model_management(args)
            return
        
        # Handle fine-tuning mode
        if args.fine_tune:
            run_fine_tuning_mode(args)
            return
        
        # Validate arguments for normal pipeline
        if args.analysis_only and args.force_reload:
            logger.warning("--force-reload is ignored when using --analysis-only")
        
        if args.analysis_only:
            # Run analysis only mode
            logger.info("Running in analysis-only mode...")
            pipeline = DataPipeline(use_existing_data=True)
            pipeline.run_analysis_only()
            
        else:
            # Run the full pipeline or use existing data
            use_existing = args.use_existing
            force_reload = args.force_reload
            
            if use_existing:
                logger.info("Pipeline will use existing data if available...")
            else:
                logger.info("Pipeline will process data from scratch...")
            
            pipeline = DataPipeline(use_existing_data=use_existing)
            pipeline.run_full_pipeline(
                dataset_names=args.datasets,
                force_reload=force_reload
            )
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.info("Tip: Run without --analysis-only or --use-existing to generate the data first")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main()