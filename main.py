"""
ValueVision: Main script for processing Amazon product data.

This script orchestrates the complete data processing pipeline including:
- Loading data from multiple Amazon product categories
- Sampling and preprocessing the data
- Creating train/test splits
- Generating visualizations
- Saving the final datasets

Usage:
    python main.py                    # Full pipeline (load and process all data)
    python main.py --use-existing     # Use existing pickle files if available
    python main.py --analysis-only   # Only run analysis on existing data
    python main.py --force-reload    # Force reload even with --use-existing
"""

import logging
import sys
import os
import argparse
from src.data.pipeline import DataPipeline

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
            python main.py                    # Full pipeline from scratch
            python main.py --use-existing     # Use existing data if available
            python main.py --analysis-only   # Quick analysis of existing data
            python main.py --use-existing --force-reload  # Force reload existing data
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
    
    return parser.parse_args()


def main():
    """Main function to run the complete data processing pipeline."""
    try:
        args = parse_arguments()
        
        # Validate arguments
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