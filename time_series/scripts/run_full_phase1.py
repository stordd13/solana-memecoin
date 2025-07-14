#!/usr/bin/env python3
# run_full_phase1.py
"""
Phase 1 Full Pipeline Runner

CEO Roadmap Implementation:
Orchestrates the complete Phase 1 pipeline from Day 1-2 through Day 9-10.
Provides end-to-end execution with proper dependency management and error handling.

Usage:
    python run_full_phase1.py --data-dir PATH [--output-dir PATH] [--n-tokens INT]
    
Interactive Mode:
    python run_full_phase1.py --interactive
    
Resume Mode:
    python run_full_phase1.py --resume --from-phase PHASE
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import time
import shutil

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import ResultsManager


class Phase1PipelineRunner:
    """
    Orchestrates the complete Phase 1 pipeline execution.
    Manages dependencies, error handling, and result tracking.
    """
    
    def __init__(self, output_dir: Path = None):
        self.results_manager = ResultsManager(output_dir or Path("../results"))
        self.scripts_dir = Path(__file__).parent
        self.pipeline_log = []
        
        # Define pipeline phases
        self.phases = {
            'day1_2': {
                'name': 'Baseline Assessment',
                'script': 'phase1_day1_2_baseline_assessment.py',
                'depends_on': [],
                'required_args': ['--data-dir', '--n-tokens'],
                'output_key': 'baseline_assessment'
            },
            'day3_4': {
                'name': 'Feature Standardization',
                'script': 'phase1_day3_4_feature_standardization.py',
                'depends_on': ['day1_2'],
                'required_args': ['--data-dir', '--baseline-results'],
                'output_key': 'feature_standardization'
            },
            'day5_6': {
                'name': 'K-Selection',
                'script': 'phase1_day5_6_k_selection.py',
                'depends_on': ['day3_4'],
                'required_args': ['--data-dir', '--standardization-results'],
                'output_key': 'k_selection'
            },
            'day7_8': {
                'name': 'Stability Testing',
                'script': 'phase1_day7_8_stability_testing.py',
                'depends_on': ['day5_6'],
                'required_args': ['--k-selection-results'],
                'output_key': 'stability_testing'
            },
            'day9_10': {
                'name': 'Archetype Characterization',
                'script': 'phase1_day9_10_archetype_characterization.py',
                'depends_on': ['day7_8'],
                'required_args': ['--stability-results'],
                'output_key': 'archetype_characterization'
            }
        }
        
        # Track execution state
        self.execution_state = {
            'started': False,
            'completed_phases': [],
            'failed_phases': [],
            'current_phase': None,
            'results_paths': {},
            'start_time': None,
            'end_time': None
        }
    
    def validate_environment(self) -> bool:
        """Validate that all required scripts and dependencies are available."""
        print("🔍 Validating environment...")
        
        # Check all script files exist
        missing_scripts = []
        for phase_key, phase_info in self.phases.items():
            script_path = self.scripts_dir / phase_info['script']
            if not script_path.exists():
                missing_scripts.append(phase_info['script'])
        
        if missing_scripts:
            print(f"❌ Missing required scripts: {missing_scripts}")
            return False
        
        # Check Python dependencies
        try:
            import numpy
            import polars
            import sklearn
            import gradio
            print("✅ All required Python packages available")
        except ImportError as e:
            print(f"❌ Missing Python package: {e}")
            return False
        
        return True
    
    def find_latest_result(self, phase_key: str) -> Optional[Path]:
        """Find the latest result file for a given phase."""
        phase_info = self.phases[phase_key]
        output_key = phase_info['output_key']
        
        # Map phases to their result directories
        result_dirs = {
            'baseline_assessment': 'phase1_day1_2_baseline',
            'feature_standardization': 'phase1_day3_4_features',
            'k_selection': 'phase1_day5_6_k_selection',
            'stability_testing': 'phase1_day7_8_stability',
            'archetype_characterization': 'phase1_day9_10_archetypes'
        }
        
        result_dir = self.results_manager.base_results_dir / result_dirs.get(output_key, output_key)
        if not result_dir.exists():
            return None
        
        # Find latest JSON result file
        json_files = list(result_dir.glob("*.json"))
        if not json_files:
            return None
        
        # Sort by modification time and return latest
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        return latest_file
    
    def run_phase(self, phase_key: str, args: Dict[str, str]) -> Tuple[bool, str]:
        """Execute a single phase of the pipeline."""
        phase_info = self.phases[phase_key]
        script_path = self.scripts_dir / phase_info['script']
        
        print(f"\n🚀 Starting {phase_info['name']} ({phase_key})")
        self.execution_state['current_phase'] = phase_key
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        
        # Add required arguments
        for arg_name in phase_info['required_args']:
            if arg_name in args:
                cmd.extend([arg_name, str(args[arg_name])])
        
        # Add output directory
        cmd.extend(['--output-dir', str(self.results_manager.base_results_dir)])
        
        # Log command
        cmd_str = ' '.join(cmd)
        print(f"📝 Command: {cmd_str}")
        self.pipeline_log.append(f"[{datetime.now().isoformat()}] {phase_key}: {cmd_str}")
        
        # Execute command
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            if result.returncode == 0:
                duration = end_time - start_time
                print(f"✅ {phase_info['name']} completed successfully in {duration:.1f}s")
                
                # Find and record result path
                result_path = self.find_latest_result(phase_key)
                if result_path:
                    self.execution_state['results_paths'][phase_key] = str(result_path)
                    print(f"📁 Results saved to: {result_path}")
                else:
                    print("⚠️ Warning: Could not locate result file")
                
                self.execution_state['completed_phases'].append(phase_key)
                self.pipeline_log.append(f"[{datetime.now().isoformat()}] {phase_key}: SUCCESS")
                return True, result.stdout
            else:
                print(f"❌ {phase_info['name']} failed with exit code {result.returncode}")
                print(f"Error output: {result.stderr}")
                self.execution_state['failed_phases'].append(phase_key)
                self.pipeline_log.append(f"[{datetime.now().isoformat()}] {phase_key}: FAILED - {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {phase_info['name']} timed out after 1 hour")
            self.execution_state['failed_phases'].append(phase_key)
            self.pipeline_log.append(f"[{datetime.now().isoformat()}] {phase_key}: TIMEOUT")
            return False, "Process timed out"
        except Exception as e:
            print(f"💥 {phase_info['name']} crashed with error: {e}")
            self.execution_state['failed_phases'].append(phase_key)
            self.pipeline_log.append(f"[{datetime.now().isoformat()}] {phase_key}: CRASHED - {str(e)}")
            return False, str(e)
    
    def check_dependencies(self, phase_key: str) -> bool:
        """Check if all dependencies for a phase are satisfied."""
        phase_info = self.phases[phase_key]
        
        for dependency in phase_info['depends_on']:
            if dependency not in self.execution_state['completed_phases']:
                print(f"❌ {phase_key} depends on {dependency} which hasn't completed")
                return False
        
        return True
    
    def build_phase_args(self, phase_key: str, base_args: Dict[str, str]) -> Dict[str, str]:
        """Build arguments for a specific phase based on pipeline state."""
        args = base_args.copy()
        
        # Add dependency result paths
        if phase_key == 'day3_4' and 'day1_2' in self.execution_state['results_paths']:
            args['--baseline-results'] = self.execution_state['results_paths']['day1_2']
        elif phase_key == 'day5_6' and 'day3_4' in self.execution_state['results_paths']:
            args['--standardization-results'] = self.execution_state['results_paths']['day3_4']
        elif phase_key == 'day7_8' and 'day5_6' in self.execution_state['results_paths']:
            args['--k-selection-results'] = self.execution_state['results_paths']['day5_6']
        elif phase_key == 'day9_10' and 'day7_8' in self.execution_state['results_paths']:
            args['--stability-results'] = self.execution_state['results_paths']['day7_8']
        
        return args
    
    def run_full_pipeline(self, data_dir: Path, n_tokens: int = 1000, 
                         from_phase: str = None) -> Dict:
        """Run the complete Phase 1 pipeline."""
        print(f"🚀 Starting Phase 1 Full Pipeline")
        print(f"📁 Data directory: {data_dir}")
        print(f"🔢 Tokens to process: {n_tokens}")
        
        if from_phase:
            print(f"🔄 Resuming from phase: {from_phase}")
        
        # Validate environment
        if not self.validate_environment():
            raise RuntimeError("Environment validation failed")
        
        # Initialize execution state
        self.execution_state['started'] = True
        self.execution_state['start_time'] = datetime.now()
        
        # Base arguments for all phases
        base_args = {
            '--data-dir': str(data_dir),
            '--n-tokens': str(n_tokens)
        }
        
        # Determine starting phase
        phases_to_run = list(self.phases.keys())
        if from_phase:
            if from_phase not in self.phases:
                raise ValueError(f"Unknown phase: {from_phase}")
            start_idx = phases_to_run.index(from_phase)
            phases_to_run = phases_to_run[start_idx:]
            
            # When resuming, try to find existing results for dependencies
            for phase_key in list(self.phases.keys())[:start_idx]:
                result_path = self.find_latest_result(phase_key)
                if result_path:
                    self.execution_state['results_paths'][phase_key] = str(result_path)
                    self.execution_state['completed_phases'].append(phase_key)
                    print(f"📂 Found existing result for {phase_key}: {result_path}")
        
        # Execute phases in order
        for phase_key in phases_to_run:
            # Check dependencies
            if not self.check_dependencies(phase_key):
                print(f"❌ Stopping pipeline due to unmet dependencies for {phase_key}")
                break
            
            # Build phase-specific arguments
            phase_args = self.build_phase_args(phase_key, base_args)
            
            # Run phase
            success, output = self.run_phase(phase_key, phase_args)
            
            if not success:
                print(f"❌ Pipeline failed at {phase_key}")
                break
        
        # Finalize execution state
        self.execution_state['end_time'] = datetime.now()
        self.execution_state['current_phase'] = None
        
        # Generate summary
        summary = self.generate_pipeline_summary()
        
        # Save pipeline results
        pipeline_results = {
            'execution_state': self.execution_state,
            'pipeline_log': self.pipeline_log,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to results directory
        pipeline_results_path = self.results_manager.base_results_dir / "phase1_full_pipeline_results.json"
        with open(pipeline_results_path, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\n📊 Pipeline execution complete!")
        print(f"📁 Pipeline results saved to: {pipeline_results_path}")
        
        return pipeline_results
    
    def generate_pipeline_summary(self) -> Dict:
        """Generate a summary of pipeline execution."""
        total_phases = len(self.phases)
        completed_phases = len(self.execution_state['completed_phases'])
        failed_phases = len(self.execution_state['failed_phases'])
        
        if self.execution_state['start_time'] and self.execution_state['end_time']:
            duration = self.execution_state['end_time'] - self.execution_state['start_time']
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "Unknown"
        
        success_rate = completed_phases / total_phases if total_phases > 0 else 0
        
        summary = {
            'total_phases': total_phases,
            'completed_phases': completed_phases,
            'failed_phases': failed_phases,
            'success_rate': success_rate,
            'duration': duration_str,
            'status': 'success' if completed_phases == total_phases else 'partial' if completed_phases > 0 else 'failed'
        }
        
        return summary
    
    def print_pipeline_summary(self):
        """Print a formatted summary of pipeline execution."""
        summary = self.generate_pipeline_summary()
        
        print(f"\n" + "="*60)
        print(f"📊 PHASE 1 PIPELINE SUMMARY")
        print(f"="*60)
        
        print(f"🎯 Status: {summary['status'].upper()}")
        print(f"⏱️  Duration: {summary['duration']}")
        print(f"✅ Completed: {summary['completed_phases']}/{summary['total_phases']} phases")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        
        if self.execution_state['completed_phases']:
            print(f"\n✅ Completed Phases:")
            for phase in self.execution_state['completed_phases']:
                phase_name = self.phases[phase]['name']
                result_path = self.execution_state['results_paths'].get(phase, 'N/A')
                print(f"  • {phase_name} ({phase}): {result_path}")
        
        if self.execution_state['failed_phases']:
            print(f"\n❌ Failed Phases:")
            for phase in self.execution_state['failed_phases']:
                phase_name = self.phases[phase]['name']
                print(f"  • {phase_name} ({phase})")
        
        print(f"="*60)


def create_gradio_interface():
    """Create interactive Gradio interface for full pipeline execution."""
    
    def run_interactive_pipeline(data_dir_str: str, n_tokens: int, from_phase: str):
        """Run pipeline with Gradio inputs."""
        try:
            data_dir = Path(data_dir_str)
            if not data_dir.exists():
                return f"❌ Data directory not found: {data_dir_str}", None
            
            # Initialize pipeline runner
            runner = Phase1PipelineRunner()
            
            # Run pipeline
            start_phase = from_phase if from_phase != "Start from beginning" else None
            results = runner.run_full_pipeline(data_dir, n_tokens, start_phase)
            
            # Generate summary
            summary = results['summary']
            
            summary_text = f"""
            ## 🚀 Phase 1 Pipeline Execution Complete!
            
            **Status**: {summary['status'].upper()}
            **Duration**: {summary['duration']}
            **Success Rate**: {summary['success_rate']:.1%}
            **Completed Phases**: {summary['completed_phases']}/{summary['total_phases']}
            
            **Phase Results**:
            """
            
            for phase in runner.execution_state['completed_phases']:
                phase_name = runner.phases[phase]['name']
                result_path = runner.execution_state['results_paths'].get(phase, 'N/A')
                summary_text += f"\n✅ **{phase_name}** ({phase}): Results saved"
            
            for phase in runner.execution_state['failed_phases']:
                phase_name = runner.phases[phase]['name']
                summary_text += f"\n❌ **{phase_name}** ({phase}): Failed"
            
            if summary['status'] == 'success':
                summary_text += f"\n\n🎉 **All phases completed successfully!**"
                summary_text += f"\n📁 **Complete results available in**: {runner.results_manager.base_results_dir}"
            elif summary['status'] == 'partial':
                summary_text += f"\n\n⚠️ **Pipeline partially completed**"
                summary_text += f"\n🔄 **Use resume mode to continue from last successful phase**"
            else:
                summary_text += f"\n\n❌ **Pipeline failed**"
                summary_text += f"\n🔍 **Check logs for error details**"
            
            return summary_text, None
            
        except Exception as e:
            error_msg = f"## ❌ Error During Pipeline Execution\n\n```\n{str(e)}\n```"
            return error_msg, None
    
    import gradio as gr
    
    interface = gr.Interface(
        fn=run_interactive_pipeline,
        inputs=[
            gr.Textbox(
                value="/Users/brunostordeur/Docs/GitHub/Solana/memecoin2/data/processed",
                label="Data Directory",
                placeholder="Path to processed data directory"
            ),
            gr.Number(
                value=1000,
                label="Number of Tokens",
                precision=0
            ),
            gr.Dropdown(
                choices=["Start from beginning", "day1_2", "day3_4", "day5_6", "day7_8", "day9_10"],
                value="Start from beginning",
                label="Resume from Phase"
            )
        ],
        outputs=[
            gr.Markdown(label="Pipeline Execution Results"),
            gr.Plot(label="Execution Timeline")
        ],
        title="🚀 Phase 1 Full Pipeline Runner",
        description="Execute the complete Phase 1 pipeline from Day 1-2 through Day 9-10"
    )
    
    return interface


def main():
    """Main entry point with CLI and interactive modes."""
    parser = argparse.ArgumentParser(description="Phase 1 Full Pipeline Runner")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("../../data/processed"),
                       help="Path to processed data directory")
    parser.add_argument("--n-tokens", type=int, default=1000,
                       help="Number of tokens to process")
    parser.add_argument("--output-dir", type=Path, default=Path("../results"),
                       help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from a specific phase")
    parser.add_argument("--from-phase", type=str, 
                       choices=['day1_2', 'day3_4', 'day5_6', 'day7_8', 'day9_10'],
                       help="Phase to resume from (requires --resume)")
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive Gradio interface")
    parser.add_argument("--share", action="store_true",
                       help="Create public Gradio link")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("🚀 Launching interactive Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(share=args.share)
    else:
        # CLI mode
        if args.resume and not args.from_phase:
            print("❌ Error: --from-phase is required when using --resume")
            return
        
        runner = Phase1PipelineRunner(args.output_dir)
        
        try:
            results = runner.run_full_pipeline(
                args.data_dir, 
                args.n_tokens, 
                args.from_phase if args.resume else None
            )
            runner.print_pipeline_summary()
            
            if results['summary']['status'] == 'success':
                print("\n🎉 Pipeline completed successfully!")
                exit(0)
            else:
                print(f"\n⚠️ Pipeline completed with status: {results['summary']['status']}")
                exit(1)
                
        except Exception as e:
            print(f"💥 Pipeline execution failed: {e}")
            exit(1)


if __name__ == "__main__":
    main()